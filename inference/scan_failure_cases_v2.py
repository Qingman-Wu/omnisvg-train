"""
V2: Enhanced scan with stricter quality filters.
Find cases where:
  1. Baseline SVG has many paths (complex prompt) but collapses in the second half
  2. HVM SVG also has many paths (not trivially short) and stays diverse
  3. The prompt describes something complex enough to be interesting for the paper
"""

import os
import re
import json
import hashlib
from collections import Counter
from pathlib import Path


def extract_paths(svg_text: str) -> list[str]:
    return re.findall(r'<path[^>]*d="([^"]*)"[^>]*/?>',  svg_text) or \
           re.findall(r'd="([^"]*)"', svg_text)


def extract_path_elements(svg_text: str) -> list[str]:
    return re.findall(r'<path[^>]*?/?>',  svg_text)


def path_fingerprint(d_attr: str) -> str:
    coords = re.findall(r'[\d.]+', d_attr)
    return hashlib.md5(' '.join(coords[:20]).encode()).hexdigest()


def compute_metrics(svg_text: str) -> dict:
    paths = extract_path_elements(svg_text)
    n = len(paths)
    if n < 2:
        return {'n_paths': n, 'rep_ratio': 0.0, 'unique_ratio': 1.0,
                'late_rep_ratio': 0.0, 'first_half_unique': 1.0, 'second_half_unique': 1.0,
                'early_diverse': True, 'late_collapse': False, 'n_unique_colors': 1,
                'second_half_color_div': 1.0}

    d_attrs = extract_paths(svg_text)
    fps = [path_fingerprint(d) for d in d_attrs]

    unique_fps = len(set(fps))
    unique_ratio = unique_fps / n

    fp_counts = Counter(fps)
    most_common_count = fp_counts.most_common(1)[0][1]
    rep_ratio = most_common_count / n

    mid = n // 2
    first_half_fps = fps[:mid]
    second_half_fps = fps[mid:]

    first_unique = len(set(first_half_fps)) / max(len(first_half_fps), 1)
    second_unique = len(set(second_half_fps)) / max(len(second_half_fps), 1)

    early_diverse = first_unique > 0.4
    late_collapse = second_unique < 0.3

    late_fp_counts = Counter(second_half_fps)
    late_most_common = late_fp_counts.most_common(1)[0][1] if late_fp_counts else 0
    late_rep_ratio = late_most_common / max(len(second_half_fps), 1)

    colors = re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg_text)
    n_unique_colors = len(set(colors))
    mid_c = len(colors) // 2
    second_half_colors = colors[mid_c:]
    second_half_color_div = len(set(second_half_colors)) / max(len(second_half_colors), 1)

    return {
        'n_paths': n,
        'rep_ratio': round(rep_ratio, 3),
        'unique_ratio': round(unique_ratio, 3),
        'first_half_unique': round(first_unique, 3),
        'second_half_unique': round(second_unique, 3),
        'late_rep_ratio': round(late_rep_ratio, 3),
        'early_diverse': early_diverse,
        'late_collapse': late_collapse,
        'n_unique_colors': n_unique_colors,
        'second_half_color_div': round(second_half_color_div, 3),
    }


def score_candidate(svg_path: str) -> dict | None:
    if not os.path.exists(svg_path):
        return None
    with open(svg_path, 'r') as f:
        svg_text = f.read()
    return compute_metrics(svg_text)


def main():
    base_dir = Path('/mnt/data2/wuqingman/omnisvg-train/inference_results/omnisvg_baseline')
    hvm_dir = Path('/mnt/data2/wuqingman/omnisvg-train/inference_results/s9_full25w_top3part_12slot_nogist_edr_parttag_nozoom_last4_test')

    txt_files = sorted(base_dir.glob('sample_*.txt'))
    sample_ids = [f.stem.replace('sample_', '') for f in txt_files]

    results = []

    for sid in sample_ids:
        txt_path = base_dir / f'sample_{sid}.txt'
        if not txt_path.exists():
            continue
        prompt = txt_path.read_text().strip()

        base_scores = []
        for ci in range(5):
            svg_path = base_dir / f'sample_{sid}_base_c{ci}.svg'
            s = score_candidate(str(svg_path))
            if s:
                base_scores.append((ci, s))

        hvm_scores = []
        for ci in range(5):
            svg_path = hvm_dir / f'sample_{sid}_hvm_c{ci}.svg'
            s = score_candidate(str(svg_path))
            if s:
                hvm_scores.append((ci, s))

        if not base_scores or not hvm_scores:
            continue

        base_collapse_cands = [(ci, s) for ci, s in base_scores
                                if s['n_paths'] >= 15 and s['rep_ratio'] > 0.35]
        base_early_good_late_bad = [(ci, s) for ci, s in base_scores
                                     if s['n_paths'] >= 15
                                     and s['early_diverse']
                                     and s['late_collapse']]

        hvm_good_cands = [(ci, s) for ci, s in hvm_scores
                           if s['n_paths'] >= 8
                           and s['rep_ratio'] < 0.25
                           and s['unique_ratio'] > 0.6]

        if not hvm_good_cands:
            continue

        has_ideal = len(base_early_good_late_bad) > 0
        has_collapse = len(base_collapse_cands) > 0

        if not has_ideal and not has_collapse:
            continue

        worst_base_ci, worst_base = max(
            base_collapse_cands if base_collapse_cands else base_early_good_late_bad,
            key=lambda x: x[1]['rep_ratio']
        )
        best_hvm_ci, best_hvm = min(hvm_good_cands, key=lambda x: x[1]['rep_ratio'])

        avg_base_rep = sum(s['rep_ratio'] for _, s in base_scores) / len(base_scores)
        avg_hvm_rep = sum(s['rep_ratio'] for _, s in hvm_scores) / len(hvm_scores)
        rep_improvement = avg_base_rep - avg_hvm_rep

        if rep_improvement < 0.08:
            continue

        gt_svg_path = base_dir / f'sample_{sid}_gt.svg'
        gt_paths = 0
        if gt_svg_path.exists():
            gt_text = gt_svg_path.read_text()
            gt_paths = len(extract_path_elements(gt_text))

        results.append({
            'sample_id': sid,
            'prompt': prompt,
            'is_ideal': has_ideal,
            'gt_paths': gt_paths,
            'avg_base_rep': round(avg_base_rep, 3),
            'avg_hvm_rep': round(avg_hvm_rep, 3),
            'rep_improvement': round(rep_improvement, 3),
            'worst_base_ci': worst_base_ci,
            'worst_base': {
                'n_paths': worst_base['n_paths'],
                'rep_ratio': worst_base['rep_ratio'],
                'late_rep_ratio': worst_base['late_rep_ratio'],
                'first_half_unique': worst_base['first_half_unique'],
                'second_half_unique': worst_base['second_half_unique'],
                'early_diverse': worst_base['early_diverse'],
                'late_collapse': worst_base['late_collapse'],
                'n_unique_colors': worst_base['n_unique_colors'],
            },
            'best_hvm_ci': best_hvm_ci,
            'best_hvm': {
                'n_paths': best_hvm['n_paths'],
                'rep_ratio': best_hvm['rep_ratio'],
                'unique_ratio': best_hvm['unique_ratio'],
                'n_unique_colors': best_hvm['n_unique_colors'],
            },
        })

    results.sort(key=lambda x: (
        -int(x['is_ideal']),
        -x['rep_improvement'],
        -x['worst_base']['n_paths'],
    ))

    print(f"\n{'='*90}")
    print(f"V2 SCAN: {len(results)} quality failure cases (base complex + collapses, HVM complex + good)")
    print(f"{'='*90}\n")

    ideal = [r for r in results if r['is_ideal']]
    good = [r for r in results if not r['is_ideal']]
    print(f"IDEAL (base starts diverse, then collapses): {len(ideal)}")
    print(f"GOOD (base high repetition, HVM low repetition): {len(good)}")

    for i, r in enumerate(results[:20]):
        tag = "IDEAL" if r['is_ideal'] else "GOOD "
        print(f"\n[{tag}] #{i+1}  sample_{r['sample_id']}   (GT: {r['gt_paths']} paths)")
        print(f"  Prompt: {r['prompt'][:120]}")
        print(f"  Base (c{r['worst_base_ci']}): {r['worst_base']['n_paths']} paths, "
              f"rep={r['worst_base']['rep_ratio']:.3f}, "
              f"1st_uniq={r['worst_base']['first_half_unique']:.3f}, "
              f"2nd_uniq={r['worst_base']['second_half_unique']:.3f}, "
              f"late_rep={r['worst_base']['late_rep_ratio']:.3f}, "
              f"colors={r['worst_base']['n_unique_colors']}")
        print(f"  HVM  (c{r['best_hvm_ci']}): {r['best_hvm']['n_paths']} paths, "
              f"rep={r['best_hvm']['rep_ratio']:.3f}, "
              f"uniq={r['best_hvm']['unique_ratio']:.3f}, "
              f"colors={r['best_hvm']['n_unique_colors']}")
        print(f"  Avg rep:  base={r['avg_base_rep']:.3f}  hvm={r['avg_hvm_rep']:.3f}  delta={r['rep_improvement']:.3f}")

    out_path = '/mnt/data2/wuqingman/omnisvg-train/inference/failure_case_report_v2.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {out_path}")


if __name__ == '__main__':
    main()

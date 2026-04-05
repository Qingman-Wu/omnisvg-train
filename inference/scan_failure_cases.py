"""
Scan paired OmniSVG-baseline vs HVM results to find typical failure cases where:
  - Baseline starts well but collapses into repetition / structural drift
  - HVM produces significantly better output

Detection heuristics:
  1. Repetition collapse: measure path-level duplication ratio in the SVG
  2. Early-good-then-bad: compare diversity in the first half vs second half of paths
  3. HVM quality: HVM should have low repetition and higher path diversity
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


def compute_repetition_metrics(svg_text: str) -> dict:
    paths = extract_path_elements(svg_text)
    n = len(paths)
    if n < 4:
        return {'n_paths': n, 'rep_ratio': 0.0, 'unique_ratio': 1.0,
                'late_rep_ratio': 0.0, 'early_diverse': True, 'late_collapse': False}

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

    return {
        'n_paths': n,
        'rep_ratio': round(rep_ratio, 3),
        'unique_ratio': round(unique_ratio, 3),
        'first_half_unique': round(first_unique, 3),
        'second_half_unique': round(second_unique, 3),
        'late_rep_ratio': round(late_rep_ratio, 3),
        'early_diverse': early_diverse,
        'late_collapse': late_collapse,
    }


def analyze_color_diversity(svg_text: str) -> dict:
    colors = re.findall(r'fill="(#[0-9a-fA-F]{6})"', svg_text)
    n = len(colors)
    if n < 2:
        return {'n_colors': n, 'unique_colors': n, 'color_diversity': 1.0}
    unique = len(set(colors))
    mid = n // 2
    first_colors = len(set(colors[:mid])) / max(mid, 1)
    second_colors = len(set(colors[mid:])) / max(n - mid, 1)
    return {
        'n_colors': n,
        'unique_colors': unique,
        'color_diversity': round(unique / n, 3),
        'first_half_color_div': round(first_colors, 3),
        'second_half_color_div': round(second_colors, 3),
    }


def score_candidate(svg_path: str) -> dict | None:
    if not os.path.exists(svg_path):
        return None
    with open(svg_path, 'r') as f:
        svg_text = f.read()
    rep = compute_repetition_metrics(svg_text)
    col = analyze_color_diversity(svg_text)
    return {**rep, **col, 'file_lines': svg_text.count('\n') + 1}


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
                base_scores.append(s)

        hvm_scores = []
        for ci in range(5):
            svg_path = hvm_dir / f'sample_{sid}_hvm_c{ci}.svg'
            s = score_candidate(str(svg_path))
            if s:
                hvm_scores.append(s)

        if not base_scores or not hvm_scores:
            continue

        def pick_best_and_worst(scores):
            by_rep = sorted(scores, key=lambda x: x['rep_ratio'])
            return by_rep[0], by_rep[-1]

        best_base, worst_base = pick_best_and_worst(base_scores)
        best_hvm, worst_hvm = pick_best_and_worst(hvm_scores)

        base_has_collapse = any(
            s['late_collapse'] and s['n_paths'] >= 10
            for s in base_scores
        )
        base_high_rep = any(s['rep_ratio'] > 0.4 and s['n_paths'] >= 8 for s in base_scores)
        base_starts_good = any(s['early_diverse'] and s['late_collapse'] for s in base_scores)

        hvm_no_collapse = all(not s['late_collapse'] for s in hvm_scores)
        hvm_low_rep = all(s['rep_ratio'] < 0.3 for s in hvm_scores)

        avg_base_rep = sum(s['rep_ratio'] for s in base_scores) / len(base_scores)
        avg_hvm_rep = sum(s['rep_ratio'] for s in hvm_scores) / len(hvm_scores)
        rep_improvement = avg_base_rep - avg_hvm_rep

        avg_base_unique = sum(s['unique_ratio'] for s in base_scores) / len(base_scores)
        avg_hvm_unique = sum(s['unique_ratio'] for s in hvm_scores) / len(hvm_scores)

        is_ideal_case = (
            base_starts_good and
            base_has_collapse and
            hvm_no_collapse and
            rep_improvement > 0.15
        )

        is_good_case = (
            base_high_rep and
            hvm_low_rep and
            rep_improvement > 0.1
        )

        if is_ideal_case or is_good_case:
            results.append({
                'sample_id': sid,
                'prompt': prompt,
                'is_ideal': is_ideal_case,
                'avg_base_rep': round(avg_base_rep, 3),
                'avg_hvm_rep': round(avg_hvm_rep, 3),
                'rep_improvement': round(rep_improvement, 3),
                'avg_base_unique': round(avg_base_unique, 3),
                'avg_hvm_unique': round(avg_hvm_unique, 3),
                'worst_base': {
                    'rep_ratio': worst_base['rep_ratio'],
                    'n_paths': worst_base['n_paths'],
                    'late_rep_ratio': worst_base['late_rep_ratio'],
                    'early_diverse': worst_base['early_diverse'],
                    'late_collapse': worst_base['late_collapse'],
                },
                'best_hvm': {
                    'rep_ratio': best_hvm['rep_ratio'],
                    'n_paths': best_hvm['n_paths'],
                    'unique_ratio': best_hvm['unique_ratio'],
                },
            })

    results.sort(key=lambda x: (-int(x['is_ideal']), -x['rep_improvement']))

    print(f"\n{'='*80}")
    print(f"SCAN COMPLETE: {len(results)} failure cases found out of {len(sample_ids)} samples")
    print(f"{'='*80}\n")

    ideal_cases = [r for r in results if r['is_ideal']]
    good_cases = [r for r in results if not r['is_ideal']]

    print(f"IDEAL cases (baseline starts good then collapses, HVM stays good): {len(ideal_cases)}")
    print(f"GOOD cases (baseline high repetition, HVM low repetition): {len(good_cases)}")
    print()

    for i, r in enumerate(results[:30]):
        tag = "IDEAL" if r['is_ideal'] else "GOOD"
        print(f"[{tag}] #{i+1}  sample_{r['sample_id']}")
        print(f"  Prompt: {r['prompt'][:100]}...")
        print(f"  Base avg rep: {r['avg_base_rep']:.3f}  |  HVM avg rep: {r['avg_hvm_rep']:.3f}  |  improvement: {r['rep_improvement']:.3f}")
        print(f"  Base unique: {r['avg_base_unique']:.3f}  |  HVM unique: {r['avg_hvm_unique']:.3f}")
        print(f"  Worst base: rep={r['worst_base']['rep_ratio']}, paths={r['worst_base']['n_paths']}, "
              f"late_rep={r['worst_base']['late_rep_ratio']}, early_div={r['worst_base']['early_diverse']}, "
              f"late_collapse={r['worst_base']['late_collapse']}")
        print(f"  Best HVM:  rep={r['best_hvm']['rep_ratio']}, paths={r['best_hvm']['n_paths']}, "
              f"unique={r['best_hvm']['unique_ratio']}")
        print()

    out_path = '/mnt/data2/wuqingman/omnisvg-train/inference/failure_case_report.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {out_path}")


if __name__ == '__main__':
    main()

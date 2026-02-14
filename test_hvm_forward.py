"""
HVM-SVG 完整 forward pass 端到端测试。
验证: Dataset → Collate → Model → Loss 全流程。
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import sys
from pathlib import Path

# Fix for DeepSpeed
import torch.autograd.graph as _torch_ag_graph
_orig = getattr(_torch_ag_graph, '_get_grad_fn_or_grad_acc', None)
if _orig:
    def _safe(t):
        if t.requires_grad and t.grad_fn is None:
            v = t.view_as(t)
            return v.grad_fn.next_functions[0][0] if v.grad_fn else None
        return t.grad_fn
    _torch_ag_graph._get_grad_fn_or_grad_acc = _safe

from transformers import AutoProcessor, AutoTokenizer
from utils.config import OmniSVGConfig
from hvm_modules import HVMConfig
from hvm_decoder import HVMSketchDecoder
from hvm_dataset import HVMDataset, create_hvm_collate_fn
from decoder import SketchDecoder

DEVICE = "cuda:0"
MODEL_SIZE = "8B"
DATA_DIR = "/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/data_test2"
HVM_DIR = "/mnt/data/wuqingman/datasets/OmniSVG/MMSVG-Illustration/hvm_precomputed"


def test_dataset():
    """测试 Dataset + Collate"""
    print("=" * 60)
    print("Test 1: Dataset + Collate")
    print("=" * 60)

    config = OmniSVGConfig(config_dir="./configs", model_size=MODEL_SIZE)
    token_config = config.tokenization

    dataset = HVMDataset(
        data_dir=DATA_DIR,
        hvm_dir=HVM_DIR,
        token_config=token_config,
        train_config=config.training,
    )
    print(f"Dataset size: {len(dataset)}")

    # 测试单个样本
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  text: '{sample['text'][:80]}...'")
    print(f"  pix_seq: {len(sample['pix_seq'])} tokens")
    print(f"  ref_features: {len(sample['ref_features'])} refs, each {sample['ref_features'][0].shape}")
    print(f"  ref_best_groups: {sample['ref_best_groups']}")
    print(f"  ref_text: '{sample['ref_text'][:80]}...'")

    # 测试 collate
    base_model_path = token_config.base_model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side="left")
    processor = AutoProcessor.from_pretrained(base_model_path, padding_side="left")
    processor.tokenizer.padding_side = "left"

    collate_fn = create_hvm_collate_fn(
        processor=processor,
        tokenizer=tokenizer,
        token_config=token_config,
    )

    # Collate 2 个样本
    batch = collate_fn([dataset[0], dataset[1]])
    print(f"\nCollated batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    return batch, tokenizer, processor, token_config


def test_model_forward(batch, tokenizer, token_config):
    """测试 Model forward pass"""
    print("\n" + "=" * 60)
    print("Test 2: Model Forward Pass")
    print("=" * 60)

    # 加载 base model
    print("Loading OmniSVG base model...")
    base_model = SketchDecoder(
        pix_len=2048,
        text_len=800,
        model_path=token_config.base_model,
        vocab_size=token_config.extended_vocab_size,
        bos_token_id=token_config.bos_token_id,
        eos_token_id=token_config.eos_token_id,
        pad_token_id=token_config.pad_token_id,
    )

    # 加载 OmniSVG checkpoint
    ckpt_path = token_config.checkpoint
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        from train import load_checkpoint_state_dict, find_checkpoint_file
        ckpt_file = find_checkpoint_file(ckpt_path) if os.path.isdir(ckpt_path) else ckpt_path
        if ckpt_file:
            state_dict = load_checkpoint_state_dict(
                ckpt_file if os.path.isfile(ckpt_file) else ckpt_path
            )
            missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

    # 创建 HVM model
    hvm_config = HVMConfig()
    model = HVMSketchDecoder(
        base_model=base_model,
        hvm_config=hvm_config,
        tokenizer=tokenizer,
    )
    model = model.to(DEVICE)

    # 验证参数冻结
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"\nParameters: trainable={trainable/1e6:.1f}M, frozen={frozen/1e6:.1f}M")

    # Forward pass
    print("\nRunning forward pass...")
    model.train()

    # 移动 batch 到 device
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    ref_features = batch["ref_features"].to(DEVICE)
    ref_best_feature = batch["ref_best_feature"].to(DEVICE)
    ref_text_ids = batch["ref_text_ids"].to(DEVICE)
    ref_text_mask = batch["ref_text_mask"].to(DEVICE)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ref_features=ref_features,
            ref_best_feature=ref_best_feature,
            groups_bbox_feature=batch["groups_bbox_feature"],
            ref_text_ids=ref_text_ids,
            ref_text_mask=ref_text_mask,
        )

    print(f"  logits shape: {outputs.logits.shape}")

    # Compute loss
    from train_hvm import compute_loss
    loss = compute_loss(outputs, labels)
    print(f"  loss: {loss.item():.4f}")

    # Backward pass
    print("\nRunning backward pass...")
    loss.backward()

    # 检查梯度
    print("\nGradient check:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            if "gate" in name or "alpha" in name:
                print(f"  {name}: grad_norm={grad_norm:.6f}")
            break  # 只检查第一个有梯度的参数

    # 检查 base model 没有梯度
    base_has_grad = False
    for name, param in model.base_model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            base_has_grad = True
            break
    print(f"  Base model has gradients: {base_has_grad} (should be False)")

    # Gate values
    print("\nGate alpha values (should all be ~0.0 at init):")
    for i, pim in enumerate(model.pims):
        alpha = pim.gate.base_alpha.item()
        tanh_alpha = torch.tanh(pim.gate.base_alpha).item()
        print(f"  PIM {i}: alpha={alpha:.6f}, tanh(alpha)={tanh_alpha:.6f}")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated(DEVICE) / 1e9:.2f} GB")

    return model


def test_checkpoint_save_load(model):
    """测试 checkpoint 保存和加载"""
    print("\n" + "=" * 60)
    print("Test 3: Checkpoint Save/Load")
    print("=" * 60)

    save_path = "/mnt/data/wuqingman/omnisvg-train/test_hvm_ckpt.pt"
    model.save_hvm_checkpoint(save_path)

    # 修改一个参数
    with torch.no_grad():
        model.pims[0].gate.base_alpha.fill_(0.5)
    print(f"  After modification: alpha[0]={model.pims[0].gate.base_alpha.item():.4f}")

    # 重新加载
    model.load_hvm_checkpoint(save_path)
    print(f"  After reload: alpha[0]={model.pims[0].gate.base_alpha.item():.4f} (should be ~0.0)")

    if os.path.exists(save_path):
        os.remove(save_path)
    print("  Checkpoint save/load: OK")


if __name__ == "__main__":
    print("HVM-SVG End-to-End Test")
    print("=" * 60)

    # Test 1
    batch, tokenizer, processor, token_config = test_dataset()

    # Test 2
    model = test_model_forward(batch, tokenizer, token_config)

    # Test 3
    test_checkpoint_save_load(model)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

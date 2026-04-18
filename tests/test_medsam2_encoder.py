import torch
from models.medsam2_encoder import MedSAM2Encoder


def test_encoder_output_shape():
    enc = MedSAM2Encoder(embed_dim=256, lora_rank=16, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    main, skip = enc(x)
    assert main.shape == (2, 256, 16, 16), f"main {main.shape}"
    assert skip.shape == (2, 128, 32, 32), f"skip {skip.shape}"


def test_encoder_lora_only_trainable():
    enc = MedSAM2Encoder(embed_dim=256, lora_rank=16, pretrained=False)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    # LoRA A/B, projection heads, skip projection must be trainable
    assert any("lora_A" in n for n in trainable)
    assert any("proj_main" in n for n in trainable)
    assert any("proj_skip" in n for n in trainable)
    # Backbone blocks must be frozen
    frozen = [n for n, p in enc.named_parameters() if not p.requires_grad]
    assert len(frozen) > 0


def test_encoder_trainable_param_count_reasonable():
    enc = MedSAM2Encoder(embed_dim=256, lora_rank=16, pretrained=False)
    n_trainable = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    # LoRA deltas (~4M) + projection heads (~0.2M) < total (~38M)
    assert 1e6 < n_trainable < 10e6, f"Trainable {n_trainable} outside expected band"

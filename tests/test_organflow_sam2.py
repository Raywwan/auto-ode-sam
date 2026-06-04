import torch
from omegaconf import OmegaConf
from models.organflow_sam2 import OrganFlowSAM2


def _cfg():
    return OmegaConf.create({
        "model": {
            "architecture": "organflow_sam2",
            "img_size": 256,
            "embed_dim": 256,
            "skip_channels": 128,
            "encoder": {"pretrained": False, "lora_rank": 16},
            "pfesa": {},
            "ode": {"n_organs": 15, "organ_emb_dim": 32, "ode_hidden": 128, "n_freqs": 6, "substeps": 4},
            "decoder": {"n_organs": 15, "transformer_depth": 4, "transformer_mlp_dim": 2048},
            "graph": {"n_organs": 15},
        }
    })


def test_forward_shape():
    m = OrganFlowSAM2(_cfg())
    m.train()
    images = torch.randn(1, 4, 3, 256, 256)  # (B, D, C, H, W) with D=4 for speed
    organ_id = torch.tensor([6])
    out = m(images, organ_id, is_3d=True)
    # V5: decoder outputs 128×128 (three-stage upsample).
    assert out["masks"].shape == (1, 15, 128, 128)
    assert out["iou_pred"].shape == (1, 15)
    # V5: all-slice deepsup (N=5 slices) — shape (B, N, K, h, w).
    ds = out["deepsup_logits"]
    assert ds.dim() == 5 and ds.shape[0] == 1 and ds.shape[2] == 15
    assert "flow_targets" in out and out["flow_targets"] is not None
    assert "xsc_pairs" in out["flow_targets"]


def test_flow_targets_none_in_eval():
    m = OrganFlowSAM2(_cfg())
    m.eval()
    images = torch.randn(1, 4, 3, 256, 256)
    organ_id = torch.tensor([6])
    with torch.no_grad():
        out = m(images, organ_id, is_3d=True)
    assert out["flow_targets"] is None


def test_param_budget():
    m = OrganFlowSAM2(_cfg())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"trainable={n_train/1e6:.1f}M total={n_total/1e6:.1f}M")
    # Post T2 stage-3 drop (commit 9a471b8), backbone shrinks from ~38M → ~13M,
    # so full-trainable band is ~5–15M not spec's original 40–80M.
    assert 5e6 < n_train < 15e6, f"trainable {n_train} outside expected 5-15M band"

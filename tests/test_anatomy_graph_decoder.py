import torch
from models.anatomy_graph_decoder import AnatomyGraphAttention, AnatomyGraphDecoder


def test_graph_attention_shape():
    ga = AnatomyGraphAttention(n_organs=15, embed_dim=256)
    q = torch.randn(15, 256)
    q_out = ga(q)
    assert q_out.shape == q.shape


def test_graph_attention_identity_if_adjacency_zero():
    ga = AnatomyGraphAttention(n_organs=15, embed_dim=256)
    with torch.no_grad():
        ga.adjacency.fill_(-100.0)
    q = torch.randn(15, 256)
    q_out = ga(q)
    assert torch.allclose(q_out, q, atol=1e-3)


def test_graph_attention_init_adjacency_buffer():
    ga = AnatomyGraphAttention(n_organs=15, embed_dim=256)
    assert hasattr(ga, "_init_adjacency")
    assert ga._init_adjacency.shape == (15, 15)
    # Buffer must equal the parameter at init
    assert torch.allclose(ga._init_adjacency, ga.adjacency.detach())


def test_decoder_output_shape():
    dec = AnatomyGraphDecoder(
        embed_dim=256, n_organs=15, skip_channels=128,
        transformer_depth=4, transformer_mlp_dim=2048,
    )
    B, C, H, W = 2, 256, 16, 16
    image_emb = torch.randn(B, C, H, W)
    dense_pe = torch.randn(B, C, H, W)
    skip = torch.randn(B, 128, 32, 32)
    masks, iou = dec(image_emb, dense_pe, skip)
    assert masks.shape == (B, 15, 64, 64)
    assert iou.shape == (B, 15)

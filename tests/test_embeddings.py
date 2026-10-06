import math
import pytest
import torch


# ============================================================
# IMPORTANT:
# Adjust these imports to the actual module names in your project.
# The class names below match the versions we built/discussed:
#
#   PatchEmbedd(patch_size, in_channels=3, embed_size=256)
#   AbsolutePosEnc1D()
#   AbsolutePosEnc2D(...)
#   RoPE1D(...)
#   RoPE_2D(base=10000)
#   CoPE1D(max_npos, head_dim)
#
# Example:
# from model.embeddings import (
#     PatchEmbedd,
#     AbsolutePosEnc1D,
#     AbsolutePosEnc2D,
#     RoPE1D,
#     RoPE_2D,
#     CoPE1D,
# )
# ============================================================

from embeddings import (
    PatchEmbedd,
    AbsolutePosEnc1D,
    AbsolutePosEnc2D,
    RoPE1D,
    RoPE2D,
    CoPE1D,
)


ATOL = 1e-5
RTOL = 1e-5


# ============================================================
# Helpers
# ============================================================

def assert_finite(x: torch.Tensor):
    assert torch.isfinite(x).all()


def manual_sinusoidal_1d(T: int, D: int, base: float = 10000.0, device="cpu"):
    """
    Standard sinusoidal positional encoding:
        PE[pos, 2i]   = sin(pos / base^(2i/D))
        PE[pos, 2i+1] = cos(pos / base^(2i/D))
    """
    assert D % 2 == 0

    pos = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)
    even_dims = torch.arange(0, D, 2, dtype=torch.float32, device=device)

    inv_freq = base ** (-even_dims / D)
    angles = pos * inv_freq.unsqueeze(0)

    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return pe


def reverse_cumsum(x: torch.Tensor):
    return torch.flip(
        torch.cumsum(torch.flip(x, dims=[-1]), dim=-1),
        dims=[-1],
    )


# ============================================================
# PatchEmbedd
# ============================================================

def test_patch_embed_shape():
    B, C, H, W = 2, 3, 32, 48
    P = 4
    D = 64

    model = PatchEmbedd(
        patch_size=P,
        in_channels=C,
        embed_size=D,
    )

    x = torch.randn(B, C, H, W)

    out, hp, wp = model(x)

    assert hp == H // P
    assert wp == W // P
    assert out.shape == (B, (H // P) * (W // P), D)


def test_patch_embed_finite():
    model = PatchEmbedd(
        patch_size=4,
        in_channels=3,
        embed_size=32,
    )

    x = torch.randn(2, 3, 16, 16)
    out, _, _ = model(x)

    assert_finite(out)


def test_patch_embed_backward():
    model = PatchEmbedd(
        patch_size=4,
        in_channels=3,
        embed_size=32,
    )

    x = torch.randn(
        2, 3, 16, 16,
        requires_grad=True,
    )

    out, _, _ = model(x)
    loss = out.square().mean()
    loss.backward()

    assert x.grad is not None
    assert_finite(x.grad)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0

    for p in trainable_params:
        assert p.grad is not None
        assert_finite(p.grad)


# ============================================================
# AbsolutePosEnc1D
# ============================================================

def test_ape_1d_shape():
    B, T, D = 2, 7, 8

    ape = AbsolutePosEnc1D()

    x = torch.zeros(B, T, D)
    out = ape(x)

    assert out.shape == x.shape


def test_ape_1d_zero_position():
    """
    For standard sinusoidal APE:
        pos = 0 -> [0, 1, 0, 1, ...]
    Since x == 0, the output itself is the PE.
    """
    B, T, D = 1, 4, 8

    ape = AbsolutePosEnc1D()

    x = torch.zeros(B, T, D)
    out = ape(x)

    expected = torch.tensor(
        [0., 1., 0., 1., 0., 1., 0., 1.],
        dtype=out.dtype,
        device=out.device,
    )

    assert torch.allclose(
        out[0, 0],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_ape_1d_matches_formula():
    B, T, D = 1, 5, 8

    ape = AbsolutePosEnc1D()

    x = torch.zeros(B, T, D)
    out = ape(x)

    expected = manual_sinusoidal_1d(
        T=T,
        D=D,
        base=10000.0,
        device=out.device,
    ).to(out.dtype)

    assert torch.allclose(
        out[0],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_ape_1d_is_added_to_input():
    B, T, D = 2, 5, 8

    ape = AbsolutePosEnc1D()

    x = torch.randn(B, T, D)
    zero = torch.zeros_like(x)

    out_x = ape(x)
    out_zero = ape(zero)

    # If forward returns x + PE, then:
    #   APE(x) - APE(0) == x
    assert torch.allclose(
        out_x - out_zero,
        x,
        atol=ATOL,
        rtol=RTOL,
    )


def test_ape_1d_same_positions_across_batch():
    B, T, D = 3, 5, 8

    ape = AbsolutePosEnc1D()

    x = torch.zeros(B, T, D)
    out = ape(x)

    assert torch.allclose(out[0], out[1], atol=ATOL, rtol=RTOL)
    assert torch.allclose(out[1], out[2], atol=ATOL, rtol=RTOL)


# ============================================================
# AbsolutePosEnc2D
# ============================================================

def test_ape_2d_shape():
    B = 2
    H = 3
    W = 4
    T = H * W
    D = 8

    ape = AbsolutePosEnc2D()

    x = torch.zeros(B, T, D)
    out = ape(x, H, W)

    assert out.shape == x.shape


def test_ape_2d_is_added_to_input():
    B = 2
    H = 3
    W = 4
    T = H * W
    D = 8

    ape = AbsolutePosEnc2D()

    x = torch.randn(B, T, D)
    zero = torch.zeros_like(x)

    out_x = ape(x, H, W)
    out_zero = ape(zero, H, W)

    assert torch.allclose(
        out_x - out_zero,
        x,
        atol=ATOL,
        rtol=RTOL,
    )


def test_ape_2d_axis_independence():
    """
    Assumption from our implementation:
      D is split equally:
        first D/2  -> row / H encoding
        second D/2 -> col / W encoding

    Therefore:
      same row => H-part identical
      same col => W-part identical
    """
    B = 1
    H = 3
    W = 4
    T = H * W
    D = 8

    ape = AbsolutePosEnc2D()

    x = torch.zeros(B, T, D)
    out = ape(x, H, W)

    pe = out[0].reshape(H, W, D)
    half = D // 2

    # (row=0,col=0) and (row=0,col=1):
    # same row -> H encoding must be the same.
    assert torch.allclose(
        pe[0, 0, :half],
        pe[0, 1, :half],
        atol=ATOL,
        rtol=RTOL,
    )

    # (row=0,col=0) and (row=1,col=0):
    # same col -> W encoding must be the same.
    assert torch.allclose(
        pe[0, 0, half:],
        pe[1, 0, half:],
        atol=ATOL,
        rtol=RTOL,
    )


def test_ape_2d_changes_when_coordinate_changes():
    B = 1
    H = 3
    W = 4
    T = H * W
    D = 8

    ape = AbsolutePosEnc2D()

    x = torch.zeros(B, T, D)
    pe = ape(x, H, W)[0].reshape(H, W, D)

    half = D // 2

    # Different row -> H part should normally differ.
    assert not torch.allclose(
        pe[0, 0, :half],
        pe[1, 0, :half],
    )

    # Different col -> W part should normally differ.
    assert not torch.allclose(
        pe[0, 0, half:],
        pe[0, 1, half:],
    )


# ============================================================
# RoPE1D
# ============================================================

def test_rope_1d_shape():
    B, H, T, D = 2, 4, 6, 8

    rope = RoPE1D()

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_1d_position_zero_identity():
    """
    At position 0:
        theta = 0
        cos(theta) = 1
        sin(theta) = 0
    therefore RoPE must be identity.
    """
    B, H, T, D = 2, 3, 5, 8

    rope = RoPE1D()

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    q_rot, k_rot = rope(q, k)

    assert torch.allclose(
        q_rot[:, :, 0],
        q[:, :, 0],
        atol=ATOL,
        rtol=RTOL,
    )

    assert torch.allclose(
        k_rot[:, :, 0],
        k[:, :, 0],
        atol=ATOL,
        rtol=RTOL,
    )


def test_rope_1d_preserves_norm():
    B, H, T, D = 2, 4, 7, 8

    rope = RoPE1D()

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    q_rot, k_rot = rope(q, k)

    assert torch.allclose(
        q_rot.norm(dim=-1),
        q.norm(dim=-1),
        atol=ATOL,
        rtol=RTOL,
    )

    assert torch.allclose(
        k_rot.norm(dim=-1),
        k.norm(dim=-1),
        atol=ATOL,
        rtol=RTOL,
    )


def test_rope_1d_preserves_same_position_dot_product():
    """
    Orthogonal rotation preserves dot products when q and k
    are rotated by the same positional rotation:
        (Rq)^T (Rk) = q^T k
    """
    B, H, T, D = 2, 3, 5, 8

    rope = RoPE1D()

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    q_rot, k_rot = rope(q, k)

    before = (q * k).sum(dim=-1)
    after = (q_rot * k_rot).sum(dim=-1)

    assert torch.allclose(
        before,
        after,
        atol=ATOL,
        rtol=RTOL,
    )


# ============================================================
# RoPE 2D
# ============================================================

def test_rope_2d_shape():
    B = 2
    N_HEADS = 4
    HP = 3
    WP = 4
    T = HP * WP
    HEAD_DIM = 8

    rope = RoPE2D()

    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    k = torch.randn(B, N_HEADS, T, HEAD_DIM)

    q_rot, k_rot = rope(q, k, HP, WP)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_2d_top_left_patch_identity():
    """
    Patch (row=0,col=0) has zero angle on both axes,
    so the full vector should stay unchanged.

    This assumes row-major flattening:
        token 0 == (0,0)
    """
    B = 2
    N_HEADS = 3
    HP = 3
    WP = 4
    T = HP * WP
    HEAD_DIM = 8

    rope = RoPE2D()

    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    k = torch.randn(B, N_HEADS, T, HEAD_DIM)

    q_rot, k_rot = rope(q, k, HP, WP)

    assert torch.allclose(
        q_rot[:, :, 0],
        q[:, :, 0],
        atol=ATOL,
        rtol=RTOL,
    )

    assert torch.allclose(
        k_rot[:, :, 0],
        k[:, :, 0],
        atol=ATOL,
        rtol=RTOL,
    )


def test_rope_2d_preserves_norm():
    B = 2
    N_HEADS = 4
    HP = 3
    WP = 4
    T = HP * WP
    HEAD_DIM = 8

    rope = RoPE2D()

    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    k = torch.randn(B, N_HEADS, T, HEAD_DIM)

    q_rot, k_rot = rope(q, k, HP, WP)

    assert torch.allclose(
        q_rot.norm(dim=-1),
        q.norm(dim=-1),
        atol=ATOL,
        rtol=RTOL,
    )

    assert torch.allclose(
        k_rot.norm(dim=-1),
        k.norm(dim=-1),
        atol=ATOL,
        rtol=RTOL,
    )


def test_rope_2d_preserves_same_position_dot_product():
    B = 2
    N_HEADS = 4
    HP = 3
    WP = 4
    T = HP * WP
    HEAD_DIM = 8

    rope = RoPE2D()

    q = torch.randn(B, N_HEADS, T, HEAD_DIM)
    k = torch.randn(B, N_HEADS, T, HEAD_DIM)

    q_rot, k_rot = rope(q, k, HP, WP)

    before = (q * k).sum(dim=-1)
    after = (q_rot * k_rot).sum(dim=-1)

    assert torch.allclose(
        before,
        after,
        atol=ATOL,
        rtol=RTOL,
    )


# ============================================================
# CoPE1D
# ============================================================

def test_cope_1d_shape():
    B, H, T, D = 2, 4, 6, 8
    MAX_NPOS = 16

    cope = CoPE1D(
        max_npos=MAX_NPOS,
        head_dim=D,
    )

    q = torch.randn(B, H, T, D)
    attn_logits = torch.randn(B, H, T, T)

    # Current version discussed:
    # forward(q, k, attn_logits, **kwargs)
    #
    # If your final implementation removed k because CoPE does not
    # actually need it, replace this with:
    #   out = cope(q, attn_logits)
    k = torch.randn_like(q)
    out = cope(q, k, attn_logits)

    assert out.shape == (B, H, T, T)


def test_cope_zero_q_gives_zero_bias():
    B, H, T, D = 2, 3, 5, 8
    MAX_NPOS = 16

    cope = CoPE1D(
        max_npos=MAX_NPOS,
        head_dim=D,
    )

    q = torch.zeros(B, H, T, D)
    k = torch.randn_like(q)
    attn_logits = torch.randn(B, H, T, T)

    out = cope(q, k, attn_logits)

    assert torch.allclose(
        out,
        torch.zeros_like(out),
        atol=ATOL,
        rtol=RTOL,
    )


def test_cope_reverse_cumsum_positions_for_zero_logits():
    """
    This directly verifies the contextual-position rule.

    If all attention logits are 0:
        sigmoid(0) = 0.5

    For T=4:
        reverse cumsum([.5,.5,.5,.5])
        = [2.0,1.5,1.0,0.5]
    """
    T = 4

    logits = torch.zeros(1, 1, T, T)
    gates = torch.sigmoid(logits)
    positions = reverse_cumsum(gates)

    expected_row = torch.tensor(
        [2.0, 1.5, 1.0, 0.5],
        dtype=positions.dtype,
    )

    for row in range(T):
        assert torch.allclose(
            positions[0, 0, row],
            expected_row,
            atol=ATOL,
            rtol=RTOL,
        )


def test_cope_interpolation():
    """
    Controlled CoPE test.

    pos_emb:
        p0 -> 0
        p1 -> 10
        p2 -> 20
        p3 -> 30

    q = 1

    attn_logits = 0
      => gates = 0.5
      => positions = [2, 1.5, 1, 0.5]

    Expected interpolated logits:
        [20, 15, 10, 5]
    """
    T = 4
    D = 1

    cope = CoPE1D(
        max_npos=4,
        head_dim=D,
    )

    with torch.no_grad():
        assert cope.pos_emb.shape[-1] == 4
        cope.pos_emb.copy_(
            torch.tensor(
                [[[0.0, 10.0, 20.0, 30.0]]],
                dtype=cope.pos_emb.dtype,
            )
        )

    q = torch.ones(1, 1, T, D)
    k = torch.zeros_like(q)
    attn_logits = torch.zeros(1, 1, T, T)

    out = cope(q, k, attn_logits)

    expected = torch.tensor(
        [20.0, 15.0, 10.0, 5.0],
        dtype=out.dtype,
        device=out.device,
    )

    assert torch.allclose(
        out[0, 0, 0],
        expected,
        atol=1e-4,
        rtol=1e-4,
    )


def test_cope_clamps_positions_to_max_npos():
    """
    max_npos = 3 -> valid indices are 0,1,2.

    With very large positive attention logits:
        sigmoid(logit) ~= 1

    For T=5:
        p ~= [5,4,3,2,1]

    After clamp(max=2):
        p = [2,2,2,2,1]

    pos_emb = [0,10,20]
      => expected = [20,20,20,20,10]
    """
    T = 5
    D = 1

    cope = CoPE1D(
        max_npos=3,
        head_dim=D,
    )

    with torch.no_grad():
        cope.pos_emb.copy_(
            torch.tensor(
                [[[0.0, 10.0, 20.0]]],
                dtype=cope.pos_emb.dtype,
            )
        )

    q = torch.ones(1, 1, T, D)
    k = torch.zeros_like(q)

    attn_logits = torch.full(
        (1, 1, T, T),
        100.0,
    )

    out = cope(q, k, attn_logits)

    expected = torch.tensor(
        [20.0, 20.0, 20.0, 20.0, 10.0],
        dtype=out.dtype,
        device=out.device,
    )

    assert torch.allclose(
        out[0, 0, 0],
        expected,
        atol=1e-4,
        rtol=1e-4,
    )


def test_cope_backward():
    B, H, T, D = 2, 4, 5, 8

    cope = CoPE1D(
        max_npos=16,
        head_dim=D,
    )

    q = torch.randn(
        B, H, T, D,
        requires_grad=True,
    )

    k = torch.randn(
        B, H, T, D,
        requires_grad=True,
    )

    attn_logits = torch.randn(
        B, H, T, T,
        requires_grad=True,
    )

    out = cope(q, k, attn_logits)
    loss = out.square().mean()
    loss.backward()

    assert q.grad is not None
    assert_finite(q.grad)

    # CoPE's contextual positions depend on attention logits,
    # so gradients should reach them.
    assert attn_logits.grad is not None
    assert_finite(attn_logits.grad)

    assert cope.pos_emb.grad is not None
    assert_finite(cope.pos_emb.grad)


# ============================================================
# General validation tests
# ============================================================

def test_rope_1d_rejects_odd_head_dim():
    """
    RoPE rotates pairs of channels, so odd head_dim
    should be invalid.

    If your implementation uses `assert` instead of ValueError,
    broaden this to:
        with pytest.raises((AssertionError, ValueError)):
    """
    rope = RoPE1D()

    q = torch.randn(1, 2, 5, 7)
    k = torch.randn(1, 2, 5, 7)

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        rope(q, k)


def test_rope_2d_rejects_head_dim_not_divisible_by_4():
    """
    Our axial 2D RoPE splits head_dim into H and W halves,
    and each half must itself contain rotation pairs.

    Therefore:
        head_dim % 4 == 0
    """
    rope = RoPE2D()

    hp, wp = 2, 3
    T = hp * wp

    q = torch.randn(1, 2, T, 10)
    k = torch.randn(1, 2, T, 10)

    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        rope(q, k, hp, wp)

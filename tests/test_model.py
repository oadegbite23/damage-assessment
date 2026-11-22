"""Smoke tests for the Siamese U-Net. Skipped automatically if torch or
segmentation-models-pytorch isn't installed. Uses encoder_weights=None so
nothing is downloaded, and small inputs so it runs on a CPU in seconds."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from model import N_DAMAGE_CLASSES, SiameseUNet  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return SiameseUNet(encoder_name="resnet18", encoder_weights=None)


def _batch(n=2, size=64):
    return torch.randn(n, 3, size, size), torch.randn(n, 3, size, size)


def test_output_shapes(model):
    pre, post = _batch()
    loc, dmg = model(pre, post)
    assert loc.shape == (2, 1, 64, 64)
    assert dmg.shape == (2, N_DAMAGE_CLASSES, 64, 64)


def test_gradients_reach_the_shared_encoder_from_both_heads(model):
    pre, post = _batch()
    model.zero_grad()
    loc, dmg = model(pre, post)
    (loc.mean() + dmg.mean()).backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)


def test_damage_output_depends_on_the_post_image(model):
    model.eval()
    pre, post = _batch(1)
    with torch.no_grad():
        _, dmg_a = model(pre, post)
        _, dmg_b = model(pre, post + 1.0)
    assert not torch.allclose(dmg_a, dmg_b)


def test_localization_ignores_the_post_image(model):
    model.eval()
    pre, post = _batch(1)
    with torch.no_grad():
        loc_a, _ = model(pre, post)
        loc_b, _ = model(pre, post + 1.0)
    # In eval mode BatchNorm uses running statistics, so pre and post being
    # batched together in one encoder pass can't leak into the localization output.
    assert torch.allclose(loc_a, loc_b, atol=1e-5)


def test_predict_returns_masks_with_valid_values(model):
    pre, post = _batch()
    loc, dmg = model.predict(pre, post)
    assert loc.shape == dmg.shape == (2, 64, 64)
    assert loc.dtype == torch.bool
    assert set(dmg.unique().tolist()) <= {0, 1, 2, 3, 4}
    assert (dmg[~loc] == 0).all()  # no damage label outside predicted buildings


def test_rejects_sizes_not_divisible_by_32_and_mismatched_inputs(model):
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.randn(1, 3, 100, 100), torch.randn(1, 3, 100, 100))
    with pytest.raises(ValueError, match="same shape"):
        model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 96, 96))

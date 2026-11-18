"""Runs only where PyTorch is installed (CI: pip install torch --index-url https://download.pytorch.org/whl/cpu)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ml.seq import V
from ml.torch_models import Classifier, MovementEncoder, export_onnx, supcon_loss


def test_encoder_shapes_and_unit_norm():
    m = MovementEncoder(embed_dim=32).eval()
    z = m(torch.randn(3, 4, 90, V))
    assert z.shape == (3, 32) and torch.allclose(z.norm(dim=1), torch.ones(3), atol=1e-5)


def test_supcon_prefers_clustered_embeddings():
    y = torch.tensor([0, 0, 1, 1, 2, 2])
    good = torch.nn.functional.normalize(torch.eye(3)[y] + 0.01 * torch.randn(6, 3), dim=1)
    bad = torch.nn.functional.normalize(torch.randn(6, 3), dim=1)
    assert supcon_loss(good, y) < supcon_loss(bad, y)
    assert float(supcon_loss(good, torch.arange(6))) == 0.0              # no positives -> no loss, no NaN


def test_can_overfit_a_tiny_batch():
    """Reaches the SupCon floor (log(positives)=log 3 for 4 samples/class) and fits the batch in eval mode."""
    torch.manual_seed(0)
    m = Classifier(3, embed_dim=16)
    x, y = torch.randn(12, 4, 90, V), torch.arange(12) % 3
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(120):
        m.train()
        z, lg = m(x)
        loss = supcon_loss(z, y) + torch.nn.functional.cross_entropy(lg, y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        _, lg = m(x)
    assert float(loss.detach()) < np.log(3) + 0.1 and (lg.argmax(1) == y).all()


def test_onnx_export_matches_torch(tmp_path):
    err = export_onnx(MovementEncoder(embed_dim=16), str(tmp_path / "e.onnx"))
    assert err < 1e-4


def test_train_script_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil, pathlib
    shutil.copy(pathlib.Path(__file__).parent.parent / "train.yaml", "train.yaml")
    from ml.train_encoder import main
    main(["--synthetic", "--store", "s", "--out", "m/e.onnx", "--epochs", "2", "--allow-untrusted"])
    assert (tmp_path / "m/e.onnx").exists() and (tmp_path / "m/e.json").exists()

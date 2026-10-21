import io
import unittest
from pathlib import Path
import torch
from PIL import Image
from ViTexperiment import build_model, get_transform, make_optimizer

class SmokeTests(unittest.TestCase):
    def test_checkpoint_and_optimizer(self):
        torch.set_num_threads(2)
        for width in (48, 96):
            config = dict(embed_dim=width, depth=6, num_head=3, patch_size=4, drop=0.1)
            model = build_model("custom", config).eval()
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad(): expected = model(x)
            self.assertEqual(tuple(expected.shape), (2, 10))
            buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
            restored = build_model("custom", config).eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            with torch.no_grad(): torch.testing.assert_close(expected, restored(x))
            optimizer = make_optimizer(model, 5e-4)
            ids = [id(v) for g in optimizer.param_groups for v in g["params"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(v) for v in model.parameters()})

    def test_transforms(self):
        for kind, size in [("custom", 32), ("torchvision_vit_b_16", 224)]:
            for training in (False, True):
                self.assertEqual(tuple(get_transform(training, kind)(Image.new("RGB", (32,32))).shape), (3,size,size))

    def test_result_logs(self):
        root = Path(__file__).resolve().parents[1] / "reports/data"
        for name in ("small50", "small150", "wide150", "pretrained30"):
            self.assertIn("Test samples=10000", (root/f"{name}.log").read_text(encoding="utf-8-sig"))

if __name__ == "__main__": unittest.main()

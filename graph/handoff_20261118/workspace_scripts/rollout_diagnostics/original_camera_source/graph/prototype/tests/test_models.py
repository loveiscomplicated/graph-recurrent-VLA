import tempfile
from pathlib import Path
import unittest

import torch

from graph.prototype.common import config, seed_all
from graph.prototype.graph_builder import GraphBuilder, flat_features
from graph.prototype.models import make_policy, model_inputs
from .fixtures import geometry


class GraphTests(unittest.TestCase):
    def setUp(self):
        seed_all(0, 2)
        self.g = geometry()

    def test_edges_features_connectivity_and_batch(self):
        for center in (False, True):
            builder = GraphBuilder(center=center)
            graph = builder(self.g)
            b, n = graph.x.shape[:2]
            src, dst = graph.edge_index
            self.assertTrue(((src >= 0) & (src < b*n) & (dst >= 0) & (dst < b*n)).all())
            self.assertTrue((src//n == dst//n).all())
            pos = graph.x[:, :, 5:8].reshape(-1, 3)
            rel = pos[dst]-pos[src]
            torch.testing.assert_close(graph.edge_attr[:, :3], rel)
            torch.testing.assert_close(graph.edge_attr[:, 3], rel.norm(dim=-1))
            self.assertTrue(torch.isfinite(graph.edge_attr).all())
            mapping = {(int(a), int(d)): i for i, (a, d) in enumerate(zip(src, dst))}
            for (a, d), i in mapping.items():
                j = mapping[(d, a)]
                torch.testing.assert_close(graph.edge_attr[i, :3], -graph.edge_attr[j, :3])
            for batch in range(b):
                visited, todo = set(), [batch*n]
                while todo:
                    node = todo.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    todo.extend(dst[src == node].tolist())
                self.assertEqual(len(visited), n)
            single = builder(self.g.select(slice(0, 1)))
            torch.testing.assert_close(graph.x[:1], single.x)
            mask = src < n
            torch.testing.assert_close(graph.edge_index[:, mask], single.edge_index)
            torch.testing.assert_close(graph.edge_attr[mask], single.edge_attr)

    def test_flat_has_exact_same_information_as_surface(self):
        graph = GraphBuilder()(self.g)
        expected = torch.cat((graph.x.flatten(1), graph.context), -1)
        torch.testing.assert_close(flat_features(self.g), expected, rtol=0, atol=0)
        self.assertEqual(graph.context.shape, (4, 81))


class ModelTests(unittest.TestCase):
    def test_forward_backward_overfit_checkpoint_determinism(self):
        seed_all(0, 2)
        g = geometry()
        cfg = config()
        target = torch.cat((g.peg_pose[:, :3]-.1, g.ee_pose[:, :3]-.1, torch.full((4, 1), -.8)), -1)
        for name in ("flat", "center", "surface"):
            with self.subTest(model=name):
                model = make_policy(name, cfg)
                inputs = model_inputs(name, g, GraphBuilder(center=name == "center"))
                self.assertEqual(model.encoder(inputs).shape, (4, 128))
                self.assertEqual(model(inputs).shape, (4, 7))
                optimizer = torch.optim.Adam(model.parameters(), lr=.002)
                before = ((model(inputs)-target)**2).mean().item()
                for _ in range(80):
                    optimizer.zero_grad()
                    loss = ((model(inputs)-target)**2).mean()
                    loss.backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    optimizer.step()
                model.eval()
                after = ((model(inputs)-target)**2).mean().item()
                self.assertLess(after, before*.05)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"checkpoint.pt"
                    torch.save(model.state_dict(), path)
                    restored = make_policy(name, cfg)
                    restored.load_state_dict(torch.load(path, weights_only=True))
                    restored.eval()
                    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
                    torch.testing.assert_close(restored(inputs), restored(inputs), rtol=0, atol=0)
                # Batch and single-example inference agree (no graph mixing).
                one = model_inputs(name, g.select(slice(0, 1)), GraphBuilder(center=name == "center"))
                torch.testing.assert_close(model(inputs)[:1], model(one), atol=2e-6, rtol=1e-5)
                print(f"{name}: overfit MSE {before:.6g} -> {after:.6g}")


if __name__ == "__main__":
    unittest.main()

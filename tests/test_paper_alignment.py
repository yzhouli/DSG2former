import unittest

import numpy as np
import torch

from dsg2former.paper_features import HistoricalCorpus, popularity_difference
from dsg2former.paper_model import PaperDSG2former
from dsg2former.paper_optimizer import TensorFlowAdam


class PaperAlignmentTests(unittest.TestCase):
    def test_keras_adam_epsilon_hat_and_bias_correction(self):
        p = torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64))
        optimizer = TensorFlowAdam([p])
        expected, m, v = np.array([1., -2.]), np.zeros(2), np.zeros(2)
        for t, grad in enumerate([[1e-9, .2], [-3e-9, .1], [.5, .3]], 1):
            g = np.asarray(grad)
            m = .9 * m + .1 * g
            v = .999 * v + .001 * g*g
            expected -= .001 * np.sqrt(1-.999**t)/(1-.9**t) * m/(np.sqrt(v)+1e-7)
            p.grad = torch.tensor(grad, dtype=p.dtype)
            optimizer.step()
            np.testing.assert_allclose(p.detach().numpy(), expected, rtol=0, atol=1e-12)

    def test_available_ancestor_initializers_are_preserved(self):
        torch.manual_seed(42)
        model = PaperDSG2former()
        for unit in model.units:
            self.assertLessEqual(float(unit.attention.bias.weight.detach().abs().max()), .04)
            self.assertTrue(torch.equal(unit.attention.qkv.bias, torch.zeros_like(unit.attention.qkv.bias)))
            self.assertTrue(torch.equal(unit.attention.projection.bias, torch.zeros_like(unit.attention.projection.bias)))
            for module in unit.mlp:
                if isinstance(module, torch.nn.Linear):
                    self.assertLess(float(module.bias.detach().abs().max()), 1e-5)
        self.assertTrue(torch.equal(model.spatial_head.bias, torch.zeros(3)))

    def fixture(self):
        records = []
        for i, (u, t, parent, kind) in enumerate([("a", 0., None, "source"), ("b", 1., None, "source"),
                                                ("a", 2., "b", "reaction"), ("c", 3., "a", "reaction"),
                                                ("a", 4., "c", "reaction")]):
            records.append({"id": str(i), "uid": u, "timestamp": t, "text": "text", "kind": kind,
                            "reply_to_uid": parent, "reply_to_id": None, "archive_path": str(i)})
        embedding = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.]], dtype=np.float32)
        sentiment = np.array([[.8, .1, .1], [.2, .3, .5], [.1, .6, .3], [.1, .7, .2], [1., 0., 0.]], dtype=np.float32)
        return records, embedding, sentiment

    def test_six_features_follow_paper_formulas(self):
        c = HistoricalCorpus(*self.fixture(), top_k=1, gamma=.23)
        pop = popularity_difference(5, 1, 3600, 3600)
        self.assertEqual(pop, 2.)
        graph = c.graph(["a", "c"], {"b"}, 4., 1, pop)
        i = graph["node_ids"].index("a")
        f = graph["feature_components"]
        np.testing.assert_allclose(f["user_interest"][i], [.8, .1, .1])
        self.assertAlmostEqual(float(f["activity"][i]), 1.23, places=6)
        self.assertEqual(f["retweet_rate"][i], .5)
        self.assertAlmostEqual(float(f["authority"][i]), 1.5)
        self.assertEqual(f["friend_score"][i], .5)
        np.testing.assert_allclose(graph["drivers"][i, 0], np.array([.8, .1, .1]) * 1.23 * .5)
        np.testing.assert_allclose(graph["drivers"][i, 1], [1.5, 1.5, 1.5])
        # a -> c is observed at exactly the cutoff and must be unavailable.
        j = graph["node_ids"].index("c")
        self.assertEqual(graph["positions"][i, j], 0)

    def test_future_text_sentiment_and_edges_do_not_change_features(self):
        records, embeddings, sentiment = self.fixture()
        first = HistoricalCorpus(records, embeddings, sentiment, top_k=1).graph(["a", "c"], {"b"}, 4., 1, 2.)
        records[-1] = {**records[-1], "text": "arbitrary future text", "reply_to_uid": "b"}
        embeddings[-1] = [0., 1.]
        sentiment[-1] = [0., 0., 1.]
        second = HistoricalCorpus(records, embeddings, sentiment, top_k=1).graph(["a", "c"], {"b"}, 4., 1, 2.)
        np.testing.assert_array_equal(first["drivers"], second["drivers"])
        np.testing.assert_array_equal(first["positions"], second["positions"])

    def test_completed_window_right_end_is_observed_next_window(self):
        c = HistoricalCorpus(*self.fixture(), top_k=1)
        before = c.eligible("a", 4., 1)
        after = c.eligible("a", float(np.nextafter(4., np.inf)), 1)
        self.assertNotIn(4, before)
        self.assertIn(4, after)

    def test_empty_target_window_keeps_boolean_history_masks(self):
        c = HistoricalCorpus(*self.fixture())
        graph = c.graph([], {"a", "b"}, 4., 1, 0.)
        self.assertEqual(len(graph["node_ids"]), 2)
        self.assertEqual(graph["seen_before"].dtype, np.dtype(bool))
        self.assertEqual(int((~graph["seen_before"]).sum()), 0)
        self.assertEqual(graph["seen_before_chain"].dtype, np.dtype(bool))

    def test_source_copy_in_reactions_is_not_invented_as_a_share(self):
        records, embedding, sentiment = self.fixture()
        records[2].update(id="0", thread_id="0", reply_to_uid=None)
        corpus = HistoricalCorpus(records, embedding, sentiment)
        _, activity, rate, count = corpus.intrinsic("a", 3., 1)
        self.assertEqual(count, 2)  # Preserve both raw records.
        self.assertEqual(rate, 0.)
        self.assertAlmostEqual(activity, 2 * .23)

    def test_held_out_source_records_are_excluded_from_training_history(self):
        records, embeddings, sentiment = self.fixture()
        allowed = np.array([True, True, False, False, False])
        c = HistoricalCorpus(records, embeddings, sentiment, top_k=1, allowed_records=allowed)
        graph = c.graph(["a", "c"], {"b"}, 10., 1, 2.)
        i = graph["node_ids"].index("a")
        self.assertEqual(graph["feature_components"]["retweet_rate"][i], 0.)
        self.assertEqual(graph["edges"], 0)

    def test_temporal_state_is_previous_spatial_not_fused(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        model = PaperDSG2former().eval()
        drivers = torch.randn(1, 4, 2, 3)
        positions = torch.zeros(1, 4, 4, dtype=torch.long)
        positions[:, torch.arange(4), torch.arange(4)] = 1
        valid = torch.ones(1, 4, dtype=torch.bool)
        mask = valid.clone()
        previous = torch.randn(1, 4, 3)
        first, _, h = model.forward_batch(drivers, positions, valid, previous, mask)
        second, _, h2 = model.forward_batch(drivers, positions, valid, previous + 2, mask)
        torch.testing.assert_close(h, h2)
        self.assertFalse(torch.allclose(first, second))
        g1, g2 = model.gate(h), model.gate(previous)
        alpha = torch.exp(g1) / (torch.exp(g1) + torch.exp(g2))
        expected = model.temporal_head(alpha * h + (1 - alpha) * previous)
        torch.testing.assert_close(first, expected)

    def test_frozen_spatial_stage_only_updates_temporal_parameters(self):
        model = PaperDSG2former()
        model.freeze_spatial()
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(names)
        self.assertTrue(all(n.startswith(("gate.", "temporal_head.")) for n in names))

    def test_graph_padding_and_batch_isolation(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        model = PaperDSG2former().eval()
        drivers = torch.randn(2, 5, 2, 3)
        positions = torch.zeros(2, 5, 5, dtype=torch.long)
        positions[:, torch.arange(5), torch.arange(5)] = 1
        valid = torch.tensor([[True, True, False, False, False], [True] * 5])
        previous = torch.randn(2, 5, 3)
        with torch.no_grad():
            together = model.forward_batch(drivers, positions, valid, previous, valid)
            apart = model.forward_batch(drivers[:1, :2], positions[:1, :2, :2], valid[:1, :2], previous[:1, :2], valid[:1, :2])
        for a, b in zip(together, apart):
            torch.testing.assert_close(a[:1, :2], b, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()

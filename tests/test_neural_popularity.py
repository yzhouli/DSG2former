import copy
import unittest
import numpy as np
import torch
from dsg2former.neural_popularity import HistoryStatePopularityHead, final_micro_state, observed_history, pool_micro_state
from dsg2former.sds_aligned import PaperAlignedDSG2former


class NeuralPopularityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_history_excludes_current_target_and_distinguishes_missing_from_zero(self):
        s = {'window': 2, 'history_summary': np.r_[3, 0, 2, 2, np.zeros(6), 3, 0, np.log1p(3)], 'popularity_target': 5}
        seq, context = observed_history(s)
        b = copy.deepcopy(s); b['popularity_target'] = 999999
        np.testing.assert_array_equal(seq, observed_history(b)[0])
        np.testing.assert_array_equal(context, observed_history(b)[1])
        np.testing.assert_array_equal(seq[:, 1], [0, 0, 0, 0, 0, 0, 1, 1])
        self.assertEqual(seq[-1, 0], 0.)
        self.assertAlmostEqual(float(seq[-2, 0]), np.log1p(3), places=6)

    def test_temporal_final_state_agrees_with_existing_action_residual(self):
        torch.manual_seed(3)
        model = PaperAlignedDSG2former(input_dim=184, dim=12, dropout=0).eval()
        torch.nn.init.normal_(model.temporal_head.weight)
        z, old, logits = torch.randn(7, 48), torch.randn(7, 48), torch.randn(7, 3)
        mask = torch.tensor([True, False, True, True, False, False, True]); age = torch.rand(7)
        evolved = final_micro_state(model, z, old, mask, age, True)
        torch.testing.assert_close(logits + model.temporal_head(evolved-z), model.action_cached(z, logits, old, mask, age), rtol=0, atol=0)
        torch.testing.assert_close(evolved[~mask], z[~mask], rtol=0, atol=0)
        torch.testing.assert_close(final_micro_state(model, z, old, mask, age, False), z, rtol=0, atol=0)

    def test_both_history_and_final_state_receive_training_gradients(self):
        torch.manual_seed(9)
        head = HistoryStatePopularityHead(52, dropout=0)
        seq = torch.stack((torch.rand(4, 8), torch.ones(4, 8)), -1)
        state = torch.randn(4, 52, requires_grad=True); context = torch.rand(4, 2)
        prediction = head(seq, state, context)
        self.assertTrue(torch.all(prediction >= 0))
        torch.nn.functional.mse_loss(torch.log1p(prediction), torch.tensor([0., 1., 2., 0.])).backward()
        self.assertGreater(head.history.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(state.grad.abs().sum().item(), 0)

    def test_empty_history_is_finite_and_mixed_lengths_work(self):
        head = HistoryStatePopularityHead(52).eval()
        seq = torch.zeros(3, 8, 2); seq[1, -1, 1] = 1; seq[2, :, 1] = 1
        p = head(seq, torch.zeros(3, 52), torch.zeros(3, 2))
        self.assertTrue(torch.isfinite(p).all())
        self.assertEqual(p.shape, (3,))

    def test_same_two_block_contract_and_order_invariant_graph_pool(self):
        model = PaperAlignedDSG2former(input_dim=184, dim=12)
        self.assertEqual(len(model.encoder.blocks), 2)
        self.assertTrue(all(len(b.units) == 2 for b in model.encoder.blocks))
        z, logits = torch.randn(9, 48), torch.randn(9, 3)
        order = torch.randperm(9)
        torch.testing.assert_close(pool_micro_state(z, logits), pool_micro_state(z[order], logits[order]))


if __name__ == '__main__':
    unittest.main()

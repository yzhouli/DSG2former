import copy
import unittest

import numpy as np

from preprocess import validate
from test import classification
from tools.create_demo import fixture
from dsg2former.causal_user_state import VisibleUserStates


class PublicInterfaceTests(unittest.TestCase):
    def test_source_overlap_and_chronology_rejected(self):
        data, meta = fixture()
        stats = validate(data, meta)
        self.assertEqual(stats["train"]["items"], 8)
        meta[-1]["source"]["timestamp"] = 1
        with self.assertRaisesRegex(ValueError, "chronological"):
            validate(data, meta)

    def test_duplicate_supervision_rows_preserved(self):
        data, meta = fixture()
        s = data[0]["snapshots"][0]
        s["sample_nodes"] = np.array([0, 0, 1, 2])
        s["target"] = np.array([0, 0, 1, 2])
        self.assertEqual(validate(data, meta)["train"]["action_rows"], 97)

    def test_macro_candidates_and_nonfinite_features_rejected(self):
        data, meta = fixture()
        bad = copy.deepcopy(data)
        bad[0]["macro_snapshots"][0]["target"] = np.array([1])
        bad[0]["macro_snapshots"][0]["sample_nodes"] = np.array([0])
        with self.assertRaisesRegex(ValueError, "Macro graphs"):
            validate(bad, meta)
        data[0]["snapshots"][0]["x"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate(data, meta)

    def test_undefined_auc_not_invented(self):
        m = classification(np.array([0, 0]), np.array([[.8, .1, .1], [.6, .2, .2]]))
        self.assertEqual(m["accuracy"], 1.)
        self.assertEqual(m["auc"], [None, None, None])
        self.assertIsNone(m["macro_auc"])

    def test_same_time_and_later_partition_cache_excluded(self):
        store = VisibleUserStates()
        store.advance(10)
        store.observe("u", 2, "test")
        self.assertIsNone(store.read("u", 2))
        store.advance(11)
        self.assertIsNone(store.read("u", 0))
        self.assertEqual(store.read("u", 2), (10, "test"))


if __name__ == "__main__":
    unittest.main()

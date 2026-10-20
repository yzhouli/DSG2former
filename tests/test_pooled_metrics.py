import unittest
import numpy as np
from dsg2former.pooled_metrics import pooled_action_metrics,pooled_popularity_metrics


class PooledMetricTests(unittest.TestCase):
    def test_micro_scores_count_every_action_and_auc_uses_all_probability_entries(self):
        target=np.array([0,1,2,0]);probability=np.array([[.8,.1,.1],[.2,.7,.1],[.1,.1,.8],[.2,.6,.2]])
        result=pooled_action_metrics(target,probability)
        for key in ('accuracy','recall','f1'):self.assertEqual(result[key],.75)
        labels=np.eye(3,dtype=int)[target].ravel();scores=probability.ravel()
        positive,negative=scores[labels==1],scores[labels==0]
        expected=np.mean([(a>b)+.5*(a==b) for a in positive for b in negative])
        self.assertAlmostEqual(result['auc'],expected);self.assertEqual(result['action_rows'],4)
        changed=pooled_action_metrics(target,probability,target)
        self.assertEqual(changed['f1'],1.);self.assertEqual(changed['auc'],result['auc'])

    def test_macro_forecasts_are_pooled_before_square_root(self):
        result=pooled_popularity_metrics([0,3],[1,1])
        self.assertEqual(result['windows'],2);self.assertEqual(result['mse'],2.5)
        self.assertEqual(result['rmse'],np.sqrt(2.5));self.assertAlmostEqual(result['msle'],np.log(2)**2)


if __name__=='__main__':unittest.main()

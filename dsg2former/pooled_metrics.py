"""Pool all action rows / forecast windows within a run, without macro averages."""
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,f1_score,roc_auc_score,confusion_matrix
from .multitask_metrics import popularity_metrics


def pooled_action_metrics(target,probability,prediction=None):
    y=np.asarray(target,dtype=np.int64).reshape(-1);p=np.asarray(probability)
    if not len(y) or p.shape!=(len(y),3) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1.,atol=1e-5):
        raise ValueError('Expected nonempty aligned labels and normalized Nx3 probabilities')
    if not np.isin(y,[0,1,2]).all():raise ValueError('Unexpected action label')
    pred=p.argmax(1) if prediction is None else np.asarray(prediction,dtype=np.int64)
    if pred.shape!=y.shape or not np.isin(pred,[0,1,2]).all():raise ValueError('Unexpected discrete decisions')
    onehot=np.eye(3,dtype=np.int8)[y]
    scores={'accuracy':float(accuracy_score(y,pred)),
            'recall':float(recall_score(y,pred,labels=[0,1,2],average='micro',zero_division=0)),
            'f1':float(f1_score(y,pred,labels=[0,1,2],average='micro',zero_division=0)),
            'auc':float(roc_auc_score(onehot.ravel(),p.ravel())),
            'action_rows':len(y),'class_support':np.bincount(y,minlength=3).tolist(),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'averaging':'All action rows pooled; Recall/F1 micro; AUC micro OvR on flattened one-hot labels and original class probabilities'}
    np.testing.assert_allclose([scores['accuracy']]*2,[scores['recall'],scores['f1']],rtol=0,atol=1e-12)
    return scores


def pooled_popularity_metrics(target,prediction):
    scores=popularity_metrics(target,prediction)
    scores.update(windows=len(np.asarray(target).reshape(-1)),averaging='All information-window forecasts pooled in one array')
    return scores

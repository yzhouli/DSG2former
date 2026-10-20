"""KC-SIFRank and DeepLex, with explicit pretrained/text-resource adapters."""
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import torch
from torch import nn
from torch.nn import functional as F


def sif_phrase(vectors,frequency,smoothing=.001):
    """Conventional smoothed inverse-frequency aggregation used by SIFRank.

    TRS-GCN Eq.11 mixes vocabulary/phrase indices and cannot uniquely specify
    an executable weighting. This conventional interpretation is documented.
    """
    vectors=np.asarray(vectors,dtype=np.float64);p=np.asarray(frequency,dtype=np.float64)
    if len(vectors)==0:return np.zeros(vectors.shape[-1])
    weights=smoothing/(smoothing+p);return np.average(vectors,axis=0,weights=weights)


def spherical_spectral(vectors,groups=3,seed=42):
    x=normalize(np.asarray(vectors,dtype=np.float64));n=len(x)
    if n<=1:return np.zeros(n,dtype=int),x.copy()
    k=min(groups,n);similarity=np.maximum(x@x.T,0);np.fill_diagonal(similarity,0)
    l=laplacian(similarity,normed=True);_,v=eigh(l,subset_by_index=[0,k-1]);v=normalize(v)
    labels=KMeans(n_clusters=k,n_init=10,random_state=seed).fit_predict(v)
    centers=np.stack([x[labels==i].mean(0) if np.any(labels==i) else np.zeros(x.shape[1]) for i in range(k)])
    return labels,normalize(centers)


def best_keyword_cluster(vectors,core,groups=3,seed=42):
    if len(vectors)==0:return np.asarray(core,dtype=np.float64)
    x=normalize(np.asarray(vectors,dtype=np.float64));core=normalize(np.asarray(core,dtype=np.float64)[None])[0]
    labels,centers=spherical_spectral(x,groups,seed);scores=[]
    for i,center in enumerate(centers):
        members=x[labels==i];coherence=float(np.mean(members@center)) if len(members) else -1
        # Paper gives qualitative dual indicators without their scalar weights.
        # Equal semantic coherence/core-relevance is a declared interpretation.
        scores.append(.5*coherence+.5*float(center@core))
    return centers[int(np.argmax(scores))]


def temporal_relevance(a,b,delta_days,semantic_weight=.6,time_weight=.4,decay=.1):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);den=np.linalg.norm(a)*np.linalg.norm(b)
    cosine=float(a@b/den) if den else 0.
    return semantic_weight*cosine+time_weight*np.exp(-decay*abs(delta_days))


def sentiment_transition(current,previous):
    """Eq.24: 1 means unchanged sign; zero-product convention is 0."""
    return float(current*previous>0)


class DeepLex(nn.Module):
    """Lexicon-guided BiLSTM sentiment representation with TF-IDF weighting."""
    def __init__(self,embedding,hidden=16,classes=3):
        super().__init__();self.embedding=nn.Embedding.from_pretrained(torch.as_tensor(embedding,dtype=torch.float32),freeze=True,padding_idx=0)
        self.lstm=nn.LSTM(self.embedding.embedding_dim,hidden,batch_first=True,bidirectional=True);self.output=nn.Linear(hidden*2,classes)

    def forward(self,tokens,tfidf,lexicon):
        h,_=self.lstm(self.embedding(tokens));valid=tokens!=0
        # Sentiment lexicon anchors increase keyword weight; context still
        # enters through both directions of the recurrent encoder.
        weight=tfidf*(1+lexicon.abs())*valid;pooled=(h*weight[:,:,None]).sum(1)/weight.sum(1,keepdim=True).clamp_min(1e-8)
        return self.output(pooled),pooled


def lexical_sentiment(words,base_scores,probability,rational_weight=.5):
    """TRS-GCN Eqs.19-24 and degree/negation/fuzzy dictionaries."""
    positive=negative=0.;sentiment_words=0
    for i,(word,score) in enumerate(zip(words,base_scores)):
        if score==0:continue
        # The lexical anchor identifies sentiment-bearing words; neural
        # context confidence weights their intensity before degree modifiers.
        intensity=.8 if abs(score)>=.6 else .5
        score=intensity*float(probability[0]) if score>0 else -intensity*float(probability[1])
        sentiment_words+=1;context=words[max(0,i-3):i]
        if any(w in {'not','no','never','neither','without',"isn't","don't"} for w in context):score=-score
        if any(w in {'very','extremely','really','so','absolutely'} for w in context):score*=2
        if any(w in {'maybe','perhaps','possibly','probably','might'} for w in context):score*=.5
        if score>0:positive+=score
        else:negative+=score
    rational=float(rational_weight)
    if not 0<=rational<=1:raise ValueError('Training rational proportion must be in [0,1]')
    positive*=1+rational;negative*=rational
    if sentiment_words:positive/=sentiment_words;negative/=sentiment_words
    else:positive=float(probability[0]);negative=-float(probability[1])
    scale=max(abs(positive),abs(negative),1.);return positive/scale,negative/scale

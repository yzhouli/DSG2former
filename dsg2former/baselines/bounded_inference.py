"""Exact eval equations with bounded temporaries; no weights/topology changed.

Attention tiles query rows, retaining every key and the original row softmax.
TRS normalization uses a work copy, row reductions and in-place scaling. Both
paths are inference-only and validated against the unchanged native modules.
"""
from types import MethodType
import torch
from torch.nn import functional as F


def chunked_attention(self,x,adjacency,valid):
    if self.training or torch.is_grad_enabled():
        raise RuntimeError('Bounded attention is eval/no-grad only')
    b,n,_=x.shape
    h=self.projection(x).reshape(b,n,self.heads,self.hidden).transpose(1,2)
    left=(h*self.left[None,:,None]).sum(-1)
    right=(h*self.right[None,:,None]).sum(-1)
    outputs=[]
    for start in range(0,n,self.query_chunk_size):
        end=min(start+self.query_chunk_size,n)
        score=F.leaky_relu(left[:,:,start:end,None]+right[:,:,None,:],.2)
        permitted=adjacency[:,start:end,:]>0
        local=torch.arange(end-start,device=x.device)
        permitted[:,local,local+start]=True
        permitted&=valid[:,None,:]
        score.masked_fill_(~permitted[:,None],torch.finfo(score.dtype).min)
        out=score.softmax(-1)@h
        if self.concatenate:out=out.transpose(1,2).reshape(b,end-start,-1)
        else:out=out.mean(1)
        outputs.append(out)
    return torch.cat(outputs,dim=1)*valid[:,:,None]


def bounded_normalized_adjacency(adjacency,valid,row_chunk_size=4096):
    if torch.is_grad_enabled():raise RuntimeError('Bounded normalization is no-grad only')
    a=adjacency.clone()
    a.mul_(valid[:,:,None]);a.mul_(valid[:,None,:])
    a.diagonal(dim1=-2,dim2=-1).add_(valid.to(a.dtype))
    degrees=[a[:,i:i+row_chunk_size].abs().sum(-1) for i in range(0,a.shape[1],row_chunk_size)]
    d=torch.cat(degrees,dim=1).clamp_min(1).rsqrt()
    a.mul_(d[:,:,None]);a.mul_(d[:,None,:])
    return a


def bounded_trs_encode(self,x,views,valid,edge_features=None):
    if self.training or torch.is_grad_enabled():raise RuntimeError('Bounded TRS is eval/no-grad only')
    if edge_features is not None:raise ValueError('This benchmark adapter accepts materialized native graph views')
    representations=[]
    for view,first,second in zip(views.unbind(1),self.first,self.second):
        a=bounded_normalized_adjacency(view,valid,self.normalization_chunk_size)
        h=self.chebyshev(x,a,first,valid)
        h=self.chebyshev(h,a,second,valid)
        representations.append(h)
        # Do not retain the previous view's work copy while normalizing the next.
        del a
    h=torch.stack(representations,2)
    weights=self.view_score(h).softmax(2);self.last_view_attention=weights.detach()
    return (weights*h).sum(2)


def bounded_chebyshev(self,x,a,layers,valid):
    # Native ``-a @ x`` and ``-2*a @ t`` materialize an additional N*N
    # matrix in Python's operator order. Scale the much smaller product.
    ts=[x]
    if self.max_degree:ts.append(-(a@x))
    for _ in range(2,self.max_degree+1):ts.append((a@ts[-1]).mul_(-2).sub_(ts[-2]))
    out=sum(layer(t) for layer,t in zip(layers,ts))
    return torch.sigmoid(out)*valid[:,:,None]


def enable_bounded_inference(model,name,query_chunk_size=2048,normalization_chunk_size=4096):
    if name=='RAM-GAT':
        for layer in (model.gat1,model.gat2):
            layer.query_chunk_size=query_chunk_size
            layer.forward=MethodType(chunked_attention,layer)
    elif name=='TRS-GCN':
        model.normalization_chunk_size=normalization_chunk_size
        model.encode=MethodType(bounded_trs_encode,model)
        model.chebyshev=MethodType(bounded_chebyshev,model)
    else:raise ValueError(name)
    return model

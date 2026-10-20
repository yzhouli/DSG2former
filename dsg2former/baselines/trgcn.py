"""TRGCN paper equations and an explicit compatibility port of OSF author t24.

Native output is four-class graph veracity. ``forward`` exposes a separate
three-action node adapter; it must not be described as native rumor detection.
The paper and deposited code differ; both modes are traced in the spec.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def gcn_matrix(adjacency, valid):
    # A[target, source]; PyG's directed GCN normalization uses incoming degree.
    a=adjacency*valid[:,:,None]*valid[:,None,:]
    a=a.clone();a.diagonal(dim1=-2,dim2=-1).copy_(valid.to(a.dtype))
    degree=a.sum(-1);inv=degree.clamp_min(1).rsqrt()
    return a*inv[:,:,None]*inv[:,None,:]


class TRGCNAttention(nn.Module):
    def __init__(self,dim,latent,heads):
        super().__init__();assert latent%heads==0
        self.heads=heads;self.latent=latent
        self.W_q=nn.Linear(dim,latent);self.W_k=nn.Linear(dim,latent);self.W_v=nn.Linear(dim,latent);self.W_o=nn.Linear(latent,dim)
    def forward(self,query,key,valid=None):
        b,n,_=query.shape;h=self.heads;d=self.latent//h
        q,k,v=[layer(x).reshape(b,n,h,d).transpose(1,2) for layer,x in [(self.W_q,query),(self.W_k,key),(self.W_v,key)]]
        mask=None if valid is None else valid[:,None,None,:]
        z=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        return self.W_o(z.transpose(1,2).reshape(b,n,self.latent))


class TRGCNBlock(nn.Module):
    def __init__(self,dim,latent,heads,dropout,author=False):
        super().__init__();self.attention=TRGCNAttention(dim,latent,heads);self.layer_norm1=nn.LayerNorm(dim);self.layer_norm2=nn.LayerNorm(dim);self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.ReLU(),nn.Linear(4*dim,dim));self.dropout=dropout;self.author=author
    def forward(self,x,query,valid):
        h=self.attention(x if self.author else query,x,None if self.author else valid)
        x=self.layer_norm1(x+F.dropout(h,self.dropout,self.training))
        # Author t24 uses ReLU between linears; Eq16 prints ReLU after both.
        z=self.ffn(x) if self.author else F.relu(self.ffn[2](self.ffn[0](x)))
        x=self.layer_norm2(x+F.dropout(z,self.dropout,self.training))
        return x if self.author else x*valid[:,:,None]


class TRGCN(nn.Module):
    def __init__(self,input_dim=5000,dim=256,latent=128,heads=4,layers=2,classes=3,mode='paper'):
        super().__init__();assert mode in ['paper','author_t24'];self.mode=mode;self.dim=dim;self.input_dim=input_dim
        self.embedding=nn.Linear(input_dim,dim);self.conv1=nn.Linear(dim,dim);self.conv2=nn.Linear(dim,dim)
        self.blocks=nn.ModuleList([TRGCNBlock(dim,dim if mode=='author_t24' else latent,heads,0. if mode=='author_t24' else .2,mode=='author_t24') for _ in range(layers)])
        self.output=nn.Linear(dim,classes);self.output_norm=nn.LayerNorm(classes)
    def position(self,depth,dtype):
        frequency=torch.exp(torch.arange(0,self.dim,2,device=depth.device,dtype=dtype)*(-math.log(10000.)/self.dim));angle=depth[:,:,None].to(dtype)*frequency;out=torch.zeros(*depth.shape,self.dim,device=depth.device,dtype=dtype);out[:,:,0::2]=angle.sin();out[:,:,1::2]=angle.cos();return out
    def encode(self,x,adjacency,valid,depth=None):
        b,n=valid.shape
        if x.layout!=torch.strided:
            h=torch.sparse.mm(x,self.embedding.weight.T)+self.embedding.bias;h=h.reshape(b,n,-1)
        else:h=self.embedding(x)
        h=h.tanh()*valid[:,:,None]
        a=gcn_matrix(adjacency,valid)
        if self.mode=='paper':h=(adjacency@h+self.position(depth,h.dtype))*valid[:,:,None] # Eq4-8
        query=h
        h=h+F.dropout(F.relu(self.conv1(a@h)),p=.2,training=self.training)
        h=h*valid[:,:,None]
        for block in self.blocks:h=block(h,query,valid)
        h=h+F.dropout(F.relu(self.conv2(a@h)),p=.2,training=self.training)
        return h*valid[:,:,None]
    def classify(self,h):return self.output_norm(self.output(h))
    def forward(self,x,adjacency,valid,depth=None):
        h=self.encode(x,adjacency,valid,depth);return self.classify(h),h
    def native_graph(self,x,adjacency,valid,depth=None):
        h=self.encode(x,adjacency,valid,depth);mean=h.sum(1)/valid.sum(1,keepdim=True).clamp_min(1);return self.classify(mean).log_softmax(-1)

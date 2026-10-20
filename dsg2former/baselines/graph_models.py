"""GraphSAGE, sparse-domain RAM-GAT, and three-view TRS-GCN kernels."""
import torch
from torch import nn
from torch.nn import functional as F
from .native import TranSparse,triparty_game


def normalized_adjacency(adjacency, valid, self_loops=True):
    a=adjacency*valid[:,:,None]*valid[:,None,:]
    if self_loops:a=a+torch.diag_embed(valid.to(a.dtype))
    d=a.abs().sum(-1).clamp_min(1).rsqrt();return d[:,:,None]*a*d[:,None,:]


class KGraphSAGE(nn.Module):
    """TranSparse relation training followed by weighted mean GraphSAGE + softmax."""
    def __init__(self,input_dim=32,dim=16,hidden=32,classes=3,head_counts=(1,1,1,1,1),tail_counts=(1,1,1,1,1),seed=42):
        super().__init__();self.entity=nn.Linear(input_dim,dim);self.knowledge=TranSparse(dim,head_counts,tail_counts,seed=seed)
        self.layers=nn.ModuleList([nn.Linear(dim,hidden),nn.Linear(hidden,hidden)]);self.output=nn.Linear(hidden,classes);self.dim=hidden

    def encode(self,x,adjacency,valid):
        h=self.entity(x);a=adjacency+torch.diag_embed(valid.to(x.dtype));a=a*valid[:,:,None]*valid[:,None,:];a=a/a.sum(-1,keepdim=True).clamp_min(1)
        # GCN-style mean over self and neighbors is explicitly allowed by
        # GraphSAGE; Eq.19 duplicates neighbor symbol and leaves self ambiguous.
        for layer in self.layers:h=F.normalize(F.relu(layer(a@h)),dim=-1)*valid[:,:,None]
        return h

    def forward(self,x,adjacency,valid):
        h=self.encode(x,adjacency,valid);return self.output(h),h

    @staticmethod
    def neighborhood_loss(h,positive,negative):
        left=h[positive[:,0]];right=h[positive[:,1]];other=h[negative]
        return -F.logsigmoid((left*right).sum(-1)).mean()-F.logsigmoid(-(left[:,None]*other).sum(-1)).sum(-1).mean()


class GraphAttention(nn.Module):
    def __init__(self,input_dim,hidden,heads=2,dropout=.0006,concatenate=True):
        super().__init__();self.hidden=hidden;self.heads=heads;self.concatenate=concatenate;self.dropout=dropout
        self.projection=nn.Linear(input_dim,heads*hidden,bias=False);self.left=nn.Parameter(torch.empty(heads,hidden));self.right=nn.Parameter(torch.empty(heads,hidden));nn.init.xavier_uniform_(self.left);nn.init.xavier_uniform_(self.right)

    def forward(self,x,adjacency,valid):
        b,n,_=x.shape;h=self.projection(x).reshape(b,n,self.heads,self.hidden).transpose(1,2)
        score=F.leaky_relu((h*self.left[None,:,None]).sum(-1)[:,:,:,None]+(h*self.right[None,:,None]).sum(-1)[:,:,None,:],.2)
        permitted=(adjacency>0)|torch.eye(n,device=x.device,dtype=torch.bool)[None];permitted&=valid[:,None,:]
        score=score.masked_fill(~permitted[:,None],torch.finfo(score.dtype).min);attention=F.dropout(score.softmax(-1),p=self.dropout,training=self.training)
        out=attention@h
        if self.concatenate:out=out.transpose(1,2).reshape(b,n,-1)
        else:out=out.mean(1)
        return out*valid[:,:,None]


class RAMGAT(nn.Module):
    """Native four-class representation; final experiment classifier is SVM.

    SparseDictionary and GaussianTCA are fitted preprocessing modules. Native
    promotion proportions remain a separate input, never nonparticipation labels.
    """
    def __init__(self,input_dim=16,hidden=128,heads=2,classes=4,dropout=6**(-4),effect_dim=13,initialization='paper_interval'):
        super().__init__();self.effect=nn.Linear(effect_dim,1);self.gat1=GraphAttention(input_dim+3,hidden,heads,dropout);self.gat2=GraphAttention(hidden*heads,hidden,heads,dropout,False);self.output=nn.Linear(hidden,classes);self.dim=hidden;self.dropout=dropout
        if initialization=='paper_interval':
            # Table2 prints W^(0), W^(1) '(0,2)' without a distribution.
            # The literal uniform interval is an explicit assumption.
            for layer in [self.gat1,self.gat2]:nn.init.uniform_(layer.projection.weight,0.,2.)
        elif initialization!='torch_default':raise ValueError(initialization)

    def encode(self,x,adjacency,valid,effect_input,proportion):
        effect=self.effect(effect_input).squeeze(-1);game=triparty_game(effect,proportion)
        h=torch.cat((x,game),-1);h=F.dropout(F.elu(self.gat1(h,adjacency,valid)),p=self.dropout,training=self.training)
        return F.elu(self.gat2(h,adjacency,valid))

    def forward(self,x,adjacency,valid,effect_input,proportion):
        h=self.encode(x,adjacency,valid,effect_input,proportion);return self.output(h),h


class TRSGCN(nn.Module):
    """Three distinct graph views, two-hop messages, adaptive cross-graph fusion.

    Native binary click output; PHEME creates a separate three-class head.
    M and graph-attention equations are not fully given in the paper: this
    explicit Chebyshev degree-3 implementation follows TableIII max_degree.
    """
    def __init__(self,input_dim=32,hidden=16,classes=2,dropout=.5,max_degree=3):
        super().__init__();self.hidden=hidden;self.dim=hidden;self.dropout=dropout;self.max_degree=max_degree
        self.first=nn.ModuleList([nn.ModuleList([nn.Linear(input_dim,hidden,bias=False) for _ in range(max_degree+1)]) for _ in range(3)])
        self.second=nn.ModuleList([nn.ModuleList([nn.Linear(hidden,hidden,bias=False) for _ in range(max_degree+1)]) for _ in range(3)])
        self.view_score=nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1,bias=False));self.output=nn.Linear(hidden,classes)
        self.edge_coefficients=nn.Parameter(torch.tensor([.5,.5,.5]))
        self.last_view_attention=None

    def chebyshev(self,x,a,layers,valid):
        # L_tilde=-D^-1/2 A D^-1/2, with lambda_max=2.
        ts=[x]
        if self.max_degree:ts.append(-a@x)
        for k in range(2,self.max_degree+1):ts.append(-2*a@ts[-1]-ts[-2])
        out=sum(layer(t) for layer,t in zip(layers,ts));return torch.sigmoid(out)*valid[:,:,None]

    def encode(self,x,views,valid,edge_features=None):
        if edge_features is not None:
            # Explicit trainable Eq.26 and Eq.31 coefficients. Their coupling
            # is not specified in the paper; multiplication is documented.
            relevance,emotion,clicks,comments=edge_features.unbind(1)
            a,b,c=self.edge_coefficients
            views=views.clone();views[:,0]=(a*relevance+b*emotion)*(clicks+c*comments)
        representations=[]
        for view,first,second in zip(views.unbind(1),self.first,self.second):
            a=normalized_adjacency(view,valid);h=self.chebyshev(F.dropout(x,p=self.dropout,training=self.training),a,first,valid)
            h=self.chebyshev(F.dropout(h,p=self.dropout,training=self.training),a,second,valid);representations.append(h)
        h=torch.stack(representations,2);weights=self.view_score(h).softmax(2);self.last_view_attention=weights.detach();return (weights*h).sum(2)

    def forward(self,x,views,valid,edge_features=None):
        h=self.encode(x,views,valid,edge_features);return self.output(h),h

    @staticmethod
    def native_click(probability,threshold=.6):return probability[...,1]>=threshold


class CountAdapter(nn.Module):
    """Declared extension shared across baselines: mean H + five causal scalars."""
    def __init__(self,dim,hidden=12):
        super().__init__();self.network=nn.Sequential(nn.Linear(dim+5,hidden),nn.Tanh(),nn.Linear(hidden,1),nn.Softplus())
        for layer in self.network:
            if isinstance(layer,nn.Linear):nn.init.xavier_uniform_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,node_mean,history,count):
        observed=torch.cat((history,count[:,None]),-1)
        if torch.any(observed<0):raise ValueError('Nonnegative causal summaries required')
        return self.network(torch.cat((node_mean,torch.log1p(observed)),-1)).squeeze(-1)

"""Native mathematical modules reconstructed from the five downloaded papers.

Class counts are native defaults. PHEME adaptations instantiate separate heads.
Underspecified training choices live in the comparison plan, never hidden here.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch
from torch import nn
from torch.nn import functional as F


def binary_game(effect, proportion):
    payoff = effect * proportion
    delta = payoff[..., 0] - payoff[..., 1]
    return torch.stack((torch.sigmoid(delta), torch.sigmoid(-delta)), -1)


def triparty_game(effect, proportion):
    payoff = effect * proportion
    opponents = (payoff.sum(-1, keepdim=True) - payoff) / 2
    return torch.sigmoid(payoff-opponents)


def influence_kgraph(activity, perception, popularity, friend_drive):
    """K-GraphSAGE Eqs.8-9: both coefficients fixed to 0.5."""
    return .5*activity*perception + .5*popularity*friend_drive


def adaptive_steps(nests, minimum=.01, maximum=.5, best_index=0):
    distance = np.linalg.norm(np.asarray(nests)-nests[best_index], axis=1)
    return minimum+(maximum-minimum)*distance/max(float(distance.max()),1e-12)


def levy_steps(rng, shape, exponent=1.5):
    sigma = (math.gamma(1+exponent)*math.sin(math.pi*exponent/2)/
             (math.gamma((1+exponent)/2)*exponent*2**((exponent-1)/2)))**(1/exponent)
    return rng.normal(0,sigma,shape)/np.maximum(abs(rng.normal(size=shape)),1e-12)**(1/exponent)


@dataclass
class ImprovedCuckooSearch:
    """ICS Algorithm1: Levy proposal, adaptive distance step, worst abandonment."""
    nests: int = 8
    iterations: int = 10
    abandonment: float = .75
    minimum: float = .01
    maximum: float = .5
    seed: int = 42

    def optimize(self, objective, lower, upper, initial=None):
        rng=np.random.default_rng(self.seed);lo=np.asarray(lower);hi=np.asarray(upper)
        x=rng.uniform(lo,hi,(self.nests,len(lo)))
        if initial is not None:x[0]=np.clip(initial,lo,hi)
        value=np.array([objective(v) for v in x]);trace=[]
        for step in range(self.iterations):
            best=int(value.argmax());scale=adaptive_steps(x,self.minimum,self.maximum,best)
            candidate=np.clip(x+scale[:,None]*levy_steps(rng,x.shape),lo,hi)
            for proposal in candidate:
                score=objective(proposal);j=int(rng.integers(self.nests))
                if score>value[j]:x[j]=proposal;value[j]=score
            # Keep the best nest, replace the declared fraction of worst nests.
            best=int(value.argmax());order=np.argsort(value);count=min(self.nests-1,int(np.ceil(self.abandonment*self.nests)))
            for j in order[:count]:
                proposal=np.clip(x[best]+self.maximum*levy_steps(rng,(len(lo),)),lo,hi)
                score=objective(proposal)
                if score>value[j]:x[j]=proposal;value[j]=score
            best=int(value.argmax());trace.append({'iteration':step+1,'best_fitness':float(value[best]),'best_log2_parameters':x[best].tolist()})
        best=int(value.argmax());return x[best],trace


def rbf_svm(c, sigma, seed=42, probability=False):
    return SVC(C=float(c),kernel='rbf',gamma=1/(2*float(sigma)**2),probability=probability,random_state=seed,cache_size=512)


class TranSparse(nn.Module):
    """Separate sparse head/tail projections, K-GraphSAGE Eqs.15-18."""
    def __init__(self, dim, head_counts, tail_counts, theta_min=.5, seed=42):
        super().__init__();h=torch.as_tensor(head_counts,dtype=torch.float32);t=torch.as_tensor(tail_counts,dtype=torch.float32)
        self.register_buffer('head_sparsity',1-(1-theta_min)*h/h.max().clamp_min(1))
        self.register_buffer('tail_sparsity',1-(1-theta_min)*t/t.max().clamp_min(1))
        generator=torch.Generator().manual_seed(seed);masks=[]
        for values in [self.head_sparsity,self.tail_sparsity]:
            mask=torch.zeros(len(h),dim*dim)
            for i,s in enumerate(values):
                k=max(1,int(round((1-float(s))*dim*dim)));mask[i,torch.randperm(dim*dim,generator=generator)[:k]]=1
            masks.append(mask.reshape(len(h),dim,dim))
        self.register_buffer('head_mask',masks[0]);self.register_buffer('tail_mask',masks[1])
        self.head=nn.Parameter(torch.eye(dim).repeat(len(h),1,1));self.tail=nn.Parameter(torch.eye(dim).repeat(len(h),1,1));self.relation=nn.Parameter(torch.empty(len(h),dim));nn.init.xavier_uniform_(self.relation)

    def score(self, head, relation, tail):
        h=torch.einsum('...ij,...j->...i',(self.head*self.head_mask)[relation],head)
        t=torch.einsum('...ij,...j->...i',(self.tail*self.tail_mask)[relation],tail)
        return ((h+self.relation[relation]-t)**2).sum(-1)

    def margin_loss(self, head, relation, tail, negative_tail, margin=1.):
        return F.relu(margin+self.score(head,relation,tail)-self.score(head,relation,negative_tail)).mean()


class SparseDictionary:
    """Alternating dictionary learning + OMP, RAM-GAT Section4.1."""
    def __init__(self, components=16, nonzero=4, seed=42, iterations=40):
        self.components=components;self.nonzero=nonzero;self.seed=seed;self.iterations=iterations

    def fit(self, x):
        from sklearn.decomposition import MiniBatchDictionaryLearning
        self.model=MiniBatchDictionaryLearning(n_components=self.components,alpha=.1,max_iter=self.iterations,batch_size=256,
            fit_algorithm='cd',transform_algorithm='omp',transform_n_nonzero_coefs=self.nonzero,random_state=self.seed,n_jobs=1)
        self.model.fit(np.asarray(x,dtype=np.float64));return self

    def transform(self,x):return self.model.transform(np.asarray(x,dtype=np.float64)).astype(np.float32)


class GaussianTCA:
    """RBF TCA: MMD minimization with variance constraint; train-only fit.

    Caller can supply all points for exact TCA or fixed training landmarks.
    No held-out target features are used when fitting an inductive comparison.
    """
    def __init__(self, dim=16, sigma=1., regularization=1.):self.dim=dim;self.sigma=sigma;self.regularization=regularization

    def fit(self, source, target):
        self.landmarks=np.concatenate((source,target)).astype(np.float64);n,m=len(source),len(target);size=n+m
        if not n or not m:raise ValueError('TCA requires both training domains')
        kernel=rbf_kernel(self.landmarks,gamma=1/(2*self.sigma**2));e=np.r_[np.ones(n)/n,-np.ones(m)/m];mmd=np.outer(e,e);center=np.eye(size)-np.ones((size,size))/size
        lhs=kernel@mmd@kernel+self.regularization*np.eye(size);rhs=kernel@center@kernel+1e-8*np.eye(size)
        values,vectors=eigh(lhs,rhs,subset_by_index=[0,min(self.dim,size)-1]);self.projection=vectors;self.eigenvalues=values
        self.objective_mmd=float(e@kernel@self.projection@self.projection.T@kernel@e);return self

    def transform(self,x):
        z=rbf_kernel(np.asarray(x,dtype=np.float64),self.landmarks,gamma=1/(2*self.sigma**2))@self.projection
        return normalize(z).astype(np.float32)


class SDNE(nn.Module):
    """Weighted reconstruction + first-order proximity, Rumor-DPM Eqs.7-10."""
    def __init__(self, width=256, hidden=64, dim=16):
        super().__init__();self.width=width;self.encoder=nn.Sequential(nn.Linear(width,hidden),nn.ReLU(),nn.Linear(hidden,dim));self.decoder=nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Linear(hidden,width))

    def forward(self, adjacency):
        z=self.encoder(adjacency);return z,self.decoder(z)

    def loss(self, adjacency, valid, beta=5., alpha=.1, regularization=1e-5):
        z,reconstructed=self(adjacency);weights=torch.where(adjacency!=0,beta,1.);mask=valid[:,:adjacency.shape[1],None]&valid[:,None,:]
        second=(((reconstructed-adjacency)*weights)**2)[mask].mean()
        batch,left,right=adjacency.nonzero(as_tuple=True)
        distance=((z[batch,left]-z[batch,right])**2).sum(-1)
        first=(distance*adjacency[batch,left,right]).sum()/adjacency.sum().clamp_min(1)
        reg=sum(p.square().sum() for p in self.parameters());return first+alpha*second+regularization*reg


def spread_pixels(coordinates, side=16):
    """Injective nearest-grid outward diffusion; every input node retained."""
    xy=np.asarray(coordinates,dtype=float);n=len(xy)
    if n>side*side:raise ValueError('Pixel grid cannot hold every node')
    if not n:return np.empty((0,2),dtype=np.int64)
    maximum=np.max(abs(xy),axis=0);unit=max(float(maximum.max()),1e-12)
    scaled=np.rint((xy/unit+1)*(side-1)/2).astype(int);taken=set();answer=np.zeros((n,2),dtype=np.int64)
    # Stable input order; ties resolve clockwise from lower-right, inside out.
    for i,(x,y) in enumerate(scaled):
        candidates=[]
        for gx in range(side):
            for gy in range(side):
                radius=max(abs(gx-x),abs(gy-y));angle=(math.atan2(gy-y,gx-x)+2*math.pi)%(2*math.pi)
                candidates.append((radius,(gx-x)**2+(gy-y)**2,angle,gx,gy))
        for _,_,_,gx,gy in sorted(candidates):
            if (gx,gy) not in taken:taken.add((gx,gy));answer[i]=gx,gy;break
    return answer


class ImageCompensator(nn.Module):
    """GAN compensation for missing pixels; observed history is preserved."""
    def __init__(self, channels=5):
        super().__init__();self.generator=nn.Sequential(nn.Conv2d(channels*2,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,channels,3,padding=1),nn.Tanh())
        self.discriminator=nn.Sequential(nn.Conv2d(channels,16,3,stride=2,padding=1),nn.LeakyReLU(.2),nn.Conv2d(16,32,3,stride=2,padding=1),nn.LeakyReLU(.2),nn.Flatten(),nn.Linear(32*4*4,1))
        self.channels=channels

    def complete(self,image,observed,noise):
        generated=self.generator(torch.cat((image,noise),1));return torch.where(observed[:,None],image,generated)

    @staticmethod
    def losses(real_score,fake_score,literal_paper_equation=False):
        if literal_paper_equation:
            # Printed Eq.12 is unbounded in D. Exposed for audit, not silently
            # equated with the standard GAN minimax expression.
            return -F.logsigmoid(real_score).mean()+F.logsigmoid(fake_score).mean()-1,-F.logsigmoid(fake_score).mean()
        return F.softplus(-real_score).mean()+F.softplus(fake_score).mean(),F.softplus(-fake_score).mean()


class RumorDPMCNN(nn.Module):
    """TableIII A/B/C; native binary output; B is the default architecture."""
    def __init__(self, channels=7, classes=2, variant='B', side=16):
        super().__init__();self.classes=classes;self.side=side;first=16 if variant=='C' else 8
        modules=[nn.Conv2d(channels,first,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)]
        if variant!='A':modules.extend([nn.Conv2d(first,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)]);last=16;size=side//4
        else:last=first;size=side//2
        self.conv=nn.Sequential(*modules);hidden=512 if variant=='C' else 256;self.fc=nn.Linear(last*size*size,hidden);self.output=nn.Linear(hidden,side*side*classes);self.dim=hidden

    def forward(self,image):
        h=F.relu(self.fc(self.conv(image).flatten(1)));return self.output(h).reshape(-1,self.side,self.side,self.classes),h

"""Published SDSformer blocks (Eq21-28), separate from frozen legacy experiments.

Two pre-LN MHA/FFN residual units per block. Only shortest-path RPE and the
literal table-index mask enter attention; there is no neighbor aggregation.
FFN width4d/GELU, dropout, beta values, norm epsilon and ancestor initialization
remain explicit reconstruction choices where the PDF is underspecified.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from .model import AttentionUnit
from .enhanced_model import EnhancedDSG2former


def shortest_path_positions(adjacency,bias_size=50):
    """Directed distances+1, unreachable0; saturate at final lookup index."""
    a=np.asarray(adjacency)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Expected a finite square adjacency matrix')
    if bias_size<2:raise ValueError('RPE lookup needs at least two entries')
    distances=shortest_path(csr_matrix(a!=0),directed=True,unweighted=True)
    result=np.zeros(a.shape,dtype=np.int64);finite=np.isfinite(distances)
    result[finite]=np.minimum(distances[finite]+1,bias_size-1).astype(np.int64)
    return result


def align_graph_positions(data,bias_size=50):
    """In-memory overlay; keep all graph edges, features, rows, labels and splits.

    The old extension added source edges after calculating user distances.
    Recompute distances on the complete observed graph instead of mixing indices.
    """
    cache={};windows=changed=0
    for thread in data:
        for branch in ['snapshots','macro_snapshots']:
            for snapshot in thread[branch]:
                a=np.asarray(snapshot['adjacency'])!=0;key=(a.shape,a.tobytes())
                if key not in cache:cache[key]=shortest_path_positions(a,bias_size)
                position=cache[key];changed+=int(not np.array_equal(snapshot['positions'],position));snapshot['positions']=position;windows+=1
    return {'windows':windows,'changed_position_matrices':changed,'unique_graphs':len(cache),'directed':True,'bias_size':bias_size,'edges_features_labels_and_samples_unchanged':True}


class SDSformerBlock(nn.Module):
    """Eq21 MHA1 -> Eq22 FFN1 -> Eq23 MHA2 -> Eq24 FFN2, all pre-LN."""
    def __init__(self,dim,heads=3,bias_size=50,mask_rates=(1.,.5),dropout=0.,norm_eps=1e-6):
        super().__init__()
        if dim<=0 or heads<=0 or dim%heads:raise ValueError('Positive dimension must be divisible by heads')
        if bias_size<2 or len(mask_rates)!=2 or any(not math.isfinite(r) or not 0<r<=1 for r in mask_rates):raise ValueError('Expected two mask rates in (0,1] and a valid lookup size')
        self.units=nn.ModuleList([AttentionUnit(dim,heads,bias_size,rate,'rpe_mask',dropout) for rate in mask_rates])
        for unit in self.units:unit.norm1.eps=unit.norm2.eps=norm_eps

    def forward(self,h,positions,valid):
        b,n,dim=h.shape
        for unit in self.units:
            attention=unit.attention;z=unit.norm1(h)
            q,k,v=attention.qkv(z).reshape(b,n,3,attention.heads,dim//attention.heads).permute(2,0,3,1,4).unbind(0)
            # Eq25-27: add BOTH RPE and mask. Do not overwrite learned bias.
            bias=attention.bias(positions).permute(0,3,1,2)
            bias=bias+(positions>=attention.bias.num_embeddings*attention.mask_rate).to(h.dtype)[:,None]*-100.
            bias=bias.masked_fill(~valid[:,None,None,:],float('-inf'))
            update=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=attention.dropout if self.training else 0.)
            h=h+attention.projection(update.transpose(1,2).reshape(b,n,dim))
            h=h+unit.mlp(unit.norm2(h))
            h=h*valid[:,:,None]
        return h


class SDSformerEncoder(nn.Module):
    def __init__(self,dim,heads=3,blocks=2,bias_size=50,mask_rates=(1.,.5),dropout=0.,norm_eps=1e-6):
        super().__init__()
        if blocks!=2:raise ValueError('The aligned encoder uses the two SDSformer blocks specified in the PDF')
        self.bias_size=bias_size;self.dim=dim
        self.blocks=nn.ModuleList([SDSformerBlock(dim,heads,bias_size,mask_rates,dropout,norm_eps) for _ in range(blocks)])
        self.norm=nn.LayerNorm(dim,eps=norm_eps)
        # Match the documented CG_LDGR initializer used by PaperDSG2former.
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
        for block in self.blocks:
            for unit in block.units:
                for layer in unit.mlp:
                    if isinstance(layer,nn.Linear):nn.init.normal_(layer.bias,std=1e-6)
                nn.init.trunc_normal_(unit.attention.bias.weight,std=.02,a=-.04,b=.04)

    def forward(self,h,positions,valid):
        b,n,d=h.shape
        if d!=self.dim or positions.shape!=(b,n,n) or valid.shape!=(b,n) or valid.dtype!=torch.bool:raise ValueError('Misaligned node, relation or validity arrays')
        if positions.dtype!=torch.long or torch.any(positions<0) or torch.any(positions>=self.bias_size):raise ValueError('RPE indices must be integer lookup indices in range; no silent clamping')
        if torch.any(~valid.any(1)):raise ValueError('Each graph needs at least one observed node')
        h=h*valid[:,:,None]
        for block in self.blocks:h=block(h,positions,valid)
        return self.norm(h)*valid[:,:,None]


class PaperAlignedDSG2former(EnhancedDSG2former):
    """Keep enhanced feature/readout/temporal adapters outside the aligned blocks.

    This aligns SDSformer only. It does not claim the whole184feature extension,
    temporal residual, three-model ensemble or independent count model is native.
    """
    def __init__(self,input_dim=184,dim=48,heads=3,blocks=2,dropout=.15,hierarchical=False,game_enabled=True,bias_size=50,mask_rates=(1.,.5),norm_eps=1e-6):
        super().__init__(input_dim,dim,heads,0,dropout,False,hierarchical,game_enabled)
        del self.units,self.neighbor,self.norm,self.local
        self.encoder=SDSformerEncoder(dim,heads,blocks,bias_size,mask_rates,dropout,norm_eps)

    def spatial(self,x,drivers,positions,adjacency,role,valid,user_count):
        # adjacency is accepted for the established batch API. All graph influence
        # inside the encoder is through the precomputed shortest-path positions.
        raw=self.feature_projection(x);signal=torch.sign(drivers)*torch.log1p(drivers.abs());h=raw+self.role(role)
        if self.game_enabled:h=h+self.game_projection(self.game(signal))
        h=self.encoder(h,positions,valid);source=h[torch.arange(len(x),device=x.device),user_count][:,None].expand(-1,h.shape[1],-1)
        representation=torch.cat((h,raw,source,h*source),-1);logits=self.spatial_head(representation)
        return (self.hierarchy(logits) if self.hierarchical else logits),representation

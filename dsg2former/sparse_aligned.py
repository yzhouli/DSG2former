"""Edge-index attention variant; block parameters match the aligned dense model.

Attention topology is an explicit extension, not a literal paper reproduction.
The packed API never creates an N-by-N attention score or bias tensor. The
dense compatibility adapter still receives the old quadratic graph inputs.
"""
import math
from dataclasses import dataclass, fields, replace
import torch
from torch.nn import functional as F
from .sds_aligned import PaperAlignedDSG2former,SDSformerEncoder


def attention_edges(edge_index,edge_rpe,valid,source_index,global_source=True):
    """[batch,query,key] edges, self loops and optional bidirectional source links.

    Original RPE wins when a computational source link duplicates a real edge.
    New source links use direct-hop RPE2; self links use RPE1. Padding is removed.
    """
    if valid.ndim!=2 or valid.dtype!=torch.bool:raise ValueError('Expected boolean batch-node validity')
    batch,nodes=valid.shape
    if edge_index.dtype!=torch.long or edge_index.ndim!=2 or edge_index.shape[0]!=3:raise ValueError('Expected int64 edge_index with shape(3,E)')
    if edge_rpe.dtype!=torch.long or edge_rpe.shape!=(edge_index.shape[1],):raise ValueError('Expected one int64 RPE index per edge')
    if source_index.shape!=(batch,) or source_index.dtype!=torch.long:raise ValueError('Expected one int64 source index per graph')
    if torch.any(source_index<0) or torch.any(source_index>=nodes) or torch.any(~valid[torch.arange(batch,device=valid.device),source_index]):raise ValueError('Source token must be valid')
    if edge_index.numel():
        b,q,k=edge_index
        if torch.any(b<0) or torch.any(b>=batch) or torch.any(q<0) or torch.any(q>=nodes) or torch.any(k<0) or torch.any(k>=nodes):raise ValueError('Edge endpoint outside batch')
        keep=valid[b,q]&valid[b,k];edge_index=edge_index[:,keep];edge_rpe=edge_rpe[keep]
    b,q=valid.nonzero(as_tuple=True);extra=[torch.stack((b,q,q))];rpe=[torch.ones_like(q)]
    if global_source:
        source=source_index[b]
        extra.extend([torch.stack((b,q,source)),torch.stack((b,source,q))]);rpe.extend([torch.full_like(q,2),torch.full_like(q,2)])
    edges=torch.cat([edge_index,*extra],1);positions=torch.cat([edge_rpe,*rpe])
    key=edges[0]*nodes*nodes+edges[1]*nodes+edges[2];order=key.argsort(stable=True)
    sorted_key=key[order];keep=torch.ones_like(sorted_key,dtype=torch.bool);keep[1:]=sorted_key[1:]!=sorted_key[:-1]
    order=order[keep];edges=edges[:,order];positions=positions[order]
    positions=torch.where(edges[1]==edges[2],torch.ones_like(positions),positions)
    return edges,positions


def edges_from_dense(adjacency,positions,valid,source_index,global_source=True):
    if adjacency.shape!=positions.shape or adjacency.shape!=(valid.shape[0],valid.shape[1],valid.shape[1]):raise ValueError('Misaligned dense graph input')
    edges=(adjacency!=0).nonzero().T.contiguous();rpe=positions[tuple(edges)]
    return attention_edges(edges,rpe,valid,source_index,global_source)


@dataclass(frozen=True)
class PackedAttentionGraph:
    """Validated edge-only layout, reusable across blocks and ensemble members.

    Edges are computational query/key pairs. Combining duplicate pairs here
    does not deduplicate source action records or alter the underlying dataset.
    Treat the tensor buffers as immutable; rebuild when the topology changes.
    """
    edge_index: torch.Tensor
    edge_rpe: torch.Tensor
    valid: torch.Tensor
    source_index: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    counts: torch.Tensor
    bias_size: int
    chunk_row: torch.Tensor = None
    chunk_start: torch.Tensor = None
    chunk_end: torch.Tensor = None
    row_chunk_offsets: torch.Tensor = None
    max_chunks: int = 1

    def to(self,device):
        """Move a prevalidated layout without repeating value checks on GPU."""
        return replace(self,**{f.name:getattr(self,f.name).to(device) for f in fields(self)
                              if isinstance(getattr(self,f.name),torch.Tensor)})


def prepare_attention_graph(edge_index,edge_rpe,valid,source_index,global_source=True,bias_size=50):
    """Perform dynamic sorting/validation once, outside each neural forward.

    Input and output storage scale with nodes plus retained edges. No pairwise
    adjacency, shortest-path matrix, attention mask or score matrix is created.
    RPE of each supplied original edge must come from the causal graph builder.
    """
    if bias_size<2:raise ValueError('RPE table needs at least two entries')
    if any(t.device!=valid.device for t in (edge_index,edge_rpe,source_index)):
        raise ValueError('Packed graph tensors must share a device')
    if edge_rpe.numel() and (torch.any(edge_rpe<0) or torch.any(edge_rpe>=bias_size)):
        raise ValueError('RPE outside lookup range')
    edges,rpe=attention_edges(edge_index,edge_rpe,valid,source_index,global_source)
    n=valid.shape[1];query=edges[0]*n+edges[1];key=edges[0]*n+edges[2]
    counts=torch.bincount(query,minlength=valid.numel())
    # Fixed-size CSR chunks avoid materializing E-by-head-by-width Q/K/V arrays
    # in the optional fused inference kernel. All topology work stays outside
    # forward and can be shared by the four units and three ensemble members.
    chunks=[];starts=[];ends=[];offsets=[0];edge_offset=0;max_chunks=1
    for row,count in enumerate(counts.detach().cpu().tolist()):
        for start in range(0,count,32):
            chunks.append(row);starts.append(edge_offset+start);ends.append(edge_offset+min(start+32,count))
        max_chunks=max(max_chunks,len(chunks)-offsets[-1]);offsets.append(len(chunks));edge_offset+=count
    def index(values):return torch.tensor(values,dtype=torch.long,device=valid.device)
    return PackedAttentionGraph(edges,rpe,valid.detach().clone(),source_index.detach().clone(),
                                query,key,counts,bias_size,index(chunks),index(starts),index(ends),
                                index(offsets),max_chunks)


def edge_attention(attention,z,edge_index,edge_rpe,training,packed=None):
    """Exact softmax over each query's retained keys, with edge-only intermediates."""
    batch,nodes,dim=z.shape;heads=attention.heads;width=dim//heads
    q,k,v=attention.qkv(z).reshape(batch*nodes,3,heads,width).unbind(1)
    if packed is None:
        b,i,j=edge_index;query=b*nodes+i;key=b*nodes+j
        counts=torch.bincount(query,minlength=batch*nodes)
    else:
        query,key,counts=packed.query,packed.key,packed.counts
    score=(q[query]*k[key]).sum(-1)/math.sqrt(width)+attention.bias(edge_rpe)
    score=score+(edge_rpe>=attention.bias.num_embeddings*attention.mask_rate)[:,None].to(score.dtype)*-100.
    # The packed factory establishes sorted, nonnegative segment lengths with
    # exact total E. Avoid repeated host-synchronizing length checks on CUDA.
    unsafe=packed is not None
    maximum=torch.segment_reduce(score.detach(),reduce='max',lengths=counts,unsafe=unsafe)
    exponent=(score-maximum[query]).exp()
    normalizer=torch.segment_reduce(exponent,reduce='sum',lengths=counts,unsafe=unsafe)
    probability=exponent/normalizer[query].clamp_min(torch.finfo(z.dtype).tiny)
    probability=F.dropout(probability,p=attention.dropout,training=training)
    # Sorted segments avoid floating-point atomic scatter accumulation, which
    # can otherwise break repeatable full inference replays on CUDA.
    output=torch.segment_reduce(probability[:,:,None]*v[key],reduce='sum',lengths=counts,unsafe=unsafe)
    return attention.projection(output.reshape(batch,nodes,dim))


class SparseSDSformerEncoder(SDSformerEncoder):
    def forward_packed(self,h,graph):
        if h.shape[:2]!=graph.valid.shape or h.shape[-1]!=self.dim or graph.bias_size!=self.bias_size:
            raise ValueError('Packed graph and checkpoint dimensions differ')
        valid=graph.valid;h=h*valid[:,:,None]
        for block in self.blocks:
            for unit in block.units:
                z=unit.norm1(h)
                if (getattr(self,'inference_backend','reference')=='fused' and not self.training
                        and not torch.is_grad_enabled() and z.is_cuda and z.dtype==torch.float32
                        and not torch.is_autocast_enabled()):
                    from .sparse_fused import fused_edge_attention
                    update=fused_edge_attention(unit.attention,z,graph)
                else:
                    update=edge_attention(unit.attention,z,graph.edge_index,graph.edge_rpe,
                                          self.training,packed=graph)
                h=h+update
                h=h+unit.mlp(unit.norm2(h));h=h*valid[:,:,None]
        return self.norm(h)*valid[:,:,None]

    def forward(self,h,edge_index,edge_rpe,valid):
        if h.shape[:2]!=valid.shape or h.shape[-1]!=self.dim:raise ValueError('Misaligned node features')
        if torch.any(edge_rpe<0) or torch.any(edge_rpe>=self.bias_size):raise ValueError('RPE outside lookup range')
        query=edge_index[0]*h.shape[1]+edge_index[1]
        if torch.any(query[1:]<query[:-1]):raise ValueError('Packed edges must be sorted by graph and query')
        h=h*valid[:,:,None]
        for block in self.blocks:
            for unit in block.units:
                h=h+edge_attention(unit.attention,unit.norm1(h),edge_index,edge_rpe,self.training)
                h=h+unit.mlp(unit.norm2(h));h=h*valid[:,:,None]
        return self.norm(h)*valid[:,:,None]


class SparseDSG2former(PaperAlignedDSG2former):
    """Same184features/heads/two blocks; social+semantic+self+source attention."""
    def __init__(self,*args,global_source=True,inference_backend='reference',**kwargs):
        if inference_backend not in ('reference','fused'):raise ValueError('Unknown sparse inference backend')
        super().__init__(*args,**kwargs);self.global_source=global_source
        # Preserve exactly the dense constructor's initialization and RNG state.
        with torch.random.fork_rng(devices=[]):
            dense=self.encoder;block=dense.blocks[0]
            encoder=SparseSDSformerEncoder(dense.dim,block.units[0].attention.heads,2,dense.bias_size,
                tuple(u.attention.mask_rate for u in block.units),block.units[0].attention.dropout,block.units[0].norm1.eps)
            encoder.load_state_dict(dense.state_dict(),strict=True);self.encoder=encoder
        self.encoder.inference_backend=inference_backend

    def spatial_edges(self,x,drivers,edge_index,edge_rpe,role,valid,user_count):
        graph=prepare_attention_graph(edge_index,edge_rpe,valid,user_count,self.global_source,self.encoder.bias_size)
        return self.spatial_packed(x,drivers,graph,role)

    def spatial_packed(self,x,drivers,graph,role):
        """Training/inference API with no N-by-N tensors or dynamic edge packing."""
        raw=self.feature_projection(x);signal=torch.sign(drivers)*torch.log1p(drivers.abs())
        h=raw+self.role(role)
        if self.game_enabled:h=h+self.game_projection(self.game(signal))
        h=self.encoder.forward_packed(h,graph)
        source=h[torch.arange(len(x),device=x.device),graph.source_index][:,None].expand(-1,h.shape[1],-1)
        representation=torch.cat((h,raw,source,h*source),-1);logits=self.spatial_head(representation)
        return (self.hierarchy(logits) if self.hierarchical else logits),representation

    def spatial(self,x,drivers,positions,adjacency,role,valid,user_count):
        # Compatibility only. For linear graph-input storage, call spatial_edges.
        edges=(adjacency!=0).nonzero().T.contiguous();rpe=positions[tuple(edges)]
        return self.spatial_edges(x,drivers,edges,rpe,role,valid,user_count)

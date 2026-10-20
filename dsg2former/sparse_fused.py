"""FP32 inference-only fused CSR attention with the same sparse neighborhood.

Two kernels compute chunk softmax statistics and merge them per query/head.
This handles the high-degree information node without padding every user's
neighborhood to N. Training/CPU/autocast keep the differentiable reference path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _chunks(QKV,KEY,RPE,BIAS,ROWS,STARTS,ENDS,MAXIMUM,NORMALIZER,WEIGHTED,
            HEADS:tl.constexpr,WIDTH:tl.constexpr,BD:tl.constexpr,MASK_AT:tl.constexpr,
            SCALE:tl.constexpr):
    chunk=tl.program_id(0);head=tl.program_id(1)
    row=tl.load(ROWS+chunk);start=tl.load(STARTS+chunk);end=tl.load(ENDS+chunk)
    e=start+tl.arange(0,32);d=tl.arange(0,BD)
    keep=e<end
    key=tl.load(KEY+e,keep,other=0)
    rpe=tl.load(RPE+e,keep,other=0)
    q=tl.load(QKV+row*(3*HEADS*WIDTH)+head*WIDTH+d,d<WIDTH,other=0)
    k=tl.load(QKV+key[:,None]*(3*HEADS*WIDTH)+HEADS*WIDTH+head*WIDTH+d[None,:],
              keep[:,None] & (d[None,:]<WIDTH),other=0)
    bias=tl.load(BIAS+rpe*HEADS+head,keep,other=0)
    score=tl.sum(k*q[None,:],1)*SCALE+bias+tl.where(rpe>=MASK_AT,-100.,0.)
    score=tl.where(keep,score,float('-inf'))
    maximum=tl.max(score,0)
    exponent=tl.exp(score-maximum)
    normalizer=tl.sum(exponent,0)
    v=tl.load(QKV+key[:,None]*(3*HEADS*WIDTH)+2*HEADS*WIDTH+head*WIDTH+d[None,:],
              keep[:,None] & (d[None,:]<WIDTH),other=0)
    value=tl.sum(exponent[:,None]*v,0)
    tl.store(MAXIMUM+chunk*HEADS+head,maximum)
    tl.store(NORMALIZER+chunk*HEADS+head,normalizer)
    tl.store(WEIGHTED+(chunk*HEADS+head)*WIDTH+d,value,d<WIDTH)


@triton.jit
def _merge(MAXIMUM,NORMALIZER,WEIGHTED,OFFSETS,OUTPUT,
           HEADS:tl.constexpr,WIDTH:tl.constexpr,BD:tl.constexpr,BC:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    start=tl.load(OFFSETS+row);end=tl.load(OFFSETS+row+1);d=tl.arange(0,BD)
    if end==start:
        tl.store(OUTPUT+(row*HEADS+head)*WIDTH+d,0.,d<WIDTH)
    elif end==start+1:
        norm=tl.load(NORMALIZER+start*HEADS+head)
        single_value=tl.load(WEIGHTED+(start*HEADS+head)*WIDTH+d,d<WIDTH,other=0)
        tl.store(OUTPUT+(row*HEADS+head)*WIDTH+d,single_value/norm,d<WIDTH)
    else:
        c=start+tl.arange(0,BC);keep=c<end
        maximum=tl.load(MAXIMUM+c*HEADS+head,keep,other=float('-inf'))
        weight=tl.exp(maximum-tl.max(maximum,0))
        normalizer=tl.load(NORMALIZER+c*HEADS+head,keep,other=0)
        denom=tl.sum(weight*normalizer,0)
        chunk_values=tl.load(WEIGHTED+(c[:,None]*HEADS+head)*WIDTH+d[None,:],
                      keep[:,None] & (d[None,:]<WIDTH),other=0)
        output=tl.sum(weight[:,None]*chunk_values,0)/denom
        tl.store(OUTPUT+(row*HEADS+head)*WIDTH+d,output,d<WIDTH)


def fused_edge_attention(attention,z,graph):
    """Same FP32 QKV/bias/mask/softmax/value formula; reduction order may differ."""
    if z.device.type!='cuda' or z.dtype!=torch.float32 or torch.is_grad_enabled() or attention.training:
        raise ValueError('Fused sparse attention requires CUDA FP32 inference/no_grad')
    if graph.chunk_row is None:
        raise ValueError('Prepare chunked CSR metadata before fused inference')
    batch,nodes,dim=z.shape;heads=attention.heads;width=dim//heads
    qkv=attention.qkv(z)
    if qkv.dtype!=torch.float32:raise ValueError('Disable autocast for the FP32 fused backend')
    chunks=graph.chunk_row.numel()
    maximum=torch.empty((chunks,heads),device=z.device,dtype=z.dtype)
    normalizer=torch.empty_like(maximum)
    weighted=torch.empty((chunks,heads,width),device=z.device,dtype=z.dtype)
    output=torch.empty((batch*nodes,heads,width),device=z.device,dtype=z.dtype)
    bd=triton.next_power_of_2(width)
    _chunks[(chunks,heads)](qkv,graph.key,graph.edge_rpe,attention.bias.weight,
                           graph.chunk_row,graph.chunk_start,graph.chunk_end,
                           maximum,normalizer,weighted,HEADS=heads,WIDTH=width,BD=bd,
                           MASK_AT=attention.bias.num_embeddings*attention.mask_rate,
                           SCALE=width**-.5,num_warps=4,enable_fp_fusion=False)
    _merge[(batch*nodes,heads)](maximum,normalizer,weighted,graph.row_chunk_offsets,output,
                               HEADS=heads,WIDTH=width,BD=bd,
                               BC=triton.next_power_of_2(graph.max_chunks),num_warps=4,
                               enable_fp_fusion=False)
    return attention.projection(output.reshape(batch,nodes,dim))

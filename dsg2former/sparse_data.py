"""Pack existing causal snapshots without dense pairwise GPU inputs.

The stored snapshots still contain dense adjacency/RPE arrays; extracting their
edges is CPU preprocessing. Raw action rows and sample_nodes are never modified.
"""
import numpy as np
import torch

from .sparse_aligned import prepare_attention_graph


def pack_sparse_snapshots(batch,device='cpu',global_source=True,bias_size=50):
    """Return x, drivers, packed graph, role for model.spatial_packed.

    Accept the same [(thread, snapshot), ...] batches as the existing dense
    trainer; its gather function retains all repeated sample_nodes/action rows.
    The topology is prepared on CPU, then only O(B*N + E) tensors move to GPU.
    """
    if not batch:raise ValueError('Cannot pack an empty snapshot batch')
    snapshots=[s for _,s in batch]
    b=len(snapshots);n=max(len(s['x']) for s in snapshots)
    dim=np.asarray(snapshots[0]['x']).shape[1]
    x=torch.zeros(b,n,dim);drivers=torch.zeros(b,n,2,3)
    role=torch.zeros(b,n,dtype=torch.long);valid=torch.zeros(b,n,dtype=torch.bool)
    sources=[];edges=[];rpe=[]
    for i,s in enumerate(snapshots):
        size=len(s['x']);features=np.asarray(s['x']);signal=np.asarray(s['drivers'])
        adjacency=np.asarray(s['adjacency']);positions=np.asarray(s['positions'])
        if features.shape!=(size,dim) or signal.shape!=(size,2,3):
            raise ValueError('Snapshot features/drivers do not align')
        if adjacency.shape!=(size,size) or positions.shape!=adjacency.shape:
            raise ValueError('Snapshot adjacency/RPE do not align')
        if np.asarray(s['role']).shape!=(size,) or not 0<=s['user_count']<size:
            raise ValueError('Snapshot roles/source do not align')
        x[i,:size]=torch.as_tensor(features,dtype=torch.float32)
        drivers[i,:size]=torch.as_tensor(signal,dtype=torch.float32)
        role[i,:size]=torch.as_tensor(s['role'],dtype=torch.long);valid[i,:size]=True
        sources.append(s['user_count'])
        q,k=np.nonzero(adjacency)
        edges.append(torch.as_tensor(np.stack((np.full_like(q,i),q,k)),dtype=torch.long))
        rpe.append(torch.as_tensor(positions[q,k],dtype=torch.long))
    graph=prepare_attention_graph(torch.cat(edges,1),torch.cat(rpe),valid,
                                  torch.tensor(sources),global_source,bias_size)
    return x.to(device),drivers.to(device),graph.to(device),role.to(device)

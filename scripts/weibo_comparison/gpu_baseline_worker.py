"""Device-only bridge for existing native trainers, preserving CPU I/O contracts.

All parameters, equations, batch sizes, losses and stopping rules are unchanged.
Differentiable transfers let the existing CPU gather/loss code train CUDA modules.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value


class DeviceBridge:
    def __init__(self,core,device):
        self.core=core.to(device);self.device=device

    def __call__(self,*args,**kwargs):
        return move(self.core(*move(args,self.device),**move(kwargs,self.device)),'cpu')

    def __getattr__(self,name):
        value=getattr(self.core,name)
        if isinstance(value,torch.nn.Module):return DeviceBridge(value,self.device)
        if callable(value):
            def call(*args,**kwargs):
                return move(value(*move(args,self.device),**move(kwargs,self.device)),'cpu')
            return call
        return value


def enable(device,method):
    if method=='TRGCN':
        from scripts.trgcn_baseline import train as m
        original=m.TRGCN;m.TRGCN=lambda **kw:DeviceBridge(original(**kw),device)
    else:
        from scripts.baseline_comparison import train_method as m
        for name in ['KGraphSAGE','RAMGAT','TRSGCN','RumorDPMCNN']:
            original=getattr(m,name)
            setattr(m,name,lambda original=original,**kw:DeviceBridge(original(**kw),device))


def smoke(device):
    from dsg2former.baselines.graph_models import KGraphSAGE,RAMGAT,TRSGCN
    from dsg2former.baselines.native import RumorDPMCNN
    from dsg2former.baselines.trgcn import TRGCN
    b,n=2,4;x=torch.randn(b,n,32);adj=torch.ones(b,n,n);valid=torch.ones(b,n,dtype=torch.bool)
    models=[(KGraphSAGE(),(x,adj,valid)),
        (RAMGAT(classes=3),(x[:,:,:16],adj,valid,torch.zeros(b,n,3,13),torch.ones(b,n,3)/3)),
        (TRSGCN(classes=3),(x,adj[:,None].expand(-1,3,-1,-1),valid)),
        (RumorDPMCNN(channels=10,classes=3,variant='A'),(torch.randn(b,10,16,16),)),
        (TRGCN(input_dim=32),(x.reshape(-1,32).to_sparse(),adj,valid,torch.zeros(b,n,dtype=torch.long)))]
    for core,inputs in models:
        bridge=DeviceBridge(core,device);bridge.eval();logits,h=bridge(*inputs)
        assert logits.device.type=='cpu' and h.device.type=='cpu'
        logits.square().mean().backward();assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in bridge.parameters())
        state=bridge.state_dict();assert set(state)==set(core.state_dict())
        assert all(v.device.type=='cpu' for v in state.values());bridge.load_state_dict(state)
        if isinstance(core,KGraphSAGE):
            bridge.zero_grad();entity=bridge.entity(x);term=bridge.knowledge.margin_loss(entity[0,:2],torch.tensor([0,1]),entity[0,1:3],entity[0,2:4]);term.backward()
        print(type(core).__name__,'device bridge forward/backward/state roundtrip passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','test','replay','smoke']);p.add_argument('--root',type=Path)
    p.add_argument('--method');p.add_argument('--variant',default='paper');p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.mode=='smoke':smoke(a.device)
    else:
        from scripts.weibo_comparison import baselines
        from scripts.weibo_comparison.prepare_features import setup
        setup(a.root);enable(a.device,a.method)
        baselines.trgcn(a) if a.method=='TRGCN' else baselines.method(a)

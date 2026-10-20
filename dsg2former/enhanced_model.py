"""DSG2former+ extension: causal feature residuals and local/global graph blocks.

The published reconstruction remains unchanged in paper_model.py. This version
explicitly adds feature fusion, source conditioning, and a temporal residual.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .model import SentimentGame,AttentionUnit


class EnhancedDSG2former(nn.Module):
 def __init__(self,input_dim=184,dim=48,heads=3,blocks=2,dropout=.15,local=True,hierarchical=False,game_enabled=True):
  super().__init__();self.dim=dim;self.local=local;self.hierarchical=hierarchical;self.game_enabled=game_enabled
  self.game=SentimentGame();self.game.alpha.data.copy_(torch.tensor([0.,.1,.1]));self.feature_projection=nn.Sequential(nn.Linear(input_dim,dim),nn.LayerNorm(dim),nn.GELU(),nn.Dropout(dropout));self.game_projection=nn.Linear(3,dim,bias=False);self.role=nn.Embedding(2,dim)
  self.units=nn.ModuleList([AttentionUnit(dim,heads,50,rate,'rpe_mask',dropout) for _ in range(blocks) for rate in (1.,.5)]);self.neighbor=nn.ModuleList([nn.Linear(dim,dim,bias=False) for _ in self.units]);self.norm=nn.LayerNorm(dim);self.dropout=dropout
  self.spatial_head=nn.Sequential(nn.Linear(dim*4,dim*2),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*2,3))
  self.gate=nn.Sequential(nn.Linear(dim*4+1,dim),nn.Tanh(),nn.Linear(dim,1,bias=False));self.temporal_head=nn.Linear(dim*4,3,bias=False);nn.init.zeros_(self.temporal_head.weight)

 def spatial(self,x,drivers,positions,adjacency,role,valid,user_count):
  raw=self.feature_projection(x);signal=torch.sign(drivers)*torch.log1p(drivers.abs());h=raw+self.role(role)
  if self.game_enabled:h=h+self.game_projection(self.game(signal))
  adj=adjacency*valid[:,:,None]*valid[:,None,:];a=adj/adj.sum(-1,keepdim=True).clamp_min(1)
  b,n,d=h.shape
  for i,(unit,neighbor) in enumerate(zip(self.units,self.neighbor)):
   attention=unit.attention;z=unit.norm1(h);q,k,v=attention.qkv(z).reshape(b,n,3,attention.heads,d//attention.heads).permute(2,0,3,1,4).unbind(0);bias=attention.bias(positions.clamp(0,49)).permute(0,3,1,2)
   if self.local and i%2==0:bias=bias.masked_fill(((positions==0)|(positions>3))[:,None],-100.)
   bias=bias.masked_fill(~valid[:,None,None,:],float('-inf'));update=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=self.dropout if self.training else 0.);update=attention.projection(update.transpose(1,2).reshape(b,n,d))
   if self.local:update=update+neighbor(a@z)
   h=h+F.dropout(update,p=self.dropout,training=self.training);h=h+unit.mlp(unit.norm2(h));h=h*valid[:,:,None]
  h=self.norm(h);source=h[torch.arange(b,device=x.device),user_count];source=source[:,None].expand(-1,n,-1);representation=torch.cat((h,raw,source,h*source),-1)
  logits=self.spatial_head(representation)
  if self.hierarchical:logits=self.hierarchy(logits)
  return logits,representation

 @staticmethod
 def hierarchy(z):
  part=F.logsigmoid(z[...,0]);non=F.logsigmoid(-z[...,0]);stance=F.log_softmax(z[...,1:],-1);return torch.cat((part[...,None]+stance,non[...,None]),-1)

 def action_cached(self,current,spatial_logits,previous,mask,age):
  old=torch.where(mask[:,None],previous,current);delta=torch.log1p(age.clamp_min(0))[:,None];zero=torch.zeros_like(delta);alpha=torch.sigmoid(self.gate(torch.cat((current,zero),-1))-self.gate(torch.cat((old,delta),-1)));evolved=alpha*current+(1-alpha)*old
  # Missing history has exactly zero residual, even after temporal training.
  change=torch.where(mask[:,None],evolved-current,torch.zeros_like(current))
  return spatial_logits+self.temporal_head(change)


class CountForecast(nn.Module):
 """Direct or hurdle head over candidate-independent graph/history summaries."""
 def __init__(self,input_dim,hidden=64,dropout=.1,hurdle=False):
  super().__init__();self.hurdle=hurdle;self.network=nn.Sequential(nn.Linear(input_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2 if hurdle else 1))
 def forward(self,x):
  z=self.network(x)
  if self.hurdle:return torch.sigmoid(z[:,0])*(1+F.softplus(z[:,1])),z
  return F.softplus(z[:,0]),z

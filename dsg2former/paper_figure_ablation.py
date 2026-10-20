"""Explicit Fig.4/5/6 interventions on the existing two-block architecture.

Dense attention is retained for the PE experiment: sparse topology itself
would introduce a second structural bias in the no-PE control. The Fig.4
driver-only projection suppresses the entire 184-feature bypass, including
feature initializers used by temporal history bootstrap.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .sds_aligned import PaperAlignedDSG2former, SDSformerEncoder


class ScopedProjection(nn.Sequential):
    def forward(self, x):
        return super().forward(torch.zeros_like(x) if self.driver_only else x)


class PositionalEncoder(SDSformerEncoder):
    def forward(self, h, positions, valid):
        if self.encoding == 'rpe_mask':
            return super().forward(h, positions, valid)
        if self.encoding == 'ape':
            n,d=h.shape[1:]
            index=torch.arange(n,device=h.device,dtype=h.dtype)[:,None]
            frequency=torch.exp(torch.arange(0,d,2,device=h.device,dtype=h.dtype)*(-math.log(10000)/d))
            pe=torch.zeros(n,d,device=h.device,dtype=h.dtype)
            pe[:,0::2]=torch.sin(index*frequency)
            pe[:,1::2]=torch.cos(index*frequency[:d//2])
            h=h+pe
        h=h*valid[:,:,None]
        for block in self.blocks:
            for unit in block.units:
                attention=unit.attention;z=unit.norm1(h);b,n,d=z.shape
                q,k,v=attention.qkv(z).reshape(b,n,3,attention.heads,d//attention.heads).permute(2,0,3,1,4).unbind(0)
                if self.encoding=='rpe':
                    bias=attention.bias(positions).permute(0,3,1,2)
                    bias=bias.masked_fill(~valid[:,None,None,:],float('-inf'))
                else:
                    bias=torch.zeros(b,1,1,n,device=h.device,dtype=h.dtype)
                    bias=bias.masked_fill(~valid[:,None,None,:],float('-inf'))
                update=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,
                    dropout_p=attention.dropout if self.training else 0.)
                h=h+attention.projection(update.transpose(1,2).reshape(b,n,d))
                h=h+unit.mlp(unit.norm2(h));h=h*valid[:,:,None]
        return self.norm(h)*valid[:,:,None]


class FigureAblationDSG2former(PaperAlignedDSG2former):
    def __init__(self, feature='game', encoding='rpe_mask', temporal='dsm',
                 feature_scope='full', **options):
        if feature not in ('self','external','payoff','game'):
            raise ValueError('Unknown feature quantification')
        if encoding not in ('none','ape','rpe','rpe_mask'):
            raise ValueError('Unknown position encoding')
        if temporal not in ('dsm','rnn','gru','mha'):
            raise ValueError('Unknown temporal mechanism')
        if feature_scope not in ('full','driver_only'):
            raise ValueError('Unknown feature scope')
        super().__init__(**options)
        self.game.mode=feature
        self.temporal_mode=temporal
        self.feature_scope=feature_scope
        projection=ScopedProjection(*list(self.feature_projection.children()))
        projection.driver_only=feature_scope=='driver_only'
        self.feature_projection=projection
        # Preserve common initial weights and the RNG stream for the full
        # reference. Added temporal modules are the intervention being tested.
        with torch.random.fork_rng(devices=[]):
            old=self.encoder
            unit=old.blocks[0].units[0]
            encoder=PositionalEncoder(old.dim,unit.attention.heads,2,old.bias_size,
                tuple(u.attention.mask_rate for u in old.blocks[0].units),
                unit.attention.dropout,unit.norm1.eps)
            encoder.load_state_dict(old.state_dict(),strict=True)
        encoder.encoding=encoding;self.encoder=encoder
        size=4*self.encoder.dim
        if temporal=='rnn':self.temporal_cell=nn.RNNCell(size,size)
        elif temporal=='gru':self.temporal_cell=nn.GRUCell(size,size)
        elif temporal=='mha':
            self.temporal_attention=nn.MultiheadAttention(size,options.get('heads',3),
                dropout=options.get('dropout',.15),batch_first=True)

    def action_cached(self,current,spatial_logits,previous,mask,age):
        if self.temporal_mode=='dsm':
            return super().action_cached(current,spatial_logits,previous,mask,age)
        old=torch.where(mask[:,None],previous,current)
        if self.temporal_mode in ('rnn','gru'):
            evolved=self.temporal_cell(current,old)
        else:
            sequence=torch.stack((old,current),1)
            evolved=self.temporal_attention(current[:,None],sequence,sequence,
                                             need_weights=False)[0][:,0]
        change=torch.where(mask[:,None],evolved-current,torch.zeros_like(current))
        return spatial_logits+self.temporal_head(change)

    def freeze_for_temporal(self):
        prefixes=('temporal_head.', 'gate.') if self.temporal_mode=='dsm' else (
            'temporal_head.','temporal_cell.','temporal_attention.')
        for name,parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))

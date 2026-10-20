"""SDSformer with the identical184feature/source/graph interface as aligned DSG+.

Reuse its train/validation-selected spatial pretraining, as in the paper's
SDSformer -> frozen-space DSG2former protocol. Temporal modules are removed.
"""
import torch
from dsg2former.sds_aligned import PaperAlignedDSG2former


class SDSformerSameFeatures(PaperAlignedDSG2former):
    def __init__(self,**options):
        super().__init__(**options)
        del self.gate,self.temporal_head

    def action_cached(self,current,spatial_logits,previous=None,mask=None,age=None):
        return spatial_logits

    @classmethod
    def from_spatial_checkpoint(cls,checkpoint):
        if not isinstance(checkpoint,dict):checkpoint=torch.load(checkpoint,weights_only=False,map_location='cpu')
        if checkpoint.get('architecture')!='PaperAlignedDSG2former':raise ValueError('Expected the aligned spatial pretraining checkpoint')
        model=cls(**checkpoint['options']);state={k:v for k,v in checkpoint['state_dict'].items() if not k.startswith(('gate.','temporal_head.'))};model.load_state_dict(state,strict=True)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,state[key],rtol=0,atol=0)
        return model

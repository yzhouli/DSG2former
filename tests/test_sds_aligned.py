"""Independent equation, mask-boundary and pipeline tests for aligned SDSformer."""
import copy,unittest
from unittest.mock import patch
import numpy as np,torch
from torch.nn import functional as F
from scipy.special import erf
from dsg2former.sds_aligned import SDSformerEncoder,PaperAlignedDSG2former,shortest_path_positions,align_graph_positions
from dsg2former.paper_model import PaperDSG2former

class SDSAlignedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_four_units_against_independent_float64_numpy_equations(self):
        torch.manual_seed(14);m=SDSformerEncoder(12).double().eval();rng=np.random.default_rng(92);n=30;x=rng.normal(size=(n,12));positions=np.zeros((n,n),np.int64)
        for i in range(n):
            for j in range(i,n):positions[i,j]=j-i+1
        w={k:v.detach().numpy() for k,v in m.state_dict().items()}
        def linear(x,p):return x@w[p+'.weight'].T+w.get(p+'.bias',0.)
        def norm(x,p):return (x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-6)*w[p+'.weight']+w[p+'.bias']
        expected=x.copy()
        for b in range(2):
            for u,rate in enumerate([1.,.5]):
                prefix=f'blocks.{b}.units.{u}';q,k,v=linear(norm(expected,prefix+'.norm1'),prefix+'.attention.qkv').reshape(n,3,3,4).transpose(1,2,0,3);bias=w[prefix+'.attention.bias.weight'][positions].transpose(2,0,1);score=q@k.transpose(0,2,1)/np.sqrt(4)+bias+(positions>=50*rate)[None]*-100.;score-=score.max(-1,keepdims=True);p=np.exp(score);p/=p.sum(-1,keepdims=True);z=(p@v).transpose(1,0,2).reshape(n,12);expected+=linear(z,prefix+'.attention.projection');z=linear(norm(expected,prefix+'.norm2'),prefix+'.mlp.0');z=.5*z*(1+erf(z/np.sqrt(2)));expected+=linear(z,prefix+'.mlp.3')
        expected=norm(expected,'norm')
        with torch.no_grad():actual=m(torch.tensor(x)[None],torch.tensor(positions)[None],torch.ones(1,n,dtype=torch.bool))
        np.testing.assert_allclose(actual[0].numpy(),expected,rtol=2e-11,atol=2e-11)

    def test_encoder_matches_existing_paper_reconstruction_with_same_weights(self):
        torch.manual_seed(8);paper=PaperDSG2former(dim=12).eval();aligned=SDSformerEncoder(12).eval()
        for i,unit in enumerate(paper.units):aligned.blocks[i//2].units[i%2].load_state_dict(unit.state_dict())
        aligned.norm.load_state_dict(paper.norm.state_dict());drivers=torch.randn(2,5,2,3);position=torch.randint(0,40,(2,5,5));valid=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
        with torch.no_grad():expected=paper.forward_batch(drivers,position,valid,torch.zeros(2,5,12),torch.zeros(2,5,dtype=torch.bool))[-1];actual=aligned(paper.embedding(paper.game(drivers)),position,valid)
        torch.testing.assert_close(actual[valid],expected[valid],rtol=1e-6,atol=1e-6)

    def test_mask_boundary_adds_rpe_and_keeps_unreachable_zero_visible(self):
        m=SDSformerEncoder(6).eval();position=torch.tensor([[[1,0,24,25],[0,1,24,25],[0,24,1,25],[0,24,25,1]]]);valid=torch.tensor([[1,1,1,0]],dtype=torch.bool);captured=[];original=F.scaled_dot_product_attention
        def record(q,k,v,**kw):captured.append(kw['attn_mask'].detach().clone());return original(q,k,v,**kw)
        with patch('dsg2former.sds_aligned.F.scaled_dot_product_attention',side_effect=record):m(torch.randn(1,4,6),position,valid)
        self.assertEqual(len(captured),4)
        # Use a valid key at25 for an explicit half-mask boundary as well.
        position[0,0,2]=25;captured.clear()
        with patch('dsg2former.sds_aligned.F.scaled_dot_product_attention',side_effect=record):m(torch.randn(1,4,6),position,valid)
        for i,bias in enumerate(captured):
            unit=m.blocks[i//2].units[i%2];rpe=unit.attention.bias(position).permute(0,3,1,2).detach();torch.testing.assert_close(bias[0,:,0,1],rpe[0,:,0,1]);torch.testing.assert_close(bias[0,:,0,2],rpe[0,:,0,2]-(100. if i%2 else 0.));self.assertTrue(torch.isneginf(bias[:,:,:,-1]).all())

    def test_padding_and_batch_composition_cannot_change_valid_outputs(self):
        torch.manual_seed(19);m=SDSformerEncoder(12).eval();x=torch.randn(1,3,12);pos=torch.ones(1,3,3,dtype=torch.long);valid=torch.ones(1,3,dtype=torch.bool)
        with torch.no_grad():
            a=m(x,pos,valid);xx=F.pad(x,(0,0,0,4),value=999);pp=F.pad(pos,(0,4,0,4));vv=F.pad(valid,(0,4));b=m(xx,pp,vv);c=m(torch.cat([xx,torch.randn_like(xx)*100]),torch.cat([pp,pp]),torch.cat([vv,vv]))
        torch.testing.assert_close(a,b[:,:3],atol=1e-6,rtol=1e-6);torch.testing.assert_close(a,c[:1,:3],atol=1e-6,rtol=1e-6);self.assertTrue((b[:,3:]==0).all())

    def test_directed_shortest_path_shift_disconnected_and_saturation(self):
        a=np.zeros((55,55),np.float32);a[np.arange(53),np.arange(1,54)]=1;p=shortest_path_positions(a);self.assertEqual(p[0,0],1);self.assertEqual(p[0,1],2);self.assertEqual(p[0,24],25);self.assertEqual(p[0,53],49);self.assertEqual(p[1,0],0);self.assertEqual(p[0,54],0)

    def test_position_overlay_corrects_source_shortcuts_without_changing_data(self):
        snapshot={'positions':np.array([[1,0,2],[0,1,2],[2,2,1]]),'adjacency':np.array([[1,0,1],[0,1,1],[1,1,1]]),'target':np.array([0,2,0]),'sample_nodes':np.array([0,1,0]),'x':np.ones((3,12))};data=[{'thread_id':'a','split':'test','snapshots':[snapshot],'macro_snapshots':[copy.deepcopy(snapshot)]}];before=copy.deepcopy(data);audit=align_graph_positions(data);self.assertEqual(data[0]['snapshots'][0]['positions'][0,1],3);self.assertEqual(audit['changed_position_matrices'],2)
        for branch in ['snapshots','macro_snapshots']:
            for k in ['adjacency','target','sample_nodes','x']:np.testing.assert_array_equal(data[0][branch][0][k],before[0][branch][0][k])

    def test_aligned_extension_has_no_neighbor_branch_and_uses_only_rpe_for_graph(self):
        torch.manual_seed(4);m=PaperAlignedDSG2former(input_dim=12,dim=12,dropout=0.).eval();x=torch.randn(1,4,12);drivers=torch.randn(1,4,2,3);pos=torch.ones(1,4,4,dtype=torch.long);role=torch.tensor([[0,0,0,1]]);valid=torch.ones(1,4,dtype=torch.bool);count=torch.tensor([3]);args=[x,drivers,pos,torch.ones(1,4,4),role,valid,count];a,h=m.spatial(*args);args[3]=torch.randn(1,4,4)*999;b,_=m.spatial(*args);torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertFalse(any('neighbor' in k for k in m.state_dict()));a[:,:3].square().mean().backward()
        for p in [m.encoder.blocks[0].units[0].attention.qkv.weight,m.feature_projection[0].weight,m.game.alpha]:self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.abs().sum()),0.)
        m.temporal_head.weight.data.normal_();z=h[0,:3];logits=a[0,:3];actual=m.action_cached(z,logits,torch.randn_like(z),torch.zeros(3,dtype=torch.bool),torch.ones(3));torch.testing.assert_close(actual,logits,rtol=0,atol=0)

    def test_invalid_lookup_indices_rejected_instead_of_silently_clamped(self):
        m=SDSformerEncoder(6);x=torch.ones(1,2,6);v=torch.ones(1,2,dtype=torch.bool)
        for bad in [-1,50]:
            with self.assertRaises(ValueError):m(x,torch.full((1,2,2),bad),v)
        with self.assertRaises(ValueError):SDSformerEncoder(6,blocks=1)
if __name__=='__main__':unittest.main()

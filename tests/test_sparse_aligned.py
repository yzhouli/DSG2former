import copy
import unittest
from unittest.mock import patch
import torch
from dsg2former.sds_aligned import PaperAlignedDSG2former
from dsg2former.sparse_aligned import SparseDSG2former,attention_edges,edge_attention,prepare_attention_graph
from dsg2former.attention_backend import make_model,model_from_checkpoint
from dsg2former.sparse_data import pack_sparse_snapshots


class SparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def inputs(self,n=5):
        return [torch.randn(1,n,12,dtype=torch.float64),torch.randn(1,n,2,3,dtype=torch.float64),
            torch.ones(1,n,n,dtype=torch.long)+torch.logical_not(torch.eye(n,dtype=torch.bool))[None],
            torch.ones(1,n,n,dtype=torch.float64),torch.tensor([[0]*(n-1)+[1]]),
            torch.ones(1,n,dtype=torch.bool),torch.tensor([n-1])]

    def test_full_edges_match_dense_outputs_and_parameter_gradients(self):
        torch.manual_seed(3);a=PaperAlignedDSG2former(input_dim=12,dim=12,dropout=0.).double().eval()
        b=SparseDSG2former(input_dim=12,dim=12,dropout=0.).double().eval();b.load_state_dict(a.state_dict(),strict=True)
        inputs=self.inputs();inputs[2]=torch.randint(0,50,inputs[2].shape);inputs[2][:,torch.arange(5),torch.arange(5)]=1
        x,ha=a.spatial(*inputs);y,hb=b.spatial(*inputs)
        torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-11);torch.testing.assert_close(ha,hb,atol=1e-11,rtol=1e-11)
        x.square().sum().backward();y.square().sum().backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb)
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,atol=1e-9,rtol=1e-9)

    def test_edge_only_api_avoids_dense_attention_and_has_linear_edge_count(self):
        n=100;valid=torch.ones(1,n,dtype=torch.bool);i=torch.arange(n-1)
        edges=torch.stack((torch.zeros(2*(n-1),dtype=torch.long),torch.cat((i,i+1)),torch.cat((i+1,i))))
        packed,pos=attention_edges(edges,torch.full((edges.shape[1],),2),valid,torch.tensor([n-1]))
        self.assertLessEqual(packed.shape[1],5*n)
        m=SparseDSG2former(input_dim=12,dim=12,dropout=0.).eval()
        with patch('torch.nn.functional.scaled_dot_product_attention',side_effect=AssertionError('Dense attention called')):
            y,_=m.spatial_edges(torch.randn(1,n,12),torch.zeros(1,n,2,3),packed,pos,torch.zeros(1,n,dtype=torch.long),valid,torch.tensor([n-1]))
        self.assertTrue(torch.isfinite(y).all())

    def test_padding_cannot_change_valid_nodes(self):
        torch.manual_seed(13);m=SparseDSG2former(input_dim=12,dim=12,dropout=0.).double().eval();args=self.inputs(4)
        original=m.spatial(*args)[0]
        x,d,p,a,r,v,c=args
        padded=[torch.nn.functional.pad(x,(0,0,0,3),value=999),torch.nn.functional.pad(d,(0,0,0,0,0,3),value=999),
            torch.nn.functional.pad(p,(0,3,0,3)),torch.nn.functional.pad(a,(0,3,0,3),value=1),
            torch.nn.functional.pad(r,(0,3)),torch.nn.functional.pad(v,(0,3)),c]
        actual=m.spatial(*padded)[0]
        torch.testing.assert_close(original,actual[:,:4],atol=1e-11,rtol=1e-11)

    def test_local_edge_softmax_matches_independent_dense_mask_reference(self):
        torch.manual_seed(27);m=SparseDSG2former(input_dim=12,dim=12,dropout=0.).double().eval()
        attention=m.encoder.blocks[0].units[1].attention;z=torch.randn(1,5,12,dtype=torch.float64)
        edges=torch.tensor([[0,0],[0,1],[1,3]]);rpe=torch.tensor([25,2]);valid=torch.ones(1,5,dtype=torch.bool)
        edges,rpe=attention_edges(edges,rpe,valid,torch.tensor([4]))
        bias=torch.full((1,3,5,5),float('-inf'),dtype=torch.float64)
        for e in range(edges.shape[1]):
            b,q,k=edges[:,e];bias[b,:,q,k]=attention.bias(rpe[e])-(100. if rpe[e]>=25 else 0.)
        q,k,v=attention.qkv(z).reshape(1,5,3,3,4).permute(2,0,3,1,4).unbind(0)
        reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=bias)
        reference=attention.projection(reference.transpose(1,2).reshape(1,5,12))
        actual=edge_attention(attention,z,edges,rpe,False)
        torch.testing.assert_close(reference,actual,atol=1e-11,rtol=1e-11)

    def test_state_keys_parameter_counts_and_initialization_match(self):
        torch.manual_seed(9);a=PaperAlignedDSG2former(input_dim=12,dim=12);after=torch.get_rng_state()
        torch.manual_seed(9);b=SparseDSG2former(input_dim=12,dim=12)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()));self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        for k in a.state_dict():torch.testing.assert_close(a.state_dict()[k],b.state_dict()[k],atol=0,rtol=0)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))

    def test_prepared_edges_match_forward_and_backward_without_dynamic_packing(self):
        torch.manual_seed(33)
        dense=PaperAlignedDSG2former(input_dim=12,dim=12,dropout=0.).double().eval()
        sparse=SparseDSG2former(input_dim=12,dim=12,dropout=0.).double().eval()
        sparse.load_state_dict(dense.state_dict(),strict=True)
        x,drivers,pos,adj,role,valid,source=self.inputs(5)
        edges=adj.nonzero().T; rpe=pos[tuple(edges)]
        graph=prepare_attention_graph(edges,rpe,valid,source)
        with patch('torch.bincount',side_effect=AssertionError('Dynamic counts in forward')):
            actual,state=sparse.spatial_packed(x,drivers,graph,role)
        expected,reference=dense.spatial(x,drivers,pos,adj,role,valid,source)
        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
        torch.testing.assert_close(state,reference,atol=1e-11,rtol=1e-11)
        actual.square().sum().backward();expected.square().sum().backward()
        for (name,a),(_,b) in zip(dense.named_parameters(),sparse.named_parameters()):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=1e-9,rtol=1e-9)

    def test_explicit_backend_and_checkpoint_metadata(self):
        model=make_model('sparse',input_dim=12,dim=12)
        cp=dict(options=dict(input_dim=12,dim=12),state_dict=model.state_dict(),attention_backend='sparse')
        self.assertIsInstance(model_from_checkpoint(cp),SparseDSG2former)
        self.assertNotIsInstance(model_from_checkpoint(cp,attention='dense'),SparseDSG2former)
        self.assertNotIsInstance(make_model(input_dim=12,dim=12),SparseDSG2former)
        with self.assertRaises(ValueError):make_model('unknown')

    def test_prepared_graph_rejects_invalid_rpe(self):
        valid=torch.ones(1,2,dtype=torch.bool)
        edges=torch.tensor([[0],[0],[1]])
        for bad in (-1,50):
            with self.assertRaises(ValueError):
                prepare_attention_graph(edges,torch.tensor([bad]),valid,torch.tensor([1]))

    def test_snapshot_packing_preserves_padding_and_duplicate_action_rows(self):
        import numpy as np
        from scripts.model_improvement.train_spatial import pack,gather
        torch.manual_seed(17);batch=[]
        for n in (4,6):
            x,drivers,pos,adj,role,valid,source=self.inputs(n)
            s=dict(x=x[0].float().numpy(),drivers=drivers[0].float().numpy(),
                   positions=pos[0].numpy(),adjacency=adj[0].float().numpy(),
                   role=role[0].numpy(),user_count=n-1,sample_nodes=np.array([0,0,1]),
                   target=np.array([0,1,2]))
            batch.append((dict(thread_id=str(n)),s))
        before=copy.deepcopy(batch)
        model=make_model('sparse',input_dim=12,dim=12,dropout=0.).eval()
        packed=pack_sparse_snapshots(batch)
        actual,state=model.spatial_packed(*packed)
        expected,_=model.spatial(*pack(batch))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        logits,_,target,_=gather(actual,state,batch)
        self.assertEqual(len(target),6)
        torch.testing.assert_close(logits[0],logits[1],rtol=0,atol=0)
        for (_,s),(_,original) in zip(batch,before):
            for k in ('sample_nodes','target','adjacency','positions'):
                np.testing.assert_array_equal(s[k],original[k])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_packed_forward_gradients_and_capture(self):
        torch.manual_seed(42)
        dense=PaperAlignedDSG2former(input_dim=12,dim=12,dropout=0.).cuda().eval()
        sparse=SparseDSG2former(input_dim=12,dim=12,dropout=0.).cuda().eval()
        sparse.load_state_dict(dense.state_dict(),strict=True)
        x,drivers,pos,adj,role,valid,source=[t.cuda() for t in self.inputs()]
        x,drivers,adj=x.float(),drivers.float(),adj.float()
        edges=adj.nonzero().T
        graph=prepare_attention_graph(edges,pos[tuple(edges)],valid,source)
        expected=dense.spatial(x,drivers,pos,adj,role,valid,source)[0]
        actual=sparse.spatial_packed(x,drivers,graph,role)[0]
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        actual.square().mean().backward();expected.square().mean().backward()
        for (_,a),(_,b) in zip(dense.named_parameters(),sparse.named_parameters()):
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-4,rtol=3e-4)
        with torch.inference_mode():
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):sparse.spatial_packed(x,drivers,graph,role)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            capture=torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):output=sparse.spatial_packed(x,drivers,graph,role)[0]
            for _ in range(3):
                capture.replay();torch.testing.assert_close(output,expected,atol=1e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()

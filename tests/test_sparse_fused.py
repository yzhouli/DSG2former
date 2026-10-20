"""Fused attention must preserve the sparse equation, graph and training path."""
import unittest
import torch

from dsg2former.sparse_aligned import SparseDSG2former,prepare_attention_graph,edge_attention
from dsg2former.model import GraphAttention


class SparseFusedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def graph(self,n,device):
        i=torch.arange(n-1);q=torch.cat((i,i+1));k=torch.cat((i+1,i))
        edges=torch.stack((torch.zeros_like(q),q,k));rpe=torch.full_like(q,2)
        rpe[0]=25;rpe[1]=49;rpe[2]=0;rpe[3]=24
        valid=torch.ones(1,n,dtype=torch.bool)
        return prepare_attention_graph(edges,rpe,valid,torch.tensor([n-1])).to(device)

    def test_chunk_metadata_and_cpu_training_fallback(self):
        graph=self.graph(70,'cpu')
        self.assertGreater(graph.max_chunks,1)
        self.assertEqual(graph.row_chunk_offsets[-1].item(),len(graph.chunk_row))
        self.assertTrue(torch.equal(graph.chunk_end-graph.chunk_start,
                                    (graph.chunk_end-graph.chunk_start).clamp(1,32)))
        torch.manual_seed(14)
        a=SparseDSG2former(input_dim=12,dim=12,dropout=0.,inference_backend='reference')
        b=SparseDSG2former(input_dim=12,dim=12,dropout=0.,inference_backend='fused')
        b.load_state_dict(a.state_dict(),strict=True)
        x=torch.randn(1,70,12);drivers=torch.randn(1,70,2,3);role=torch.zeros(1,70,dtype=torch.long)
        ya=a.spatial_packed(x,drivers,graph,role)[0];yb=b.spatial_packed(x,drivers,graph,role)[0]
        torch.testing.assert_close(ya,yb,atol=0,rtol=0)
        ya.square().mean().backward();yb.square().mean().backward()
        for (_,pa),(_,pb) in zip(a.named_parameters(),b.named_parameters()):
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_widths_masks_long_rows_and_repeatability(self):
        from dsg2former.sparse_fused import fused_edge_attention
        for width in (5,8,16,32):
            torch.manual_seed(30+width)
            m=GraphAttention(width*3,3,50,.5,'rpe_mask',0.).cuda().eval()
            g=self.graph(70,'cuda');z=torch.randn(1,70,width*3,device='cuda')*.3
            with torch.inference_mode():
                ref=edge_attention(m,z,g.edge_index,g.edge_rpe,False,packed=g)
                out=fused_edge_attention(m,z,g)
                torch.testing.assert_close(out,ref,atol=2e-6,rtol=2e-5)
                for _ in range(3):self.assertTrue(torch.equal(out,fused_edge_attention(m,z,g)))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_model_and_capture(self):
        torch.manual_seed(9)
        model=SparseDSG2former(input_dim=12,dim=24,dropout=0.).cuda().eval()
        g=self.graph(70,'cuda');x=torch.randn(1,70,12,device='cuda')*.3
        drivers=torch.randn(1,70,2,3,device='cuda');role=torch.zeros(1,70,dtype=torch.long,device='cuda')
        with torch.inference_mode():
            expected=model.spatial_packed(x,drivers,g,role)[0]
            model.encoder.inference_backend='fused'
            actual=model.spatial_packed(x,drivers,g,role)[0]
            torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
            self.assertTrue(torch.equal(actual.argmax(-1),expected.argmax(-1)))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):model.spatial_packed(x,drivers,g,role)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):out=model.spatial_packed(x,drivers,g,role)[0]
            graph.replay();torch.testing.assert_close(out,expected,atol=1e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()

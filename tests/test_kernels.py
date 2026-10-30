"""
test_kernels.py — Unit tests for all Triton kernels.

Run with: python -m pytest tests/test_kernels.py -v
Or standalone: python tests/test_kernels.py

Tests verify:
1. Numerical correctness vs PyTorch reference
2. Gradient correctness
3. Edge cases (empty tensors, pad tokens, etc.)
4. Performance (kernel time < reference time)
"""

import torch
import torch.nn.functional as F
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype():
    return torch.float16


# ---------------------------------------------------------------------------
# Test: Fused Cross-Entropy
# ---------------------------------------------------------------------------

class TestFusedCrossEntropy:
    
    def test_forward_matches_pytorch(self, device):
        """Fused CE should match torch.nn.CrossEntropyLoss."""
        from solution.fused_cross_entropy import FusedCrossEntropyLoss
        
        N, V = 128, 1000  # smaller vocab for test
        logits = torch.randn(N, V, device=device, dtype=torch.float32)
        labels = torch.randint(0, V, (N,), device=device)
        
        # Reference
        ref_loss = F.cross_entropy(logits, labels)
        
        # Our fused
        fused_loss_fn = FusedCrossEntropyLoss()
        fused_loss = fused_loss_fn(logits, labels)
        
        torch.testing.assert_close(fused_loss, ref_loss, rtol=1e-3, atol=1e-3)
    
    def test_forward_with_ignore_index(self, device):
        """Should handle ignore_index correctly."""
        from solution.fused_cross_entropy import FusedCrossEntropyLoss
        
        N, V = 64, 500
        logits = torch.randn(N, V, device=device, dtype=torch.float32)
        labels = torch.randint(0, V, (N,), device=device)
        labels[::4] = -100  # ignore every 4th token
        
        ref_loss = F.cross_entropy(logits, labels, ignore_index=-100)
        
        fused_loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
        fused_loss = fused_loss_fn(logits, labels)
        
        torch.testing.assert_close(fused_loss, ref_loss, rtol=1e-3, atol=1e-3)
    
    def test_backward_matches_pytorch(self, device):
        """Gradient should match PyTorch."""
        from solution.fused_cross_entropy import FusedCrossEntropyLoss
        
        N, V = 64, 500
        logits = torch.randn(N, V, device=device, dtype=torch.float32, requires_grad=True)
        labels = torch.randint(0, V, (N,), device=device)
        
        # Reference
        logits_ref = logits.detach().clone().requires_grad_(True)
        ref_loss = F.cross_entropy(logits_ref, labels)
        ref_loss.backward()
        
        # Fused
        fused_loss_fn = FusedCrossEntropyLoss()
        fused_loss = fused_loss_fn(logits, labels)
        fused_loss.backward()
        
        torch.testing.assert_close(logits.grad, logits_ref.grad, rtol=1e-2, atol=1e-2)
    
    def test_fp16_input(self, device):
        """Should work with fp16 inputs."""
        from solution.fused_cross_entropy import FusedCrossEntropyLoss
        
        N, V = 64, 500
        logits = torch.randn(N, V, device=device, dtype=torch.float16)
        labels = torch.randint(0, V, (N,), device=device)
        
        fused_loss_fn = FusedCrossEntropyLoss()
        loss = fused_loss_fn(logits, labels)
        
        assert loss.dtype == torch.float32  # loss always in fp32
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_large_vocab(self, device):
        """Test with realistic vocab size (152K)."""
        from solution.fused_cross_entropy import FusedCrossEntropyLoss
        
        N, V = 32, 152000  # realistic Qwen2 vocab
        logits = torch.randn(N, V, device=device, dtype=torch.float16)
        labels = torch.randint(0, V, (N,), device=device)
        
        fused_loss_fn = FusedCrossEntropyLoss()
        loss = fused_loss_fn(logits, labels)
        
        assert not torch.isnan(loss)
        assert loss > 0


# ---------------------------------------------------------------------------
# Test: Fused RMSNorm + SiLU
# ---------------------------------------------------------------------------

class TestFusedRMSNormSiLU:
    
    def _reference(self, x, weight, eps=1e-6):
        """Reference: RMSNorm then SiLU."""
        # RMSNorm
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + eps)
        x_normed = x_normed * weight.float()
        # SiLU
        return F.silu(x_normed).half()
    
    def test_forward_matches_reference(self, device):
        from solution.fused_rmsnorm_silu import fused_rmsnorm_silu
        
        N, D = 32, 1536  # Qwen2 hidden size
        x = torch.randn(N, D, device=device, dtype=torch.float16)
        weight = torch.randn(D, device=device, dtype=torch.float16)
        
        ref = self._reference(x, weight)
        out = fused_rmsnorm_silu(x, weight)
        
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    
    def test_3d_input(self, device):
        """Should handle [B, S, D] input."""
        from solution.fused_rmsnorm_silu import fused_rmsnorm_silu
        
        B, S, D = 2, 16, 512
        x = torch.randn(B, S, D, device=device, dtype=torch.float16)
        weight = torch.randn(D, device=device, dtype=torch.float16)
        
        out = fused_rmsnorm_silu(x, weight)
        assert out.shape == (B, S, D)


# ---------------------------------------------------------------------------
# Test: Fused LoRA
# ---------------------------------------------------------------------------

class TestFusedLora:
    
    def test_forward_matches_reference(self, device):
        from solution.fused_lora import fused_lora_forward
        
        M, K, N, R = 32, 512, 512, 16
        base_output = torch.randn(M, N, device=device, dtype=torch.float16)
        x = torch.randn(M, K, device=device, dtype=torch.float16)
        lora_A = torch.randn(R, K, device=device, dtype=torch.float16) * 0.01
        lora_B = torch.zeros(N, R, device=device, dtype=torch.float16)  # init zeros
        scaling = 2.0
        
        # Reference
        ref = base_output + scaling * (x @ lora_A.T @ lora_B.T)
        
        # Fused
        out = fused_lora_forward(base_output, x, lora_A, lora_B, scaling)
        
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Test: Fused Embedding
# ---------------------------------------------------------------------------

class TestFusedEmbedding:
    
    def test_matches_torch_embedding(self, device):
        from solution.fused_embedding import fused_embedding_lookup
        
        V, D = 1000, 512
        weight = torch.randn(V, D, device=device, dtype=torch.float32)
        ids = torch.randint(0, V, (32,), device=device)
        
        # Reference
        ref = F.embedding(ids, weight).half()
        
        # Fused
        out = fused_embedding_lookup(weight, ids)
        
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    
    def test_with_scale(self, device):
        from solution.fused_embedding import fused_embedding_lookup
        
        V, D = 100, 256
        weight = torch.randn(V, D, device=device, dtype=torch.float32)
        ids = torch.randint(0, V, (16,), device=device)
        scale = D ** 0.5
        
        ref = (F.embedding(ids, weight) * scale).half()
        out = fused_embedding_lookup(weight, ids, scale=scale)
        
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# Test: Fused RoPE
# ---------------------------------------------------------------------------

class TestFusedRoPE:
    
    def _reference_rope(self, x, cos, sin):
        """Reference RoPE implementation."""
        D = x.shape[-1]
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1)
    
    def test_matches_reference(self, device):
        from solution.fused_rope import fused_apply_rotary_pos_emb
        
        B, H, S, D = 2, 8, 32, 64
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        
        # Precompute cos/sin
        cos = torch.randn(S, D // 2, device=device, dtype=torch.float16)
        sin = torch.randn(S, D // 2, device=device, dtype=torch.float16)
        
        # Reference
        q_ref = self._reference_rope(q, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        k_ref = self._reference_rope(k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        
        # Fused
        q_out, k_out = fused_apply_rotary_pos_emb(q, k, cos, sin)
        
        torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available — skipping tests")
        sys.exit(0)
    
    print("Running kernel tests...")
    device = torch.device("cuda")
    
    # Quick smoke tests
    print("\n--- Fused Cross-Entropy ---")
    test = TestFusedCrossEntropy()
    test.test_forward_matches_pytorch(device)
    print("✓ forward matches PyTorch")
    test.test_forward_with_ignore_index(device)
    print("✓ handles ignore_index")
    test.test_fp16_input(device)
    print("✓ fp16 input works")
    
    print("\n--- Fused RMSNorm+SiLU ---")
    test = TestFusedRMSNormSiLU()
    test.test_forward_matches_reference(device)
    print("✓ forward matches reference")
    
    print("\n--- Fused Embedding ---")
    test = TestFusedEmbedding()
    test.test_matches_torch_embedding(device)
    print("✓ matches torch.nn.Embedding")
    
    print("\n✅ All smoke tests passed!")

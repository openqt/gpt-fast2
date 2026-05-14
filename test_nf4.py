"""NF4 quantization validation tests."""
import sys
sys.path.insert(0, ".")

import torch


def test_nf4_levels():
    NF4_LEVELS = torch.tensor([
        -1.0000, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
         0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
         0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0000,
    ], dtype=torch.float32)
    assert NF4_LEVELS.shape == (16,)
    assert NF4_LEVELS[0] == -1.0
    assert NF4_LEVELS[-1] == 1.0
    assert NF4_LEVELS[7] == 0.0
    print(f"  NF4 levels: {NF4_LEVELS}")
    return NF4_LEVELS

def test_nf4_quantize_dequantize():
    from quantize import quantize_nf4, dequantize_nf4
    
    # Small test
    w = torch.randn(16, 256, dtype=torch.float32)
    qw, scales, _, _ = quantize_nf4(w, 128)
    dq = dequantize_nf4(qw, scales, 128)
    mae = (w - dq).abs().mean()
    print(f"  MAE (gs=128): {mae:.6f}")
    assert w.shape == dq.shape, f"Shape mismatch: {w.shape} vs {dq.shape}"
    assert mae < 0.2, f"NF4 error too high: {mae}"
    
    # Test groupsize variants
    for gs in [32, 64, 128]:
        qw, sc, _, _ = quantize_nf4(w, gs)
        dq = dequantize_nf4(qw, sc, gs)
        mae = (w - dq).abs().mean()
        print(f"  MAE (gs={gs}): {mae:.6f}")
        assert w.shape == dq.shape
    
    print("  quantize_nf4 / dequantize_nf4 PASSED")

def test_fp8_double_quant():
    from quantize import _fp8_quantize, _fp8_dequantize
    
    ts = torch.randn(4096, dtype=torch.bfloat16)
    qsc, absmax = _fp8_quantize(ts, blocksize=256)
    ts_dq = _fp8_dequantize(qsc, absmax, ts.shape, blocksize=256)
    mae = (ts.float() - ts_dq.float()).abs().mean()
    print(f"  FP8 MAE: {mae:.6f}")
    assert mae < 0.02, f"FP8 error too high: {mae}"
    print("  _fp8_quantize / _fp8_dequantize PASSED")

def test_weight_only_nf4_linear():
    from quantize import WeightOnlyNF4Linear
    
    layer = WeightOnlyNF4Linear(2048, 64, groupsize=128)
    x = torch.randn(1, 1, 2048, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == (1, 1, 64), f"Output shape unexpected: {out.shape}"
    print(f"  Forward: {x.shape} -> {out.shape}")
    print("  WeightOnlyNF4Linear PASSED")

def test_compression_ratio():
    from quantize import quantize_nf4
    
    w = torch.randn(4096, 4096, dtype=torch.float32)
    qw, sc, qsc, sab = quantize_nf4(w, 128)
    
    orig_bytes = w.numel() * 2  # FP16
    quant_bytes = (qw.numel() * 1 + sc.numel() * 2 + qsc.numel() * 1 + sab.numel() * 2)
    ratio = orig_bytes / quant_bytes
    print(f"  {orig_bytes//1024**2}MB -> {quant_bytes//1024**2}MB = {ratio:.2f}x compression")
    assert ratio > 3.0, f"Compression too low: {ratio:.2f}x"
    print("  Compression ratio PASSED")


if __name__ == "__main__":
    import time
    t0 = time.time()
    
    print("=== NF4 Level Table ===")
    test_nf4_levels()
    
    print("\n=== NF4 Quantize/Dequantize Roundtrip ===")
    test_nf4_quantize_dequantize()
    
    print("\n=== FP8 Double Quantization ===")
    test_fp8_double_quant()
    
    print("\n=== WeightOnlyNF4Linear Forward ===")
    test_weight_only_nf4_linear()
    
    print("\n=== Compression Ratio ===")
    test_compression_ratio()
    
    print(f"\nAll tests PASSED ({time.time()-t0:.2f}s)")

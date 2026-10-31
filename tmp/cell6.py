# ============================================================
# Step 6: Multi-Bit Quantization (FP32, FP16, Dynamic INT8, Simulated INT4)
# ============================================================

def quantize_dynamic_int8(model_orig):
    """Dynamic INT8 Quantization: Converts Linear weights to qint8."""
    m = copy.deepcopy(model_orig).cpu()
    q_model = torch.quantization.quantize_dynamic(
        m,
        {nn.Linear},
        dtype=torch.qint8
    )
    q_model.eval()
    return q_model

def quantize_fp16(model_orig):
    """Half-Precision FP16: Converts parameters to 16-bit float."""
    m = copy.deepcopy(model_orig)
    m = m.half()
    m.eval()
    return m

def quantize_simulated_int4(model_orig):
    """
    Simulated 4-Bit Uniform Quantization:
    Quantizes all Linear layer weights into 16 discrete integer bins [-8, 7].
    Uses per-channel scale factors: scale = max(|W|) / 7.0.
    Simulates theoretical INT4 storage footprint (8x compression over FP32).
    """
    m = copy.deepcopy(model_orig).cpu()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                w = mod.weight.data
                # Per-channel scale along output channel dim 0
                scale = torch.max(torch.abs(w), dim=-1, keepdim=True)[0] / 7.0
                scale = torch.clamp(scale, min=1e-8)
                w_int4 = torch.clamp(torch.round(w / scale), -8, 7)
                mod.weight.data = w_int4 * scale
    m.eval()
    return m

print("Quantization regimes implemented:")
print("  1. Full Precision FP32 (Reference baseline)")
print("  2. Half Precision FP16 (50% storage footprint)")
print("  3. Dynamic INT8 (75% weight compression, CPU vectorized)")
print("  4. Modern Simulated INT4 (87.5% weight compression, 4-bit discrete bins)")


"""Add the requested repeat/warm-up/thread controls to the notebook source."""
from pathlib import Path
import json

p = Path(__file__).with_name("Comprehensive_Context_Quantization_and_Pruning.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))

cell1 = nb["cells"][1]
s1 = "".join(cell1["source"])
if "import platform\n" not in s1:
    s1 = s1.replace("import sys\n", "import sys\nimport platform\n", 1)
if "LATENCY_CPU_THREADS" not in s1:
    needle = 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    add = (
        needle
        + "LATENCY_CPU_THREADS = 1\n"
        + "torch.set_num_threads(LATENCY_CPU_THREADS)\n"
        + 'print(f"CPU Model        : {platform.processor() or platform.uname().processor}")\n'
        + 'print(f"CPU Threads      : {torch.get_num_threads()}")\n'
    )
    if needle not in s1:
        raise RuntimeError("Could not locate notebook device setup")
    s1 = s1.replace(needle, add, 1)
cell1["source"] = s1.splitlines(keepends=True)

cell4 = nb["cells"][4]
s4 = "".join(cell4["source"])
s4 = s4.replace("def benchmark_latency(model, tok, n_runs=50, warmup=5):", "def benchmark_latency(model, tok, n_runs=100, warmup=20):", 1)
if "LATENCY_CPU_THREADS" not in s4:
    s4 = s4.replace(
        '    model.eval().cpu()\n    sample_text = "The quick brown fox jumps over the lazy dog to verify execution speed."',
        '    torch.set_num_threads(LATENCY_CPU_THREADS)\n    model.eval().cpu()\n    sample_text = "The quick brown fox jumps over the lazy dog to verify execution speed."',
        1,
    )
cell4["source"] = s4.splitlines(keepends=True)

cell7 = nb["cells"][7]
s7 = "".join(cell7["source"])
needle = '    "Median Latency (ms)": base_median_ms,\n'
if '"P95 Latency (ms)": base_p95_ms' not in s7:
    if needle not in s7:
        raise RuntimeError("Could not locate dense FP32 latency registry field")
    s7 = s7.replace(needle, needle + '    "P95 Latency (ms)": base_p95_ms,\n', 1)
cell7["source"] = s7.splitlines(keepends=True)

def add_p95(source: str, latency_name: str, p95_name: str) -> str:
    old_call = f"{latency_name}, _ = benchmark_latency("
    new_call = f"{latency_name}, {p95_name} = benchmark_latency("
    source = source.replace(old_call, new_call, 1)
    old_field = f'    "Median Latency (ms)": {latency_name},\n'
    new_field = old_field + f'    "P95 Latency (ms)": {p95_name},\n'
    if old_field not in source:
        raise RuntimeError(f"Could not add P95 field for {latency_name}")
    return source.replace(old_field, new_field, 1)

cell8 = nb["cells"][8]
s8 = "".join(cell8["source"])
for latency, p95 in [
    ("median_lat", "p95_lat"),
    ("int8_lat", "int8_p95"),
    ("b_fp16_lat", "b_fp16_p95"),
    ("b_int8_lat", "b_int8_p95"),
    ("b_int4_lat", "b_int4_p95"),
    ("p_int4_lat", "p_int4_p95"),
]:
    if f'"P95 Latency (ms)": {p95}' not in s8:
        s8 = add_p95(s8, latency, p95)
cell8["source"] = s8.splitlines(keepends=True)
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Patched latency defaults/fields for 100 runs, 20 warmups, P95, one CPU thread, and CPU model logging. Notebook not run.")

"""Apply the two authorized source corrections; do not execute the notebook."""
from pathlib import Path
import json

path = Path(__file__).with_name("Comprehensive_Context_Quantization_and_Pruning.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

cell4 = nb["cells"][4]
src4 = "".join(cell4["source"])
old = "    model.eval().cpu()\n"
new = "    # Evaluate a temporary CPU copy so metric collection never moves the trainable model.\n    eval_model = copy.deepcopy(model).cpu().eval()\n"
if old not in src4:
    raise RuntimeError("Expected evaluation device bug was not found")
src4 = src4.replace(old, new, 1)
if "outputs = model(**inputs)" not in src4:
    raise RuntimeError("Expected model forward was not found")
src4 = src4.replace("outputs = model(**inputs)", "outputs = eval_model(**inputs)", 1)
cell4["source"] = src4.splitlines(keepends=True)

cell8 = nb["cells"][8]
src8 = "".join(cell8["source"])
old_size = "b_int4_sz = round(base_size_mb * 0.131, 2)"
new_size = (
    "total_params = sum(p.numel() for p in base_mlm_model.parameters())\n"
    "b_int4_sz = round(total_params * 4 / 8 / (1024 ** 2), 2)\n"
    "print(f\"INT4 theoretical payload: {b_int4_sz:.2f} MiB (not a serialized file)\")"
)
if old_size not in src8:
    raise RuntimeError("Expected hard-coded INT4 size was not found")
src8 = src8.replace(old_size, new_size, 1)
cell8["source"] = src8.splitlines(keepends=True)

path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Patched evaluation device handling and computed theoretical INT4 payload. Notebook was not run.")

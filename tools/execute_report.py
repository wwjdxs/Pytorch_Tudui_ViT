"""执行报告并保存嵌入图表；避免依赖本机Jupyter内核注册。"""
import base64
import contextlib
import io
import os
from pathlib import Path
import nbformat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1] / "reports"
os.chdir(root)
path = root / "ViT实验对比报告.ipynb"
nb = nbformat.read(path, as_version=4)
namespace = {"__name__": "__main__"}
outputs = []
def display_capture(obj):
    data = {"text/markdown": obj._repr_markdown_()} if hasattr(obj, "_repr_markdown_") else {"text/plain": str(obj)}
    outputs.append(nbformat.v4.new_output("display_data", data=data))
def show_capture(*args, **kwargs):
    for number in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(number).savefig(buffer, format="png", bbox_inches="tight", dpi=120)
        outputs.append(nbformat.v4.new_output("display_data", data={"image/png": base64.b64encode(buffer.getvalue()).decode("ascii")}))
plt.show = show_capture
count = 0
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    outputs = []
    count += 1
    stream = io.StringIO()
    namespace["display"] = display_capture
    source = cell.source.replace("from IPython.display import display, Markdown", "from IPython.display import Markdown")
    with contextlib.redirect_stdout(stream):
        exec(compile(source, str(path), "exec"), namespace)
    cell.execution_count = count
    cell.outputs = ([nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue())] if stream.getvalue() else []) + outputs
nbformat.validate(nb)
nbformat.write(nb, path)
print("Executed notebook:", path)

"""Build Samuel individual notebooks with the same self-contained approved EDA.
No data/model execution occurs at build time; historical notebooks are preserved.
"""
import ast
import copy
import re
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks/samuel/final"
QUESTION = "Apakah persentase laporan berlabel `artifact_signal = 1` berbeda antara `signal_stream` kosong dan terisi, setelah mempertimbangkan waktu dan volume laporan?"

def build():
    eda = nbformat.read(ROOT / "notebooks/samuel/12_samuel_formal_eda_availability.ipynb", 4)
    # Namespace EDA temporaries so Regression's train remains Regression Train.
    assigned = {n.id for c in eda.cells if c.cell_type == "code" for n in ast.walk(ast.parse(c.source)) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    names = {name: "eda_" + name for name in assigned}
    class Rename(ast.NodeTransformer):
        def visit_arg(self, node):
            node.arg = names.get(node.arg, node.arg)
            return node
        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id=names.get(node.id, node.id), ctx=node.ctx), node)
    shared = []
    for i, original in enumerate(eda.cells):
        cell = copy.deepcopy(original)
        cell.id = f"samuel-approved-eda-{i:02d}"
        if cell.cell_type == "code":
            cell.source = ast.unparse(Rename().visit(ast.parse(cell.source)))
            cell.outputs = []
            cell.execution_count = None
        else:
            cell.source = re.sub(r"^## ([1-8])\. ", r"### EDA \1. ", cell.source, flags=re.M)
            if i == 0:
                cell.source = "## 2. EDA individual: teks kosong dan terisi\n\n**" + QUESTION + "**\n\nAnalisis berikut sama pada kedua notebook individual Samuel. Sumbernya adalah official Classification Train dan targetnya `artifact_signal`; label Regression tidak digabungkan. `text_load` adalah ukuran volume yang disediakan dataset, bukan jumlah token."
        shared.append(cell)
    paths = []
    for task, source_name, start, end in [
        ("Classification", "10_samuel_final_classification.ipynb", 6, 9),
        ("Regression", "09_samuel_final_regression.ipynb", 5, 6),
    ]:
        nb = nbformat.read(ROOT / "notebooks/samuel" / source_name, 4)
        nb.cells[start:end] = copy.deepcopy(shared)
        nb.cells[0].source += "\n\n**Identitas:** A06 â€” KasdudLovers | Samuel Indriano | 2406400524.\n\n**Ruang lingkup:** notebook individual akademis dengan EDA lengkap. Recipe modeling individual yang telah diverifikasi dipertahankan; notebook ini tidak mereplikasi ensemble champion kelompok terbaru. Bagian EDA memakai Classification Train pada kedua task. Refit penuh dan pembuatan submission tetap dimatikan."
        if task == "Regression":
            nb.cells[start].source += "\n\n**Konteks Regression:** ini tetap pertanyaan EDA individual Samuel tentang Classification. Hasilnya tidak membuktikan hubungan dengan `threat_rank` atau manfaat teks untuk MAE; evidence Regression berasal dari evaluasi model pada bagian berikut. Kedua official Train diperlukan untuk menjalankan notebook ini."
        for cell in nb.cells:
            if cell.cell_type == "markdown":
                cell.source = cell.source.replace("(12_samuel_formal_eda_availability.ipynb)", "(../12_samuel_formal_eda_availability.ipynb)").replace("(03_samuel_formal_eda.ipynb)", "(../03_samuel_formal_eda.ipynb)").replace("EDA A memberikan", "EDA availability memberikan")
            if cell.cell_type == "code":
                cell.source = cell.source.replace("artifacts/samuel_final/", "artifacts/samuel_eda_final_review/")
                cell.outputs = []
                cell.execution_count = None
        nb.metadata["samuel_scope"] = "Approved availability EDA integrated; historical individual modeling preserved; no new research or Kaggle action"
        nbformat.validate(nb)
        path = DEST / f"TP2_{task}_A06_KasdudLovers_SamuelIndriano_2406400524.ipynb"
        DEST.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, path)
        paths.append(path)
    return paths

if __name__ == "__main__":
    for path in build():
        print(path.relative_to(ROOT))

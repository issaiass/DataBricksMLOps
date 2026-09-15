import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gen():
    spec = importlib.util.spec_from_file_location(
        "generate_notebooks", ROOT / "scripts" / "generate_notebooks.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_notebooks_inline_python_not_main_caller():
    gen = _gen()
    repo = ROOT
    for group, stem, title in gen.STEPS:
        nb = gen.notebook_for(repo, group, stem, title)
        codes = [
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code"
        ]
        joined = "\n".join(codes)
        assert "BUNDLE_ROOT" not in joined
        assert "from src.n00_shared" in joined or "from src.n03_prod_cutover" in joined
        assert "Bundle root on `sys.path`" not in "".join(
            "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown"
        )
        assert "settings = load_settings()" in joined
        for block in codes:
            ast.parse(block)


def test_generated_notebook_files_match_generator():
    gen = _gen()
    repo = ROOT
    for group, stem, title in gen.STEPS:
        path = repo / "notebooks" / group / f"{stem}.ipynb"
        py = repo / "src" / group / f"{stem}.py"
        assert py.is_file(), f"missing python twin {py}"
        assert path.is_file(), f"missing notebook twin {path}"
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        expected = gen.notebook_for(repo, group, stem, title)
        assert len(on_disk["cells"]) == len(expected["cells"]), stem
        assert on_disk["cells"][0]["cell_type"] == "markdown"
        joined = "\n".join(
            "".join(c.get("source") or [])
            for c in on_disk["cells"]
            if c["cell_type"] == "code"
        )
        assert "if __name__" not in joined
        assert "settings = load_settings()" in joined


def test_every_step_python_has_run_and_main():
    gen = _gen()
    for group, stem, _title in gen.STEPS:
        src = (ROOT / "src" / group / f"{stem}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
        assert "run" in names, stem
        assert "main" in names, stem
        assert "if __name__ == \"__main__\":" in src or "if __name__ == '__main__':" in src

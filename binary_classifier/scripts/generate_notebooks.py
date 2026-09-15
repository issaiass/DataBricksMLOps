"""Generate notebooks that inline each step .py across cells (not a single main() caller)."""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

ADULT_MD = (
    "UCI Adult Census Income ([dataset](https://archive.ics.uci.edu/dataset/2/adult)): "
    "binary target `income_gt_50k` where **`>50K` = 1** and **`<=50K` = 0**. "
    "Fourteen census features; official split is `adult.data` (train) / `adult.test` (holdout).\n"
)

STEPS: list[tuple[str, str, str]] = [
    ("n01_dev_train", "n01_seed", "Folder 01 / file 01 — seed UCI Adult tables (also feature_refresh)"),
    ("n01_dev_train", "n02_data_checks", "Folder 01 / file 02 — Adult data checks (also feature_refresh)"),
    ("n01_dev_train", "n03_eda", "Folder 01 / file 03 — Adult EDA (dev only)"),
    ("n01_dev_train", "n04_features", "Folder 01 / file 04 — Adult Feature Store (also feature_refresh)"),
    ("n01_dev_train", "n05_train", "Folder 01 / file 05 — train XGBoost from Feature Store + register winner (dev only)"),
    ("n01_dev_train", "n06_evaluate", "Folder 01 file 06 / folder 02 file 02 — evaluate Adult holdout (mode from task env)"),
    ("n01_dev_train", "n07_compare", "Folder 01 / file 07 — Compare Adult tags only (after evaluate; parallel with XAI)"),
    ("n01_dev_train", "n08_xai", "Folder 01 / file 08 — Adult XAI (after evaluate; parallel with Compare; not a gate)"),
    ("n01_dev_train", "n09_promotion_ready", "Folder 01 / file 09 — promotion_ready (after Compare; sets @champion on go)"),
    ("n01_dev_train", "n10_smoke", "Folder 01 / file 10 — smoke Adult @dev (after Compare; not after promotion_ready)"),
    ("n02_prod_gate", "n01_copy_register", "Folder 02 / file 01 — copy_register Adult go version (then n01_dev_train/n06_evaluate)"),
    ("n03_prod_cutover", "n01_approval", "Folder 03 / file 01 — approval_check (read Approved only)"),
    ("n03_prod_cutover", "n02_deploy", "Folder 03 / file 02 — deploy Adult dest version"),
    ("n03_prod_cutover", "n03_canary", "Folder 03 / file 03 — canary_start / metrics_gate / to_100"),
    ("n03_prod_cutover", "n04_rollback", "Folder 03 / file 04 — rollback (failure path, not success OR)"),
    ("n04_ops", "n01_batch", "Folder 04 / file 01 — Adult prediction job (not train): load data+features, score batch or serving, write gold"),
    ("n04_ops", "n02_monitor", "Folder 04 / file 02 — scheduled Adult monitor"),
    ("n04_ops", "n03_cleanup", "Folder 04 / file 03 — manual cleanup of Adult example objects (always drops serving)"),
]

STOP_HELPER = '''\
STOP = False

def _stop(msg: str = "") -> None:
    global STOP
    STOP = True
    print(msg)
    try:
        dbutils.notebook.exit(msg or "ok")  # noqa: F821
    except Exception:
        pass
'''


def _md_cell(text: str) -> dict:
    src = text if text.endswith("\n") else text + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


def _code_cell_lines(text: str) -> dict:
    body = textwrap.dedent(text).strip("\n")
    lines = body.split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [ln + "\n" for ln in lines],
    }


def _segment(src: str, node: ast.AST) -> str:
    """Slice whole source lines so if/elif/else keeps one indent level."""
    lines = src.splitlines()
    start = node.lineno - 1
    end = node.end_lineno or node.lineno
    return "\n".join(lines[start:end])


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _rewrite_returns(text: str) -> str:
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        lineno = node.lineno - 1
        raw = lines[lineno]
        indent = raw[: len(raw) - len(raw.lstrip())]
        if node.value is None:
            lines[lineno] = f"{indent}_stop()"
        else:
            expr = ast.get_source_segment(text, node.value) or "''"
            lines[lineno] = f"{indent}_stop(str({expr}))"
    return "\n".join(lines)


def _group_statements(stmts: list[ast.stmt]) -> list[list[ast.stmt]]:
    """Chunk `run()` into notebook cells (~25 lines), keeping with/try/for intact."""
    budget = 25
    groups: list[list[ast.stmt]] = []
    buf: list[ast.stmt] = []

    def nlines(nodes: list[ast.stmt]) -> int:
        if not nodes:
            return 0
        return (nodes[-1].end_lineno or nodes[-1].lineno) - nodes[0].lineno + 1

    def flush() -> None:
        nonlocal buf
        if buf:
            groups.append(buf)
        buf = []

    for stmt in stmts:
        if _is_docstring(stmt):
            continue
        if isinstance(stmt, (ast.With, ast.AsyncWith, ast.Try, ast.For, ast.AsyncFor, ast.While)):
            flush()
            groups.append([stmt])
            continue
        buf.append(stmt)
        if nlines(buf) >= budget:
            flush()
    flush()
    return groups


def _has_return(fn: ast.FunctionDef) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(fn))


def notebook_for(repo: Path, group: str, stem: str, title: str) -> dict:
    py_path = repo / "src" / group / f"{stem}.py"
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cells: list[dict] = []

    doc = ast.get_docstring(tree) or ""
    cells.append(
        _md_cell(
            f"# {title}\n\n"
            f"{ADULT_MD}\n"
            f"Inlined replica of `{py_path.relative_to(repo).as_posix()}`. "
            f"Run cells **in order** (local or Jobs). Catalog/schema/model/mode come from task env.\n"
            + (f"\n{doc}\n" if doc else "")
        )
    )

    imports: list[ast.stmt] = []
    helpers: list[ast.FunctionDef] = []
    run_fn: ast.FunctionDef | None = None
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports.append(stmt)
        elif isinstance(stmt, ast.FunctionDef):
            if stmt.name == "run":
                run_fn = stmt
            elif stmt.name == "main":
                continue
            else:
                helpers.append(stmt)

    if run_fn is None:
        raise RuntimeError(f"{py_path} has no run()")
    use_stop = _has_return(run_fn)

    cells.append(_md_cell("## 1 — Imports"))
    import_bits = []
    for n in imports:
        if isinstance(n, ast.ImportFrom) and n.module == "__future__":
            continue
        import_bits.append(_segment(src, n))
    import_src = "\n".join(import_bits)
    if use_stop:
        import_src = STOP_HELPER.rstrip() + "\n\n" + import_src
    cells.append(_code_cell_lines(import_src))

    for i, fn in enumerate(helpers, start=2):
        cells.append(_md_cell(f"## {i} — `{fn.name}`"))
        cells.append(_code_cell_lines(_segment(src, fn)))

    step_n = 2 + len(helpers)
    cells.append(_md_cell(f"## {step_n} — `settings = load_settings()`"))
    cells.append(_code_cell_lines("settings = load_settings()"))
    step_n += 1

    groups = _group_statements(list(run_fn.body))
    for g_i, group_stmts in enumerate(groups, start=1):
        chunk = "\n".join(_segment(src, s) for s in group_stmts)
        chunk = textwrap.dedent(chunk)
        if use_stop:
            chunk = _rewrite_returns(chunk)
            inner = textwrap.indent(chunk.strip("\n"), "    ")
            chunk = f"if not STOP:\n{inner}\n"
        cells.append(_md_cell(f"## {step_n} — `run()` step {g_i}/{len(groups)}"))
        cells.append(_code_cell_lines(chunk))
        step_n += 1

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "notebooks"
    root.mkdir(parents=True, exist_ok=True)
    for group, stem, title in STEPS:
        folder = root / group
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}.ipynb"
        nb = notebook_for(repo, group, stem, title)
        path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        print(f"{path} cells={len(nb['cells'])}")


if __name__ == "__main__":
    main()

"""scripts/gate_required.py: which pull requests need the real-weight quality gate (AGENTS.md, Quality bar). The rule is the
model server's own import graph, so a module it starts loading is covered without a list to keep up to date."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("gate_required", ROOT / "scripts/gate_required.py")
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_the_server_and_what_it_loads_need_the_gate():
    loaded = gate.server_modules()
    assert {"d1a/serving/serve.py", "d1a/backends/torch.py", "d1a/backends/checkpoint.py", "d1a/backends/mlx.py",
            "d1a/core/api.py", "d1a/learning/feedback.py", "d1a/__init__.py"} <= loaded
    assert not {"d1a/training/train.py", "d1a/training/study.py", "d1a/eval/benchmark.py"} & loaded   # never imported by the server
    assert gate.reasons(["d1a/backends/torch.py", "uv.lock", "README.md", "scripts/quality_gate.py", "tests/test_unit.py"]) == [
        "d1a/backends/torch.py: the model server loads it", "uv.lock: a dependency change"]
    assert gate.reasons(["docs/UPSTREAM.md", "changes/1-x.md", "d1a/training/study.py"]) == []


def test_imports_are_read_everywhere_in_a_file(tmp_path):
    """Lazy imports inside functions and relative imports count; every package on the way counts (its __init__ runs)."""
    for f, text in {"d1a/__init__.py": "", "d1a/serving/__init__.py": "", "d1a/serving/serve.py": "from .media import x\ndef f():\n    from d1a.core import api\n",
                    "d1a/serving/media.py": "import d1a.backends.torch\n", "d1a/core/__init__.py": "", "d1a/core/api.py": "",
                    "d1a/backends/__init__.py": "", "d1a/backends/torch.py": "", "d1a/training/__init__.py": "", "d1a/training/train.py": ""}.items():
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True); (tmp_path / f).write_text(text, encoding="utf-8")
    assert gate.server_modules(tmp_path) == {"d1a/__init__.py", "d1a/serving/__init__.py", "d1a/serving/serve.py", "d1a/serving/media.py",
                                             "d1a/core/__init__.py", "d1a/core/api.py", "d1a/backends/__init__.py", "d1a/backends/torch.py"}

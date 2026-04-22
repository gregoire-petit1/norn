from pathlib import Path

from benchmarks.runner.sandbox import cleanup_sandbox, create_sandbox
from benchmarks.runner.models import TaskDef

import subprocess


def _make_seed(tmp_path: Path) -> Path:
    """Create a minimal seed directory for testing."""
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "src").mkdir()
    (seed / "src" / "solution.py").write_text("# placeholder\n")
    (seed / "tests").mkdir()
    (seed / "tests" / "test_it.py").write_text("def test_pass(): pass\n")
    return seed


def _make_task(tmp_path: Path, seed: Path) -> TaskDef:
    td = TaskDef(
        id="test-001",
        category="test",
        description="d",
        prompt="p",
        seed_dir="seed/",
        eval_command="echo ok",
    )
    # Set _source_path so absolute_seed_dir = tmp_path / "seed/"
    td._source_path = tmp_path / "task.yaml"
    return td


def test_create_sandbox_copies_files(tmp_path):
    seed = _make_seed(tmp_path)
    task = _make_task(tmp_path, seed)
    sandbox = create_sandbox(task)
    assert (sandbox / "src" / "solution.py").exists()
    assert (sandbox / "src" / "solution.py").read_text() == "# placeholder\n"
    assert (sandbox / "tests" / "test_it.py").exists()
    cleanup_sandbox(sandbox)
    assert not sandbox.exists()


def test_sandbox_has_git_init_and_seed_commit(tmp_path):
    seed = _make_seed(tmp_path)
    task = _make_task(tmp_path, seed)
    sandbox = create_sandbox(task)
    assert (sandbox / ".git").is_dir()
    result = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=sandbox,
        capture_output=True,
        text=True,
    )
    assert "seed" in result.stdout
    cleanup_sandbox(sandbox)


def test_sandbox_tmpdir_prefix(tmp_path):
    seed = _make_seed(tmp_path)
    task = _make_task(tmp_path, seed)
    sandbox = create_sandbox(task)
    assert "norn-bench-test-001" in sandbox.name
    cleanup_sandbox(sandbox)


def test_cleanup_nonexistent_path_no_error(tmp_path):
    cleanup_sandbox(tmp_path / "nonexistent")  # must not raise

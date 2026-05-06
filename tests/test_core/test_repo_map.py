"""Tests for repo map module (W3.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from norn.core.repo_map import RepoMap

if TYPE_CHECKING:
    from pathlib import Path


class TestPythonParsing:
    """Test Python AST-based symbol extraction."""

    def test_extracts_class(self, tmp_path: Path):
        """Extracts class definitions."""
        src = tmp_path / "module.py"
        src.write_text("class Foo:\n    pass\n")
        rm = RepoMap()
        result = rm._parse_python(src, src.read_text())
        assert len(result) == 1
        assert result[0].name == "Foo"
        assert result[0].kind == "class"
        assert result[0].line == 1

    def test_extracts_function(self, tmp_path: Path):
        """Extracts function definitions with signature."""
        src = tmp_path / "module.py"
        src.write_text("def hello(name: str) -> str:\n    return f'hi {name}'\n")
        rm = RepoMap()
        result = rm._parse_python(src, src.read_text())
        assert len(result) == 1
        assert result[0].name == "hello"
        assert result[0].kind == "function"
        assert "name: str" in result[0].signature

    def test_extracts_async_function(self, tmp_path: Path):
        """Extracts async function definitions."""
        src = tmp_path / "module.py"
        src.write_text("async def fetch(url: str) -> bytes:\n    pass\n")
        rm = RepoMap()
        result = rm._parse_python(src, src.read_text())
        assert result[0].kind == "async_function"
        assert "async def" in result[0].signature

    def test_extracts_methods_inside_class(self, tmp_path: Path):
        """Extracts methods as children of their class."""
        src = tmp_path / "module.py"
        src.write_text(
            "class Agent:\n"
            "    def __init__(self, name: str):\n"
            "        self.name = name\n"
            "    async def run(self, input: str) -> str:\n"
            "        return input\n"
        )
        rm = RepoMap()
        result = rm._parse_python(src, src.read_text())
        assert len(result) == 1
        cls = result[0]
        assert cls.name == "Agent"
        assert len(cls.children) == 2
        assert cls.children[0].name == "__init__"
        assert cls.children[1].name == "run"
        assert cls.children[1].kind == "async_method"

    def test_extracts_module_constants(self, tmp_path: Path):
        """Extracts UPPER_CASE module-level assignments."""
        src = tmp_path / "module.py"
        src.write_text("MAX_RETRIES = 3\n_INTERNAL = 'x'\nsome_var = 10\n")
        rm = RepoMap()
        result = rm._parse_python(src, src.read_text())
        # Only MAX_RETRIES (UPPER_CASE)
        assert len(result) == 1
        assert result[0].name == "MAX_RETRIES"
        assert result[0].kind == "constant"

    def test_handles_syntax_error_gracefully(self, tmp_path: Path):
        """Files with syntax errors return empty list."""
        src = tmp_path / "broken.py"
        src.write_text("def foo(\n")  # invalid syntax
        rm = RepoMap()
        result = rm._parse_python(src, src.read_text())
        assert result == []


class TestTypeScriptParsing:
    """Test TypeScript regex-based extraction."""

    def test_extracts_function(self, tmp_path: Path):
        """Extracts function declarations."""
        src = tmp_path / "module.ts"
        src.write_text("export function greet(name: string): string {\n  return name;\n}\n")
        rm = RepoMap()
        result = rm._parse_typescript(src, src.read_text())
        assert len(result) >= 1
        assert result[0].name == "greet"

    def test_extracts_class(self, tmp_path: Path):
        """Extracts class declarations."""
        src = tmp_path / "module.ts"
        src.write_text("export class UserService {\n  async getUser(id: string) {\n  }\n}\n")
        rm = RepoMap()
        result = rm._parse_typescript(src, src.read_text())
        assert any(s.name == "UserService" for s in result)

    def test_extracts_const(self, tmp_path: Path):
        """Extracts const declarations."""
        src = tmp_path / "module.ts"
        src.write_text("export const MAX_RETRIES = 3;\nconst helper = () => {};\n")
        rm = RepoMap()
        result = rm._parse_typescript(src, src.read_text())
        assert any(s.name == "MAX_RETRIES" for s in result)


class TestRepoMapScan:
    """Test full repo scanning and rendering."""

    def test_scan_python_project(self, tmp_path: Path):
        """Scans a Python project and produces a map."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text(
            "class App:\n    def run(self):\n        pass\n\ndef setup() -> None:\n    pass\n"
        )
        (src_dir / "utils.py").write_text("def helper(x: int) -> int:\n    return x + 1\n")
        rm = RepoMap()
        result = rm.scan(str(tmp_path))
        assert "App" in result
        assert "run" in result
        assert "setup" in result
        assert "helper" in result
        assert "## Repo Map" in result

    def test_respects_max_chars(self, tmp_path: Path):
        """Output is capped at max_chars."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        # Create many files to exceed budget
        for i in range(50):
            (src_dir / f"mod_{i}.py").write_text(
                f"class Class{i}:\n    def method_a(self):\n        pass\n"
                f"    def method_b(self):\n        pass\n"
            )
        rm = RepoMap(max_chars=500)
        result = rm.scan(str(tmp_path))
        assert len(result) <= 550  # some margin for truncation message

    def test_excludes_venv(self, tmp_path: Path):
        """Files in .venv are excluded."""
        (tmp_path / "main.py").write_text("def app(): pass\n")
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        (venv / "something.py").write_text("def hidden(): pass\n")
        rm = RepoMap()
        result = rm.scan(str(tmp_path))
        assert "app" in result
        assert "hidden" not in result

    def test_excludes_pycache(self, tmp_path: Path):
        """__pycache__ directories are excluded."""
        (tmp_path / "main.py").write_text("def app(): pass\n")
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "main.cpython-311.pyc").write_bytes(b"\x00")
        rm = RepoMap()
        result = rm.scan(str(tmp_path))
        assert "app" in result

    def test_empty_project_returns_minimal(self, tmp_path: Path):
        """Empty project returns a minimal map."""
        rm = RepoMap()
        result = rm.scan(str(tmp_path))
        assert "Repo Map" in result or result == ""

    def test_caching_uses_mtime(self, tmp_path: Path):
        """Second scan uses cache for unchanged files."""
        src = tmp_path / "main.py"
        src.write_text("def foo(): pass\n")
        rm = RepoMap()
        rm.scan(str(tmp_path))
        assert len(rm._cache) == 1
        # Scan again — should use cache
        rm.scan(str(tmp_path))
        assert len(rm._cache) == 1

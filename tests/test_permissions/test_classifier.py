"""Tests for the risk classifier."""

import pytest

from norn.permissions.classifier import RiskClassifier


@pytest.fixture
def classifier():
    return RiskClassifier()


class TestBashPatterns:
    """Test destructive bash command detection."""

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf --no-preserve-root /",
            "sudo rm -rf /var",
            "DROP TABLE users;",
            "drop table users",
            "DELETE FROM users WHERE 1=1;",
            "git push --force origin main",
            "git push -f origin master",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
            ":(){ :|:& };:",
            "> /dev/sda",
            "chmod -R 777 /",
            "chown -R nobody /",
        ],
    )
    def test_destructive_commands_flagged(self, classifier, command):
        result = classifier.classify_bash(command)
        assert result.is_destructive is True, f"Expected destructive: {command}"

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat file.txt",
            "echo hello",
            "git status",
            "git push origin feature-branch",
            "python script.py",
            "pytest tests/",
            "rm file.txt",  # single file rm is not mass-destructive
            "grep -r pattern .",
        ],
    )
    def test_safe_commands_not_flagged(self, classifier, command):
        result = classifier.classify_bash(command)
        assert result.is_destructive is False, f"Expected safe: {command}"


class TestProtectedPaths:
    """Test protected path detection."""

    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            "/home/user/.env",
            ".ssh/id_rsa",
            "~/.ssh/authorized_keys",
            ".gitconfig",
            ".bashrc",
            ".zshrc",
            "secrets/credentials.json",
            "server.pem",
            "private.key",
            "/etc/passwd",
            "/etc/shadow",
        ],
    )
    def test_protected_paths_detected(self, classifier, path):
        assert classifier.is_protected_path(path) is True, f"Expected protected: {path}"

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "tests/test_foo.py",
            "README.md",
            "configs/default.yaml",
            "data/input.csv",
        ],
    )
    def test_normal_paths_not_protected(self, classifier, path):
        assert classifier.is_protected_path(path) is False, f"Expected normal: {path}"


class TestPathTraversal:
    """Test path traversal detection."""

    @pytest.mark.parametrize(
        "path",
        [
            "../../../etc/passwd",
            "foo/../../bar",
            "%2e%2e/%2e%2e/etc/passwd",
            "..\\..\\windows\\system32",
            "foo/..%5c..%5cbar",
        ],
    )
    def test_traversal_detected(self, classifier, path):
        assert classifier.has_path_traversal(path) is True, f"Expected traversal: {path}"

    @pytest.mark.parametrize(
        "path",
        [
            "src/core/agent.py",
            "/absolute/path/file.py",
            "./relative/file.py",
            "file.txt",
        ],
    )
    def test_normal_paths_no_traversal(self, classifier, path):
        assert classifier.has_path_traversal(path) is False, f"Expected no traversal: {path}"


class TestEscalation:
    """Test risk escalation logic."""

    def test_bash_destructive_escalates(self, classifier):
        escalated = classifier.escalate(
            tool_name="bash",
            base_risk="high",
            arguments={"command": "rm -rf /"},
        )
        assert escalated.risk == "high"
        assert escalated.is_destructive is True
        assert "destructive" in escalated.reason.lower()

    def test_file_write_to_protected_escalates(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_write",
            base_risk="medium",
            arguments={"path": ".env", "content": "SECRET=abc"},
        )
        assert escalated.risk == "high"
        assert escalated.is_protected is True

    def test_file_write_normal_no_escalation(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_write",
            base_risk="medium",
            arguments={"path": "src/main.py", "content": "print('hi')"},
        )
        assert escalated.risk == "medium"
        assert escalated.is_destructive is False
        assert escalated.is_protected is False

    def test_file_edit_with_traversal_escalates(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_edit",
            base_risk="medium",
            arguments={"path": "../../../etc/passwd", "old_string": "x", "new_string": "y"},
        )
        assert escalated.risk == "high"
        assert escalated.has_traversal is True
        assert escalated.reason is not None

    def test_bash_destructive_escalates_from_low_risk(self, classifier):
        escalated = classifier.escalate(
            tool_name="bash",
            base_risk="low",
            arguments={"command": "rm -rf /"},
        )
        assert escalated.risk == "high"
        assert escalated.is_destructive is True
        assert "destructive" in escalated.reason.lower()

    def test_read_tool_no_escalation(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_read",
            base_risk="low",
            arguments={"path": ".env"},
        )
        # Read-only tools don't get escalated even for protected paths
        assert escalated.risk == "low"

import pytest
from src.solution import UserManager


@pytest.fixture
def mgr():
    m = UserManager()
    m.add_user("alice", "alice@example.com", 30)
    return m


def test_add_user(mgr):
    user = mgr.add_user("bob", "bob@test.com", 25)
    assert user["username"] == "bob"


def test_add_duplicate(mgr):
    with pytest.raises(ValueError, match="already exists"):
        mgr.add_user("alice", "other@test.com", 20)


def test_invalid_username():
    mgr = UserManager()
    with pytest.raises(ValueError, match="at least 3"):
        mgr.add_user("ab", "x@y.z", 20)


def test_invalid_email():
    mgr = UserManager()
    with pytest.raises(ValueError, match="Invalid email"):
        mgr.add_user("test", "nope", 20)


def test_invalid_age():
    mgr = UserManager()
    with pytest.raises(ValueError, match="Invalid age"):
        mgr.add_user("test", "a@b.c", -1)


def test_get_user(mgr):
    assert mgr.get_user("alice")["email"] == "alice@example.com"


def test_get_missing():
    mgr = UserManager()
    assert mgr.get_user("nobody") is None


def test_delete(mgr):
    assert mgr.delete_user("alice") is True
    assert mgr.get_user("alice") is None


def test_delete_missing(mgr):
    assert mgr.delete_user("nobody") is False


def test_list_users(mgr):
    mgr.add_user("bob", "bob@test.com", 25)
    users = mgr.list_users()
    assert len(users) == 2


def test_format(mgr):
    assert mgr.format_user("alice") == "alice <alice@example.com> (age: 30)"


def test_format_missing(mgr):
    assert mgr.format_user("nobody") == "User not found"


def test_search_by_email(mgr):
    assert mgr.search_by_email("alice@example.com")["username"] == "alice"


def test_search_missing_email(mgr):
    assert mgr.search_by_email("nope@test.com") is None


def test_count(mgr):
    assert mgr.count() == 1
    mgr.add_user("bob", "bob@test.com", 25)
    assert mgr.count() == 2

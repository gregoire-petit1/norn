class UserManager:
    """God class: handles validation, storage, and formatting."""

    def __init__(self):
        self._users = {}

    def add_user(self, username: str, email: str, age: int) -> dict:
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if "@" not in email or "." not in email:
            raise ValueError("Invalid email format")
        if age < 0 or age > 150:
            raise ValueError("Invalid age")
        if username in self._users:
            raise ValueError("Username already exists")
        user = {"username": username, "email": email, "age": age}
        self._users[username] = user
        return user

    def get_user(self, username: str) -> dict | None:
        return self._users.get(username)

    def delete_user(self, username: str) -> bool:
        if username in self._users:
            del self._users[username]
            return True
        return False

    def list_users(self) -> list[dict]:
        return list(self._users.values())

    def format_user(self, username: str) -> str:
        user = self._users.get(username)
        if user is None:
            return "User not found"
        return f"{user['username']} <{user['email']}> (age: {user['age']})"

    def search_by_email(self, email: str) -> dict | None:
        for user in self._users.values():
            if user["email"] == email:
                return user
        return None

    def count(self) -> int:
        return len(self._users)

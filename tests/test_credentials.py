"""Tests for Credentials management."""

import logging

import keyring
import keyring.errors
import pytest
from keyring.backend import KeyringBackend

from qs.config import API_KEY_ACCOUNT, API_SERVICE_NAME, GOOGLE_API_KEY_MIN_LENGTH
from qs.credentials.credentials import Credentials
from qs.credentials.credentials import logger as cred_logger
from qs.errors.exceptions import ConfigurationError, MissingAPIKeyError


class MemoryKeyring(KeyringBackend):
    """In-memory keyring backend for testing."""

    priority = 10

    def __init__(self):
        self._passwords: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._passwords[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._passwords.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        key = (service, username)
        if key not in self._passwords:
            raise keyring.errors.PasswordDeleteError("Password not found")
        del self._passwords[key]


@pytest.fixture(autouse=True)
def memory_keyring(monkeypatch):
    """Use memory keyring for all test isolation."""
    mem_backend = MemoryKeyring()
    monkeypatch.setattr(keyring, "get_keyring", lambda: mem_backend)
    monkeypatch.setattr(keyring, "set_password", mem_backend.set_password)
    monkeypatch.setattr(keyring, "get_password", mem_backend.get_password)
    monkeypatch.setattr(keyring, "delete_password", mem_backend.delete_password)
    monkeypatch.setattr(keyring, "get_credential", mem_backend.get_credential)
    return mem_backend


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure environment variables do not leak into tests."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


# --- Blackboard Credentials Tests ---


def test_set_and_get_credentials_bb():
    Credentials.set_credentials_bb("student1", "secret123")
    user, pwd = Credentials.get_credentials_bb("student1")
    assert user == "student1"
    assert pwd == "secret123"


def test_get_credentials_bb_direct_password():
    user, pwd = Credentials.get_credentials_bb("student1", "direct-pass")
    assert user == "student1"
    assert pwd == "direct-pass"


def test_get_credentials_bb_direct_password_invalid():
    with pytest.raises(ConfigurationError, match="must be a string"):
        Credentials.get_credentials_bb("student1", 12345)  # type: ignore
    with pytest.raises(ConfigurationError, match="must not be empty"):
        Credentials.get_credentials_bb("student1", "")


def test_set_credentials_strips_whitespace():
    Credentials.set_credentials_bb("  student2  ", "secret")
    user, pwd = Credentials.get_credentials_bb("student2")
    assert user == "student2"
    assert pwd == "secret"
    # Also retrieval with whitespace works
    user2, pwd2 = Credentials.get_credentials_bb(" student2 ")
    assert user2 == "student2"
    assert pwd2 == "secret"


def test_set_credentials_duplicate_fails():
    Credentials.set_credentials_bb("user1", "pass1")
    with pytest.raises(ConfigurationError, match="already exist"):
        Credentials.set_credentials_bb("user1", "pass2")


def test_set_credentials_invalid_types():
    with pytest.raises(ConfigurationError, match="must be strings"):
        Credentials.set_credentials_bb(123, "pass")  # type: ignore
    with pytest.raises(ConfigurationError, match="must be strings"):
        Credentials.set_credentials_bb("user", None)  # type: ignore


def test_set_credentials_empty():
    with pytest.raises(ConfigurationError, match="must not be empty"):
        Credentials.set_credentials_bb("   ", "pass")
    with pytest.raises(ConfigurationError, match="must not be empty"):
        Credentials.set_credentials_bb("user", "")


def test_update_credentials_bb():
    Credentials.set_credentials_bb("user1", "pass1")
    Credentials.update_credentials_bb("user1", "pass2")
    _, pwd = Credentials.get_credentials_bb("user1")
    assert pwd == "pass2"


def test_update_credentials_not_found():
    with pytest.raises(ConfigurationError, match="No credentials found"):
        Credentials.update_credentials_bb("missing_user", "pass")


def test_delete_credentials_bb():
    Credentials.set_credentials_bb("user1", "pass1")
    Credentials.delete_credentials_bb("user1")
    with pytest.raises(ConfigurationError, match="No credentials found"):
        Credentials.get_credentials_bb("user1")


def test_delete_credentials_not_found():
    with pytest.raises(ConfigurationError, match="No credentials found"):
        Credentials.delete_credentials_bb("missing_user")


def test_show_credentials_bb():
    Credentials.set_credentials_bb("student1", "secret123")
    masked_user, masked_pwd = Credentials.show_credentials_bb("student1")
    assert masked_user == "student1"
    assert masked_pwd == "*********"


def test_show_credentials_not_found():
    with pytest.raises(ConfigurationError, match="No credentials found"):
        Credentials.show_credentials_bb("missing_user")


# --- Gemini API Key Tests ---


def test_get_api_key_from_env(monkeypatch):
    valid_key = "AIzaSy" + "A" * 33
    monkeypatch.setenv("GEMINI_API_KEY", valid_key)
    assert Credentials.get_api_key() == valid_key


def test_get_api_key_from_google_api_key_env(monkeypatch):
    valid_key = "AIzaSy" + "B" * 33
    monkeypatch.setenv("GOOGLE_API_KEY", valid_key)
    assert Credentials.get_api_key() == valid_key


def test_get_api_key_direct():
    valid_key = "AIzaSy" + "C" * 33
    assert Credentials.get_api_key(api_key=valid_key) == valid_key


def test_get_api_key_direct_invalid():
    with pytest.raises(ConfigurationError, match="must be a string"):
        Credentials.get_api_key(api_key=12345)  # type: ignore
    with pytest.raises(ConfigurationError, match="must not be empty"):
        Credentials.get_api_key(api_key="   ")
    with pytest.raises(
        ConfigurationError, match=f"at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
    ):
        Credentials.get_api_key(api_key="too-short")


def test_get_api_key_from_keyring(memory_keyring):
    valid_key = "AIzaSy" + "D" * 33
    memory_keyring.set_password(API_SERVICE_NAME, API_KEY_ACCOUNT, valid_key)
    assert Credentials.get_api_key() == valid_key


def test_get_api_key_env_too_short(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "too-short")
    with pytest.raises(
        ConfigurationError, match=f"at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
    ):
        Credentials.get_api_key()


def test_get_api_key_keyring_too_short(memory_keyring):
    memory_keyring.set_password(API_SERVICE_NAME, API_KEY_ACCOUNT, "too-short")
    with pytest.raises(
        ConfigurationError, match=f"at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
    ):
        Credentials.get_api_key()


def test_get_api_key_missing_raises():
    with pytest.raises(MissingAPIKeyError):
        Credentials.get_api_key()


def test_set_and_delete_api_key():
    valid_key = "AIzaSy" + "E" * 33
    Credentials.set_api_key(valid_key)
    assert Credentials.get_api_key() == valid_key
    Credentials.delete_api_key()
    with pytest.raises(MissingAPIKeyError):
        Credentials.get_api_key()


def test_set_api_key_invalid():
    with pytest.raises(ConfigurationError, match="must be a string"):
        Credentials.set_api_key(12345)  # type: ignore
    with pytest.raises(ConfigurationError, match="must not be empty"):
        Credentials.set_api_key("   ")
    with pytest.raises(
        ConfigurationError, match=f"at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
    ):
        Credentials.set_api_key("too-short")


def test_delete_api_key_not_found():
    with pytest.raises(ConfigurationError, match="No API key found"):
        Credentials.delete_api_key()


def test_show_api_key():
    valid_key = "AIzaSy" + "F" * 33
    Credentials.set_api_key(valid_key)
    assert Credentials.show_api_key() == "*" * 39
    assert Credentials.show_creds() == "*" * 39


def test_show_api_key_direct():
    valid_key = "AIzaSy" + "G" * 33
    assert Credentials.show_api_key(valid_key) == "*" * 39
    assert Credentials.show_creds(valid_key) == "*" * 39


def test_show_api_key_missing_raises():
    with pytest.raises(MissingAPIKeyError):
        Credentials.show_api_key()
    with pytest.raises(MissingAPIKeyError):
        Credentials.show_creds()


# --- Logging Integration Tests ---


@pytest.fixture
def log_capture():
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    cred_logger.addHandler(handler)
    original_level = cred_logger.level
    cred_logger.setLevel(logging.DEBUG)
    yield records
    cred_logger.removeHandler(handler)
    cred_logger.setLevel(original_level)


def test_logging_bb_credentials_lifecycle(log_capture):
    Credentials.set_credentials_bb("student_log", "super_secret_pwd")
    assert any(
        r.levelno == logging.INFO
        and "Saved Blackboard credentials for 'student_log'" in r.getMessage()
        for r in log_capture
    )
    # Ensure plaintext password is not in any log message
    assert not any("super_secret_pwd" in r.getMessage() for r in log_capture)

    log_capture.clear()
    Credentials.get_credentials_bb("student_log")
    assert any(
        r.levelno == logging.DEBUG
        and "Retrieved Blackboard credentials for 'student_log' from keyring" in r.getMessage()
        for r in log_capture
    )

    log_capture.clear()
    Credentials.update_credentials_bb("student_log", "new_secret_pwd")
    assert any(
        r.levelno == logging.INFO
        and "Updated Blackboard credentials for 'student_log'" in r.getMessage()
        for r in log_capture
    )
    assert not any("new_secret_pwd" in r.getMessage() for r in log_capture)

    log_capture.clear()
    Credentials.delete_credentials_bb("student_log")
    assert any(
        r.levelno == logging.INFO
        and "Deleted Blackboard credentials for 'student_log'" in r.getMessage()
        for r in log_capture
    )


def test_logging_api_key_lifecycle(log_capture):
    valid_key = "AIzaSy" + "Z" * 33
    Credentials.set_api_key(valid_key)
    assert any(
        r.levelno == logging.INFO and "Saved Gemini API key to keyring" in r.getMessage()
        for r in log_capture
    )
    # Ensure raw API key is never logged
    assert not any(valid_key in r.getMessage() for r in log_capture)

    log_capture.clear()
    Credentials.get_api_key()
    assert any(
        r.levelno == logging.DEBUG and "Loaded Gemini API key from keyring" in r.getMessage()
        for r in log_capture
    )
    assert not any(valid_key in r.getMessage() for r in log_capture)

    log_capture.clear()
    Credentials.delete_api_key()
    assert any(
        r.levelno == logging.INFO and "Deleted Gemini API key from keyring" in r.getMessage()
        for r in log_capture
    )


def test_logging_missing_api_key_warning(log_capture):
    with pytest.raises(MissingAPIKeyError):
        Credentials.get_api_key()
    assert any(
        r.levelno == logging.WARNING
        and "Gemini API key not found in environment or keyring" in r.getMessage()
        for r in log_capture
    )

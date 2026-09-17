from __future__ import annotations

import contextlib
import os

import keyring
import keyring.credentials
from keyring.errors import KeyringError, PasswordDeleteError, PasswordSetError

from qs.config import (
    API_KEY_ACCOUNT,
    API_SERVICE_NAME,
    BB_SERVICE_NAME,
    GOOGLE_API_KEY_MIN_LENGTH,
)
from qs.errors.exceptions import ConfigurationError, MissingAPIKeyError
from qs.logger import get_logger

logger = get_logger(__name__)


class Credentials:
    """Manages Blackboard credentials and Gemini API keys via keyring and environment."""

    # Blackboard Credentials

    @classmethod
    def set_credentials_bb(cls, username: str, password: str) -> None:
        if not isinstance(username, str) or not isinstance(password, str):
            raise ConfigurationError("username and password must be strings")

        sanitized_username = username.strip()
        if not sanitized_username or not password:
            raise ConfigurationError("username and password must not be empty")

        try:
            if keyring.get_credential(BB_SERVICE_NAME, sanitized_username) is not None:
                raise ConfigurationError(
                    f"Credentials for {sanitized_username} already exist, use update_credentials_bb to modify"
                )
            keyring.set_password(BB_SERVICE_NAME, sanitized_username, password)
            logger.info("Saved Blackboard credentials for '%s'", sanitized_username)
        except PasswordSetError as e:
            logger.error("Failed to set password: %s", e)
            raise ConfigurationError(f"Failed to set password: {e}") from e
        except KeyringError as e:
            logger.error("Keyring error: %s", e)
            raise ConfigurationError(f"Keyring error: {e}") from e

    @classmethod
    def update_credentials_bb(cls, username: str, password: str) -> None:
        if not isinstance(username, str) or not isinstance(password, str):
            raise ConfigurationError("username and password must be strings")

        sanitized_username = username.strip()
        if not sanitized_username or not password:
            raise ConfigurationError("username and password must not be empty")

        try:
            if keyring.get_credential(BB_SERVICE_NAME, sanitized_username) is None:
                raise ConfigurationError(
                    f"No credentials found for {sanitized_username}, use set_credentials_bb to create."
                )
            keyring.set_password(BB_SERVICE_NAME, sanitized_username, password)
            logger.info("Updated Blackboard credentials for '%s'", sanitized_username)
        except PasswordSetError as e:
            logger.error("Failed to update password: %s", e)
            raise ConfigurationError(f"Failed to update password: {e}") from e
        except KeyringError as e:
            logger.error("Keyring error: %s", e)
            raise ConfigurationError(f"Keyring error: {e}") from e

    @classmethod
    def delete_credentials_bb(cls, username: str) -> None:
        if not isinstance(username, str):
            raise ConfigurationError("username must be a string")

        sanitized_username = username.strip()
        if not sanitized_username:
            raise ConfigurationError("username must not be empty")

        try:
            if keyring.get_credential(BB_SERVICE_NAME, sanitized_username) is None:
                raise ConfigurationError(
                    f"No credentials found for {sanitized_username}, use set_credentials_bb to create."
                )
            keyring.delete_password(BB_SERVICE_NAME, sanitized_username)
            logger.info("Deleted Blackboard credentials for '%s'", sanitized_username)
        except PasswordDeleteError as e:
            logger.error("Failed to delete password: %s", e)
            raise ConfigurationError(f"Failed to delete password: {e}") from e
        except KeyringError as e:
            logger.error("Keyring error: %s", e)
            raise ConfigurationError(f"Keyring error: {e}") from e

    @classmethod
    def get_credentials_bb(cls, username: str, password: str | None = None) -> tuple[str, str]:
        if not isinstance(username, str):
            raise ConfigurationError("username must be a string")

        sanitized_username = username.strip()
        if not sanitized_username:
            raise ConfigurationError("username must not be empty")

        if password is not None:
            if not isinstance(password, str):
                raise ConfigurationError("password must be a string")
            if not password:
                raise ConfigurationError("password must not be empty")
            logger.debug("Using directly provided Blackboard password for '%s'", sanitized_username)
            return (sanitized_username, password)

        try:
            creds = keyring.get_credential(BB_SERVICE_NAME, sanitized_username)
        except KeyringError as e:
            logger.error("Failed to access keyring: %s", e)
            raise ConfigurationError(f"Failed to access keyring: {e}") from e

        if creds is None or creds.password is None:
            raise ConfigurationError(f"No credentials found for {sanitized_username}")

        logger.debug("Retrieved Blackboard credentials for '%s' from keyring", sanitized_username)
        return (creds.username, creds.password)

    @classmethod
    def show_credentials_bb(cls, username: str) -> tuple[str, str]:
        if not isinstance(username, str):
            raise ConfigurationError("username must be a string")

        sanitized_username = username.strip()
        if not sanitized_username:
            raise ConfigurationError("username must not be empty")

        try:
            creds = keyring.get_credential(BB_SERVICE_NAME, sanitized_username)
        except KeyringError as e:
            logger.error("Keyring error: %s", e)
            raise ConfigurationError(f"Keyring error: {e}") from e

        if creds is None:
            raise ConfigurationError(f"No credentials found for {sanitized_username}")

        masked = cls.mask(creds)
        if masked is None:
            raise ConfigurationError(f"No credentials found for {sanitized_username}")
        logger.debug("Retrieved masked Blackboard credentials for '%s'", sanitized_username)
        return masked

    @classmethod
    def mask(cls, creds: keyring.credentials.Credential | None) -> tuple[str, str] | None:
        if creds is None:
            return None
        pwd = creds.password or ""
        return (creds.username, "*" * len(pwd))

    @classmethod
    def get_default_bb_username(cls) -> str | None:
        """Retrieve default Blackboard username from env or keyring."""
        env_user = os.environ.get("BB_USERNAME")
        if env_user and env_user.strip():
            return env_user.strip()
        try:
            stored = keyring.get_password(BB_SERVICE_NAME, "__default_user__")
            if stored and stored.strip():
                return stored.strip()
        except KeyringError:
            pass
        return None

    @classmethod
    def set_default_bb_username(cls, username: str) -> None:
        """Store default Blackboard username."""
        if not isinstance(username, str) or not username.strip():
            raise ConfigurationError("username must be a non-empty string")
        try:
            keyring.set_password(BB_SERVICE_NAME, "__default_user__", username.strip())
        except KeyringError as e:
            logger.error("Failed to store default BB username: %s", e)
            raise ConfigurationError(f"Failed to store default BB username: {e}") from e

    @classmethod
    def delete_default_bb_username(cls) -> None:
        """Remove default Blackboard username."""
        with contextlib.suppress(Exception):
            keyring.delete_password(BB_SERVICE_NAME, "__default_user__")

    # Gemini API Key

    @classmethod
    def get_api_key(cls, api_key: str | None = None) -> str:
        if api_key is not None:
            if not isinstance(api_key, str):
                raise ConfigurationError("api_key must be a string")
            sanitized_key = api_key.strip()
            if not sanitized_key:
                raise ConfigurationError("api_key must not be empty")
            if len(sanitized_key) < GOOGLE_API_KEY_MIN_LENGTH:
                raise ConfigurationError(
                    f"api_key must be at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
                )
            logger.debug("Using directly provided Gemini API key")
            return sanitized_key

        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key and env_key.strip():
            sanitized_env_key = env_key.strip()
            if len(sanitized_env_key) < GOOGLE_API_KEY_MIN_LENGTH:
                raise ConfigurationError(
                    f"API key in environment must be at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
                )
            logger.debug("Loaded Gemini API key from environment")
            return sanitized_env_key

        try:
            stored_key = keyring.get_password(API_SERVICE_NAME, API_KEY_ACCOUNT)
            if stored_key and stored_key.strip():
                sanitized_stored_key = stored_key.strip()
                if len(sanitized_stored_key) < GOOGLE_API_KEY_MIN_LENGTH:
                    raise ConfigurationError(
                        f"API key in keyring must be at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
                    )
                logger.debug("Loaded Gemini API key from keyring")
                return sanitized_stored_key
        except KeyringError as e:
            logger.error("Failed to access keyring for Gemini API key: %s", e)
            raise ConfigurationError(f"Failed to access keyring: {e}") from e

        logger.warning("Gemini API key not found in environment or keyring")
        raise MissingAPIKeyError(
            "Gemini API key not found in environment or keyring. "
            "Set GEMINI_API_KEY environment variable or configure via CLI."
        )

    @classmethod
    def set_api_key(cls, api_key: str) -> None:
        if not isinstance(api_key, str):
            raise ConfigurationError("api_key must be a string")

        sanitized_key = api_key.strip()
        if not sanitized_key:
            raise ConfigurationError("api_key must not be empty")

        if len(sanitized_key) < GOOGLE_API_KEY_MIN_LENGTH:
            raise ConfigurationError(
                f"api_key must be at least {GOOGLE_API_KEY_MIN_LENGTH} characters"
            )

        try:
            keyring.set_password(API_SERVICE_NAME, API_KEY_ACCOUNT, sanitized_key)
            logger.info("Saved Gemini API key to keyring")
        except PasswordSetError as e:
            logger.error("Failed to set API key in keyring: %s", e)
            raise ConfigurationError(f"Failed to set API key: {e}") from e
        except KeyringError as e:
            logger.error("Keyring error while saving API key: %s", e)
            raise ConfigurationError(f"Keyring error: {e}") from e

    @classmethod
    def delete_api_key(cls) -> None:
        try:
            stored = keyring.get_password(API_SERVICE_NAME, API_KEY_ACCOUNT)
            if stored is None:
                raise ConfigurationError("No API key found in keyring to delete")
            keyring.delete_password(API_SERVICE_NAME, API_KEY_ACCOUNT)
            logger.info("Deleted Gemini API key from keyring")
        except PasswordDeleteError as e:
            logger.error("Failed to delete API key from keyring: %s", e)
            raise ConfigurationError(f"Failed to delete API key: {e}") from e
        except KeyringError as e:
            logger.error("Keyring error while deleting API key: %s", e)
            raise ConfigurationError(f"Keyring error: {e}") from e

    @classmethod
    def show_api_key(cls, api_key: str | None = None) -> str:
        key = cls.get_api_key(api_key=api_key)
        return "*" * len(key)

    @classmethod
    def show_creds(cls, api_key: str | None = None) -> str:
        return cls.show_api_key(api_key=api_key)

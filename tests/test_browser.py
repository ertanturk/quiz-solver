"""Tests for BrowserSession and Playwright Blackboard authentication."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from playwright.sync_api import Error as PlaywrightError

from qs.browser.session import (
    SELECTOR_PASSWORD,
    SELECTOR_SUBMIT,
    SELECTOR_USERNAME,
    BrowserSession,
)
from qs.browser.session import (
    logger as browser_logger,
)
from qs.config import BB_LINK, DEFAULT_BROWSER_CHANNEL, DEFAULT_BROWSER_PROFILE_DIR
from qs.errors.exceptions import BrowserSessionError, ConfigurationError, PageNavigationError


@pytest.fixture
def log_capture():
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    handler.setLevel(logging.DEBUG)
    browser_logger.addHandler(handler)
    original_level = browser_logger.level
    browser_logger.setLevel(logging.DEBUG)
    yield records
    browser_logger.removeHandler(handler)
    browser_logger.setLevel(original_level)


# --- Initialization & Properties Tests ---


def test_browser_session_init_defaults():
    session = BrowserSession()
    assert session.user_data_dir == DEFAULT_BROWSER_PROFILE_DIR
    assert session.channel == DEFAULT_BROWSER_CHANNEL
    assert session.headless is False
    assert session.timeout == 30000.0
    assert "--disable-blink-features=AutomationControlled" in session.args
    assert session.viewport == {"width": 1280, "height": 800}
    assert session.is_open is False


def test_browser_session_init_custom(tmp_path):
    custom_dir = tmp_path / "custom_profile"
    session = BrowserSession(
        user_data_dir=custom_dir,
        channel="chromium",
        headless=True,
        timeout=15000.0,
        args=["--custom-flag"],
        viewport={"width": 1920, "height": 1080},
    )
    assert session.user_data_dir == custom_dir.resolve()
    assert session.channel == "chromium"
    assert session.headless is True
    assert session.timeout == 15000.0
    assert session.args == ["--custom-flag"]
    assert session.viewport == {"width": 1920, "height": 1080}
    assert session.is_open is False


def test_page_and_context_raise_when_not_open():
    session = BrowserSession()
    with pytest.raises(BrowserSessionError, match="not open"):
        _ = session.page
    with pytest.raises(BrowserSessionError, match="not open"):
        _ = session.context


# --- Lifecycle & Mocking Tests ---


def test_open_and_close_lifecycle(tmp_path, log_capture):
    mock_playwright = MagicMock()
    mock_chromium = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()

    mock_playwright.chromium = mock_chromium
    mock_chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path, channel="chrome", headless=True)
        session.open()

        assert session.is_open is True
        assert session.page == mock_page
        assert session.context == mock_context
        mock_chromium.launch_persistent_context.assert_called_once()
        _, kwargs = mock_chromium.launch_persistent_context.call_args
        assert kwargs["user_data_dir"] == str(tmp_path.resolve())
        assert kwargs["channel"] == "chrome"
        assert kwargs["headless"] is True

        # Second open() call should be a no-op
        session.open()
        assert mock_chromium.launch_persistent_context.call_count == 1

        # Close session
        session.close()
        assert session.is_open is False
        mock_context.close.assert_called_once()
        mock_playwright.stop.assert_called_once()

    assert any("Opening browser with channel='chrome'" in r.getMessage() for r in log_capture)
    assert any("Browser opened successfully" in r.getMessage() for r in log_capture)
    assert any("Browser session closed" in r.getMessage() for r in log_capture)


def test_open_creates_new_page_if_no_pages(tmp_path):
    mock_playwright = MagicMock()
    mock_chromium = MagicMock()
    mock_context = MagicMock()
    mock_new_page = MagicMock()

    mock_playwright.chromium = mock_chromium
    mock_chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = []  # No existing page
    mock_context.new_page.return_value = mock_new_page

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path)
        session.open()
        assert session.page == mock_new_page
        mock_context.new_page.assert_called_once()
        session.close()


def test_open_failure_raises_browser_session_error(tmp_path, log_capture):
    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_playwright = MagicMock()
        mock_sync_pw.return_value.start.return_value = mock_playwright
        mock_playwright.chromium.launch_persistent_context.side_effect = PlaywrightError(
            "Executable not found"
        )

        session = BrowserSession(user_data_dir=tmp_path)
        with pytest.raises(BrowserSessionError, match="Failed to launch browser"):
            session.open()

        assert session.is_open is False
        assert any("Failed to launch browser" in r.getMessage() for r in log_capture)


def test_context_manager(tmp_path):
    mock_playwright = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()

    mock_playwright.chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        with BrowserSession(user_data_dir=tmp_path) as session:
            assert session.is_open is True
            assert session.page == mock_page

        assert session.is_open is False
        mock_context.close.assert_called_once()
        mock_playwright.stop.assert_called_once()


# --- Navigation Tests ---


def test_goto_success(tmp_path, log_capture):
    mock_playwright = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()

    mock_playwright.chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path)
        session.open()
        page = session.goto("https://mef.blackboard.com/")

        assert page == mock_page
        mock_page.goto.assert_called_once_with(
            "https://mef.blackboard.com/", wait_until="domcontentloaded"
        )
        session.close()

    assert any("Navigating to https://mef.blackboard.com/" in r.getMessage() for r in log_capture)


def test_goto_failure_raises_page_navigation_error(tmp_path, log_capture):
    mock_playwright = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_page.goto.side_effect = PlaywrightError("Navigation timeout of 30000ms exceeded")

    mock_playwright.chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path)
        session.open()

        with pytest.raises(PageNavigationError, match="Failed to navigate"):
            session.goto("https://invalid.domain/")

        session.close()

    assert any(
        "Failed to navigate to https://invalid.domain/" in r.getMessage() for r in log_capture
    )


# --- Login Tests ---


def test_login_success(tmp_path, log_capture):
    mock_playwright = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_page.url = "about:blank"

    mock_username_locator = MagicMock()
    mock_username_locator.is_visible.return_value = True
    mock_page.locator.return_value = mock_username_locator

    mock_playwright.chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path)
        session.login(username="student_mef", password="secret_password123")

        # Navigated to BB_LINK
        mock_page.goto.assert_called_once_with(BB_LINK, wait_until="domcontentloaded")
        # Checked selector visibility
        mock_page.locator.assert_called_once_with(SELECTOR_USERNAME)
        # Filled username and password in right section
        mock_page.fill.assert_any_call(SELECTOR_USERNAME, "student_mef")
        mock_page.fill.assert_any_call(SELECTOR_PASSWORD, "secret_password123")
        # Clicked submit
        mock_page.click.assert_called_once_with(SELECTOR_SUBMIT)
        # Waited for load state
        mock_page.wait_for_load_state.assert_called_once_with("domcontentloaded")

        session.close()

    # Verify logs: username logged, but secret password NEVER logged!
    assert any(
        "Autofilling Blackboard credentials for user 'student_mef'" in r.getMessage()
        for r in log_capture
    )
    assert any("Clicking sign in button" in r.getMessage() for r in log_capture)
    assert not any("secret_password123" in r.getMessage() for r in log_capture)


def test_login_from_env_and_keyring(tmp_path, monkeypatch):
    monkeypatch.setenv("BB_USERNAME", "env_user")

    mock_playwright = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_page.url = BB_LINK

    mock_username_locator = MagicMock()
    mock_username_locator.is_visible.return_value = True
    mock_page.locator.return_value = mock_username_locator

    mock_playwright.chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with (
        patch("qs.browser.session.sync_playwright") as mock_sync_pw,
        patch(
            "qs.credentials.Credentials.get_credentials_bb",
            return_value=("env_user", "keyring_pwd"),
        ) as mock_get_bb,
    ):
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path)
        session.login()

        mock_get_bb.assert_called_once_with("env_user")
        mock_page.fill.assert_any_call(SELECTOR_USERNAME, "env_user")
        mock_page.fill.assert_any_call(SELECTOR_PASSWORD, "keyring_pwd")
        mock_page.click.assert_called_once_with(SELECTOR_SUBMIT)
        session.close()


def test_login_missing_username_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("BB_USERNAME", raising=False)
    session = BrowserSession(user_data_dir=tmp_path)
    # Mock open
    session._page = MagicMock()
    session._context = MagicMock()

    with pytest.raises(ConfigurationError, match="username is required"):
        session.login(username="")


def test_login_skips_when_form_not_visible(tmp_path, log_capture):
    mock_playwright = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    mock_page.url = f"{BB_LINK}ultra/course"

    mock_username_locator = MagicMock()
    mock_username_locator.is_visible.return_value = False
    mock_page.locator.return_value = mock_username_locator

    mock_playwright.chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path)
        session.login(username="student_mef", password="pass")

        # Should NOT fill or click
        mock_page.fill.assert_not_called()
        mock_page.click.assert_not_called()
        session.close()

    assert any(
        "Login form not visible; session may already be authenticated" in r.getMessage()
        for r in log_capture
    )


# --- Screenshot Test ---


def test_screenshot(tmp_path):
    mock_playwright = MagicMock()
    mock_context = MagicMock()
    mock_page = MagicMock()
    fake_png = b"\x89PNG\r\n\x1a\nfakeimagebytes"
    mock_page.screenshot.return_value = fake_png

    mock_playwright.chromium.launch_persistent_context.return_value = mock_context
    mock_context.pages = [mock_page]

    with patch("qs.browser.session.sync_playwright") as mock_sync_pw:
        mock_sync_pw.return_value.start.return_value = mock_playwright

        session = BrowserSession(user_data_dir=tmp_path)
        session.open()

        out = session.screenshot()
        assert out == fake_png
        mock_page.screenshot.assert_called_once_with(path=None, full_page=False)

        save_path = tmp_path / "shot.png"
        out2 = session.screenshot(path=save_path, full_page=True)
        assert out2 == fake_png
        mock_page.screenshot.assert_called_with(path=str(save_path.resolve()), full_page=True)

        session.close()


# --- End-to-End Test with Real Playwright (Headless with Chrome) ---


def test_real_playwright_html_form_fill(tmp_path):
    """Verify real Playwright interacts with Blackboard login form markup."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Blackboard Learn</title></head>
    <body>
      <div id="login-form">
        <form action="#" onsubmit="return false;">
          <input type="text" id="user_id" name="user_id" />
          <input type="password" id="password" name="password" />
          <input type="submit" id="entry-login" value="Sign In" />
        </form>
      </div>
    </body>
    </html>
    """
    html_file = tmp_path / "login.html"
    html_file.write_text(html_content, encoding="utf-8")
    file_url = f"file://{html_file.resolve()}"

    session = BrowserSession(
        user_data_dir=tmp_path / "chrome_profile",
        channel="chrome",
        headless=True,
        timeout=3000.0,
    )
    with session:
        session.login(
            username="real_student",
            password="test_password_456",
            url=file_url,
        )
        username_val = session.page.input_value(SELECTOR_USERNAME)
        password_val = session.page.input_value(SELECTOR_PASSWORD)
        assert username_val == "real_student"
        assert password_val == "test_password_456"

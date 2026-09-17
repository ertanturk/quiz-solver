"""Playwright browser session management and Blackboard authentication."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from qs.config import BB_LINK, DEFAULT_BROWSER_CHANNEL, DEFAULT_BROWSER_PROFILE_DIR
from qs.credentials.credentials import Credentials
from qs.errors.exceptions import BrowserSessionError, ConfigurationError, PageNavigationError
from qs.logger import get_logger

logger = get_logger(__name__)


WaitUntil = Literal["commit", "domcontentloaded", "load", "networkidle"]


# Blackboard login form selectors (right section)
SELECTOR_USERNAME = "#user_id"
SELECTOR_PASSWORD = "#password"
SELECTOR_SUBMIT = "#entry-login"


class BrowserSession:
    """Manages Playwright persistent browser context and Blackboard navigation."""

    def __init__(
        self,
        user_data_dir: Path | str | None = None,
        channel: str | None = DEFAULT_BROWSER_CHANNEL,
        headless: bool = False,
        timeout: float = 30000.0,
        args: list[str] | None = None,
        viewport: dict[str, int] | None = None,
    ) -> None:
        """Initialize BrowserSession configuration.

        Args:
            user_data_dir: Directory to persist cookies and local storage.
            channel: Browser distribution channel (e.g. 'chrome', 'msedge', or None for chromium).
            headless: Whether to run browser without GUI.
            timeout: Default timeout in milliseconds for Playwright actions.
            args: Extra command line flags passed to the browser.
            viewport: Dimensions for page viewport or None to use default.
        """
        self.user_data_dir = (
            Path(user_data_dir).expanduser().resolve()
            if user_data_dir is not None
            else DEFAULT_BROWSER_PROFILE_DIR
        )
        self.channel = channel
        self.headless = headless
        self.timeout = timeout
        self.args = args or ["--disable-blink-features=AutomationControlled"]
        self.viewport = viewport if viewport is not None else {"width": 1280, "height": 800}

        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    @property
    def is_open(self) -> bool:
        """Return True if browser context and page are active."""
        return self._context is not None and self._page is not None

    @property
    def page(self) -> Page:
        """Return the active page instance."""
        if self._page is None:
            raise BrowserSessionError("Browser session is not open. Call open() first.")
        return self._page

    @property
    def context(self) -> BrowserContext:
        """Return the active browser context instance."""
        if self._context is None:
            raise BrowserSessionError("Browser session is not open. Call open() first.")
        return self._context

    def open(self) -> BrowserSession:
        """Launch persistent Playwright browser context using Chrome.

        Returns:
            Self instance.
        """
        if self.is_open:
            logger.debug("Browser session is already open")
            return self

        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Opening browser with channel='%s', profile='%s', headless=%s",
            self.channel,
            self.user_data_dir,
            self.headless,
        )

        try:
            self._playwright = sync_playwright().start()
            launch_kwargs: dict[str, Any] = {
                "user_data_dir": str(self.user_data_dir),
                "headless": self.headless,
                "args": self.args,
                "ignore_default_args": ["--enable-automation"],
                "viewport": self.viewport,
            }
            if self.channel:
                launch_kwargs["channel"] = self.channel

            self._context = self._playwright.chromium.launch_persistent_context(**launch_kwargs)
            self._context.set_default_timeout(self.timeout)

            if self._context.pages:
                self._page = self._context.pages[0]
            else:
                self._page = self._context.new_page()

            logger.info("Browser opened successfully")
            return self
        except PlaywrightError as e:
            logger.error("Failed to launch browser: %s", e)
            self.close()
            raise BrowserSessionError(f"Failed to launch browser: {e}") from e
        except Exception as e:
            logger.error("Unexpected error opening browser: %s", e)
            self.close()
            raise BrowserSessionError(f"Unexpected error opening browser: {e}") from e

    def close(self) -> None:
        """Close browser context and stop Playwright process."""
        if self._context is not None:
            try:
                self._context.close()
            except Exception as e:
                logger.debug("Error closing context: %s", e)
            finally:
                self._context = None
                self._page = None

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception as e:
                logger.debug("Error stopping Playwright: %s", e)
            finally:
                self._playwright = None

        logger.info("Browser session closed")

    def __enter__(self) -> BrowserSession:
        """Enter runtime context."""
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit runtime context and clean up browser."""
        self.close()

    def goto(
        self,
        url: str = BB_LINK,
        wait_until: WaitUntil | None = "domcontentloaded",
    ) -> Page:
        """Navigate active page to URL.

        Args:
            url: Destination URL, defaults to BB_LINK.
            wait_until: Navigation event to wait for.

        Returns:
            Active Page instance.
        """
        if not self.is_open:
            self.open()

        logger.info("Navigating to %s", url)
        try:
            self.page.goto(url, wait_until=wait_until)
            return self.page
        except PlaywrightError as e:
            logger.error("Failed to navigate to %s: %s", url, e)
            raise PageNavigationError(f"Failed to navigate to {url}: {e}") from e

    def login(
        self,
        username: str | None = None,
        password: str | None = None,
        url: str = BB_LINK,
        screenshot_path: Path | str | None = None,
    ) -> None:
        """Navigate to Blackboard, autofill credentials in right section, and submit.

        Args:
            username: Blackboard username. If None, falls back to env or keyring.
            password: Blackboard password. If None, falls back to env or keyring.
            url: Blackboard URL, defaults to BB_LINK.
            screenshot_path: Optional path to save screenshot after login.
        """
        if not self.is_open:
            self.open()

        # Resolve credentials
        user = (
            username.strip()
            if isinstance(username, str)
            else os.environ.get("BB_USERNAME", "").strip()
        )
        pwd = password if isinstance(password, str) else os.environ.get("BB_PASSWORD")

        if not user:
            logger.error("Blackboard username is required for login")
            raise ConfigurationError("Blackboard username is required for login")

        if not pwd:
            user, pwd = Credentials.get_credentials_bb(user)

        # Navigate to Blackboard if not already on the page
        current_url = self.page.url
        if not current_url.startswith(url):
            self.goto(url)

        # Check if login form is present
        try:
            username_locator = self.page.locator(SELECTOR_USERNAME)
            if not username_locator.is_visible():
                logger.info("Login form not visible; session may already be authenticated")
                if screenshot_path is not None:
                    self.screenshot(path=screenshot_path)
                return
        except PlaywrightError:
            logger.info("Login form not found; session may already be authenticated")
            if screenshot_path is not None:
                self.screenshot(path=screenshot_path)
            return

        # Autofill credentials in the right section
        logger.info("Autofilling Blackboard credentials for user '%s'", user)
        self.page.fill(SELECTOR_USERNAME, user)
        self.page.fill(SELECTOR_PASSWORD, pwd)

        # Click Sign In button
        logger.info("Clicking sign in button")
        self.page.click(SELECTOR_SUBMIT)

        try:
            self.page.wait_for_load_state("domcontentloaded")
        except PlaywrightError as e:
            logger.debug("Wait for load state timed out or failed: %s", e)

        logger.info("Sign in submitted successfully for '%s'", user)

        if screenshot_path is not None:
            self.screenshot(path=screenshot_path)

    def screenshot(self, path: Path | str | None = None, full_page: bool = False) -> bytes:
        """Capture screenshot of the active page.

        Args:
            path: Optional filesystem path to save image file.
            full_page: Whether to take full scrollable page screenshot.

        Returns:
            Screenshot image bytes.
        """
        target_path = str(Path(path).expanduser().resolve()) if path is not None else None
        image_bytes = self.page.screenshot(path=target_path, full_page=full_page)
        if target_path:
            logger.info("Saved screenshot to %s", target_path)
        return image_bytes

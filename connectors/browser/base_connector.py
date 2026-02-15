"""
Life OS — Browser Base Connector

Extends the standard BaseConnector with browser automation capabilities.
Any connector can subclass this to gain:

    1. Automatic fallback from API to browser when API fails
    2. Login flow with credential vault + 2FA support
    3. Session persistence (log in once, reuse for weeks)
    4. Human-emulated interactions that don't trigger bot detection
    5. Rate limiting and respectful crawling

The idea: you configure a connector's API credentials first. If the API
breaks, gets rate-limited, or the service kills it, the browser layer
kicks in automatically and scrapes the same data through the UI.

For services that NEVER had APIs (WhatsApp Web, Reddit old.reddit,
YouTube subscriptions), the browser IS the primary connector.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from connectors.base.connector import BaseConnector
from connectors.browser.engine import (
    BrowserEngine,
    CredentialVault,
    HumanEmulator,
    PageInteractor,
)
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


class BrowserBaseConnector(BaseConnector):
    """
    A connector that can operate in two modes:
        - API mode (fast, reliable, preferred)
        - Browser mode (fallback, works when APIs fail or don't exist)

    Subclasses implement:
        - api_sync() — attempt API-based data retrieval
        - browser_sync() — browser-based fallback
        - get_login_url() — where to navigate for login
        - get_login_selectors() — CSS selectors for login form
    """

    # Subclasses override these
    SITE_ID: str = ""
    LOGIN_URL: str = ""
    REQUIRES_2FA: bool = False

    # Rate limiting
    MIN_REQUEST_INTERVAL: float = 2.0  # Seconds between page loads
    MAX_PAGES_PER_SYNC: int = 20

    def __init__(self, event_bus: EventBus, db: DatabaseManager,
                 config: dict[str, Any],
                 browser_engine: Optional[BrowserEngine] = None,
                 credential_vault: Optional[CredentialVault] = None):
        super().__init__(event_bus, db, config)

        self._browser_engine = browser_engine or BrowserEngine(
            data_dir=config.get("browser_data_dir", "./data/browser")
        )
        self._credential_vault = credential_vault or CredentialVault(
            vault_path=config.get("credential_vault_path", "./data/credentials")
        )
        self._human = HumanEmulator(
            speed_factor=config.get("human_speed_factor", 1.0)
        )
        self._interactor = PageInteractor(self._human)

        self._context = None  # Browser context (persistent per site)
        self._page = None     # Active page

        self._api_mode = config.get("prefer_api", True)
        self._api_failures = 0
        self._api_failure_threshold = config.get("api_failure_threshold", 3)
        self._last_request_time = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def authenticate(self) -> bool:
        """
        Authenticate via API first. If that fails, fall back to browser login.
        """
        if self._api_mode:
            try:
                result = await self.api_authenticate()
                if result:
                    print(f"  [{self.CONNECTOR_ID}] Authenticated via API")
                    return True
            except Exception as e:
                print(f"  [{self.CONNECTOR_ID}] API auth failed: {e}")

        # Fall back to browser
        return await self._browser_login()

    async def sync(self) -> int:
        """
        Try API sync first. On failure, switch to browser mode.
        After N consecutive API failures, auto-switch to browser-only.
        """
        if self._api_mode and self._api_failures < self._api_failure_threshold:
            try:
                count = await self.api_sync()
                self._api_failures = 0  # Reset on success
                return count
            except Exception as e:
                self._api_failures += 1
                print(
                    f"  [{self.CONNECTOR_ID}] API sync failed "
                    f"({self._api_failures}/{self._api_failure_threshold}): {e}"
                )

                if self._api_failures >= self._api_failure_threshold:
                    print(
                        f"  [{self.CONNECTOR_ID}] Switching to browser mode "
                        f"after {self._api_failures} consecutive API failures"
                    )

        # Browser mode
        try:
            return await self._browser_sync_wrapper()
        except Exception as e:
            print(f"  [{self.CONNECTOR_ID}] Browser sync failed: {e}")
            traceback.print_exc()
            return 0

    async def stop(self):
        """Clean up browser resources."""
        if self._context:
            # Save session before closing
            try:
                await self._browser_engine.save_session(self._context, self.SITE_ID)
            except Exception:
                pass
            try:
                await self._context.close()
            except Exception:
                pass
        await super().stop()

    # ------------------------------------------------------------------
    # Browser login flow
    # ------------------------------------------------------------------

    async def _browser_login(self) -> bool:
        """
        Log in to a website using stored credentials.
        Handles: credential lookup → navigate → fill form → 2FA → save session
        """
        creds = self._credential_vault.get_credential(self.SITE_ID)
        if not creds:
            print(f"  [{self.CONNECTOR_ID}] No credentials found for {self.SITE_ID}")
            print(f"  [{self.CONNECTOR_ID}] Load Proton Pass export or add to manual vault")
            return False

        try:
            # Start browser if needed
            await self._browser_engine.start()

            # Create context (reuses saved session if available)
            self._context = await self._browser_engine.create_context(
                self.SITE_ID,
                timezone_id=self.config.get("timezone", "America/Chicago"),
            )
            self._page = await self._browser_engine.new_page(self._context)

            # Check if session is still valid
            if self._browser_engine.session_manager.has_session(self.SITE_ID):
                await self._page.goto(self.get_login_url(), wait_until="networkidle")
                if await self.is_logged_in(self._page):
                    print(f"  [{self.CONNECTOR_ID}] Session still valid")
                    return True

            # Fresh login needed
            print(f"  [{self.CONNECTOR_ID}] Logging in via browser...")
            await self._page.goto(self.get_login_url(), wait_until="networkidle")
            await self._human.wait_human(1.0, 3.0)

            # Get selectors from subclass
            selectors = self.get_login_selectors()

            # Perform login
            await self._interactor.login(
                self._page, creds,
                username_selector=selectors.get("username", "input[type='email']"),
                password_selector=selectors.get("password", "input[type='password']"),
                submit_selector=selectors.get("submit", "button[type='submit']"),
            )

            # Handle 2FA if needed
            if self.REQUIRES_2FA:
                totp_code = self._credential_vault.get_totp(self.SITE_ID)
                if totp_code:
                    code_selector = selectors.get("totp", "input[name='code']")
                    await self._interactor.handle_2fa(self._page, totp_code, code_selector)
                else:
                    print(f"  [{self.CONNECTOR_ID}] 2FA required but no TOTP URI configured")
                    print(f"  [{self.CONNECTOR_ID}] Add totp_uri to credential vault")
                    return False

            # Verify login succeeded
            await self._human.wait_human(2.0, 5.0)
            if await self.is_logged_in(self._page):
                # Save session for reuse
                await self._browser_engine.save_session(self._context, self.SITE_ID)
                print(f"  [{self.CONNECTOR_ID}] Browser login successful")
                return True
            else:
                print(f"  [{self.CONNECTOR_ID}] Login appears to have failed")
                return False

        except Exception as e:
            print(f"  [{self.CONNECTOR_ID}] Browser login error: {e}")
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Browser sync wrapper (rate limiting + session management)
    # ------------------------------------------------------------------

    async def _browser_sync_wrapper(self) -> int:
        """Wraps browser_sync with rate limiting and error recovery."""
        if not self._page:
            success = await self._browser_login()
            if not success:
                return 0

        try:
            count = await self.browser_sync(self._page, self._human, self._interactor)

            # Save session after successful sync
            if self._context:
                await self._browser_engine.save_session(self._context, self.SITE_ID)

            return count

        except Exception as e:
            # Session might have expired
            if "session" in str(e).lower() or "login" in str(e).lower():
                print(f"  [{self.CONNECTOR_ID}] Session expired, re-authenticating...")
                self._browser_engine.session_manager.clear_session(self.SITE_ID)
                if await self._browser_login():
                    return await self.browser_sync(self._page, self._human, self._interactor)
            raise

    async def rate_limit_wait(self):
        """Enforce minimum time between page loads."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            jitter = self.MIN_REQUEST_INTERVAL * 0.3
            wait_time = (self.MIN_REQUEST_INTERVAL - elapsed) + (jitter * (0.5 - __import__("random").random()))
            await asyncio.sleep(max(0, wait_time))
        self._last_request_time = asyncio.get_event_loop().time()

    async def navigate_with_rate_limit(self, page: Any, url: str):
        """Navigate to a URL with rate limiting."""
        await self.rate_limit_wait()
        await page.goto(url, wait_until="networkidle")

    # ------------------------------------------------------------------
    # Subclass interface (override these)
    # ------------------------------------------------------------------

    async def api_authenticate(self) -> bool:
        """API-based authentication. Override if the service has an API."""
        return False

    async def api_sync(self) -> int:
        """API-based sync. Override if the service has an API."""
        raise NotImplementedError("No API available — using browser mode")

    async def browser_sync(self, page: Any, human: HumanEmulator,
                           interactor: PageInteractor) -> int:
        """
        Browser-based sync. Subclasses implement this to scrape data.
        
        Args:
            page: The logged-in Playwright page
            human: Human emulator for realistic interactions
            interactor: High-level page interaction helpers
            
        Returns:
            Number of events extracted
        """
        raise NotImplementedError("Subclass must implement browser_sync()")

    def get_login_url(self) -> str:
        """URL to navigate to for login."""
        return self.LOGIN_URL

    def get_login_selectors(self) -> dict[str, str]:
        """
        CSS selectors for the login form.
        Return dict with keys: username, password, submit, totp (optional)
        """
        return {
            "username": "input[type='email'], input[name='username'], input[name='email']",
            "password": "input[type='password']",
            "submit": "button[type='submit'], input[type='submit']",
        }

    async def is_logged_in(self, page: Any) -> bool:
        """
        Check if currently logged in. Subclasses should override this
        to check for a logged-in indicator on the page.
        """
        # Default: check if we're NOT on a login page
        url = page.url.lower()
        return not any(kw in url for kw in ["login", "signin", "auth", "sso"])

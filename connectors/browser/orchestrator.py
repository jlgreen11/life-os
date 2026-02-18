"""
Life OS — Browser Fallback Orchestrator

The central manager for all browser-based automation. Handles:

    1. Shared browser engine (one Chromium instance, many contexts)
    2. Shared credential vault (one Proton Pass export, all sites)
    3. API → Browser fallback detection and switching
    4. Rate limiting across all browser connectors
    5. Health monitoring and session refresh

The orchestrator ensures that:
    - Only one browser instance runs (saves ~500MB RAM vs one per connector)
    - Credentials are loaded once and shared
    - Rate limits are respected globally, not just per-connector
    - Failed sessions are cleaned up and retried

Configuration:
    browser:
      enabled: true
      headless: true
      data_dir: "./data/browser"
      credential_source: "proton_pass"
      proton_pass_export: "./data/credentials/proton_pass_export.json"
      manual_vault: "./data/credentials/manual_vault.json"
      human_speed_factor: 1.0       # 1.0 = normal, 0.5 = faster
      global_rate_limit: 2.0        # Min seconds between ANY page load
      max_concurrent_contexts: 3    # Max sites browsed simultaneously
      session_refresh_hours: 168    # Re-login after 7 days
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from connectors.browser.engine import BrowserEngine, CredentialVault
from connectors.browser.base_connector import BrowserBaseConnector
from connectors.browser.generic import GenericBrowserConnector, create_browser_connectors
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class BrowserOrchestrator:
    """
    Manages the shared browser infrastructure for all browser-based connectors.
    """

    def __init__(self, event_bus: EventBus, db: DatabaseManager, config: dict):
        self.event_bus = event_bus
        self.db = db
        # Extract the "browser" sub-key from the top-level config
        self.config = config.get("browser", {})

        self._enabled = self.config.get("enabled", False)
        self._data_dir = self.config.get("data_dir", "./data/browser")

        # Shared infrastructure — one engine and one vault are created once
        # and injected into every browser connector to avoid duplication.
        self._engine: Optional[BrowserEngine] = None
        self._vault: Optional[CredentialVault] = None

        # Managed connectors — all browser-based connectors created by
        # _create_connectors() are stored here for lifecycle management.
        self._connectors: list[BrowserBaseConnector] = []

        # Global rate limiting — ensures ALL browser connectors collectively
        # respect a minimum delay between page loads, and limits how many
        # sites can be browsed concurrently via an asyncio semaphore.
        self._last_global_request = 0
        self._global_rate_limit = self.config.get("global_rate_limit", 2.0)
        self._semaphore = asyncio.Semaphore(
            self.config.get("max_concurrent_contexts", 3)
        )

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def connectors(self) -> list[BrowserBaseConnector]:
        return self._connectors

    async def start(self):
        """Initialize the shared browser engine and credential vault."""
        if not self._enabled:
            return

        logger.info("Initializing browser automation layer...")

        # Launch the single shared Chromium instance that all browser
        # connectors will share (one process, many isolated contexts).
        self._engine = BrowserEngine(data_dir=self._data_dir)
        await self._engine.start()
        logger.info("Chromium launched (stealth mode)")

        # Load credentials
        self._vault = CredentialVault(
            vault_path=str(Path(self._data_dir) / "credentials")
        )

        cred_source = self.config.get("credential_source", "manual")

        if cred_source == "proton_pass":
            export_path = self.config.get("proton_pass_export", "")
            if export_path and Path(export_path).exists():
                self._vault.load_proton_pass_export(export_path)
                logger.info("Proton Pass vault loaded (%d sites)", len(self._vault.list_sites()))
            else:
                logger.warning("Proton Pass export not found at %s", export_path)

        # Also load manual vault
        manual_path = self.config.get("manual_vault", "")
        if manual_path:
            self._vault.load_manual_vault(manual_path)

        # Create browser-specific connectors
        await self._create_connectors()

        logger.info("%d browser connectors ready", len(self._connectors))

    async def _create_connectors(self):
        """Create all browser-based connectors from config."""
        connector_configs = self.config.get("connectors", {})

        # Named browser connectors — each is instantiated only if its key
        # appears in the config, and receives the shared engine + vault.
        if "whatsapp" in connector_configs:
            from connectors.browser.whatsapp import WhatsAppConnector
            c = WhatsAppConnector(
                self.event_bus, self.db, connector_configs["whatsapp"],
                browser_engine=self._engine, credential_vault=self._vault,
            )
            self._connectors.append(c)

        if "youtube" in connector_configs:
            from connectors.browser.youtube import YouTubeConnector
            c = YouTubeConnector(
                self.event_bus, self.db, connector_configs["youtube"],
                browser_engine=self._engine, credential_vault=self._vault,
            )
            self._connectors.append(c)

        if "reddit" in connector_configs:
            from connectors.browser.reddit import RedditConnector
            c = RedditConnector(
                self.event_bus, self.db, connector_configs["reddit"],
                browser_engine=self._engine, credential_vault=self._vault,
            )
            self._connectors.append(c)

        # Generic browser sources — user-defined scraping targets configured
        # entirely via YAML (CSS selectors, URLs). Uses the factory function
        # to create one GenericBrowserConnector per source entry.
        generic_configs = connector_configs.get("generic_sources", [])
        if generic_configs:
            generics = create_browser_connectors(
                self.event_bus, self.db, generic_configs,
                browser_engine=self._engine, credential_vault=self._vault,
            )
            self._connectors.extend(generics)

    async def start_connectors(self):
        """Authenticate and start all browser connectors."""
        if not self._enabled:
            return

        for connector in self._connectors:
            try:
                await connector.start()
                logger.info("%s (browser) started", connector.DISPLAY_NAME)
            except Exception as e:
                logger.error("%s (browser) failed to start: %s", connector.DISPLAY_NAME, e)

    async def stop(self):
        """Shut down all browser connectors and the shared engine."""
        # Stop connectors first (saves sessions), then tear down the
        # shared engine. Errors are swallowed so one failing connector
        # does not prevent the others from cleaning up.
        for connector in self._connectors:
            try:
                await connector.stop()
            except Exception:
                pass

        if self._engine:
            await self._engine.stop()

    async def global_rate_limit(self):
        """Enforce global rate limiting across all browser connectors."""
        # The semaphore caps concurrent page loads; the elapsed-time check
        # enforces a minimum delay between any two requests system-wide.
        async with self._semaphore:
            now = time.time()
            elapsed = now - self._last_global_request
            if elapsed < self._global_rate_limit:
                await asyncio.sleep(self._global_rate_limit - elapsed)
            self._last_global_request = time.time()

    def get_status(self) -> dict:
        """Get status of all browser connectors, including current mode (API or browser)."""
        return {
            "enabled": self._enabled,
            "engine_running": self._engine is not None,
            "credential_sites": len(self._vault.list_sites()) if self._vault else 0,
            "connectors": [
                {
                    "id": c.CONNECTOR_ID,
                    "name": c.DISPLAY_NAME,
                    "api_failures": c._api_failures,
                    "mode": "api" if (c._api_mode and c._api_failures < c._api_failure_threshold) else "browser",
                }
                for c in self._connectors
            ],
        }

    def get_vault_sites(self) -> list[str]:
        """List all sites with stored credentials."""
        if self._vault:
            return self._vault.list_sites()
        return []


# ===========================================================================
# API Fallback Wrapper
# ===========================================================================

class APIFallbackWrapper:
    """
    Wraps an existing API-based connector and adds browser fallback.
    
    Usage:
        proton_connector = ProtonMailConnector(...)
        wrapped = APIFallbackWrapper(
            api_connector=proton_connector,
            browser_engine=engine,
            credential_vault=vault,
            fallback_config={...},
        )
        # Now `wrapped` tries API first, falls back to browser
    """

    def __init__(self, api_connector: Any,
                 browser_engine: BrowserEngine,
                 credential_vault: CredentialVault,
                 fallback_config: dict):
        # Wraps an existing API connector and adds browser fallback on failure
        self.api_connector = api_connector
        self._browser_engine = browser_engine
        self._credential_vault = credential_vault
        self._fallback_config = fallback_config

        # Tracks consecutive API failures to decide when to switch modes
        self._api_failures = 0
        self._failure_threshold = fallback_config.get("failure_threshold", 3)
        self._browser_context = None

    async def sync(self) -> int:
        """Try API, fall back to browser on failure."""
        if self._api_failures < self._failure_threshold:
            try:
                count = await self.api_connector.sync()
                self._api_failures = 0
                return count
            except Exception as e:
                self._api_failures += 1
                logger.warning(
                    "[%s] API failed (%d/%d): %s",
                    self.api_connector.CONNECTOR_ID,
                    self._api_failures,
                    self._failure_threshold,
                    e,
                )

        # Browser fallback
        logger.info("[%s] Using browser fallback...", self.api_connector.CONNECTOR_ID)
        return await self._browser_fallback_sync()

    async def _browser_fallback_sync(self) -> int:
        """Override in subclasses with site-specific browser scraping."""
        logger.warning("[%s] No browser fallback implemented", self.api_connector.CONNECTOR_ID)
        return 0

    def reset_api(self):
        """Reset API failure count (e.g., after config change)."""
        self._api_failures = 0

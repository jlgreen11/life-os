"""
Life OS — Generic Browser Connector

A configurable browser connector that can scrape ANY website.
Instead of writing custom code for every service, you define
extraction rules in YAML.

This is the "teach the AI a new source in 5 minutes" connector.

Example configuration:
    connectors:
      browser_sources:
        - site_id: "hackernews"
          name: "Hacker News"
          login_url: null              # No login needed
          feed_url: "https://news.ycombinator.com/"
          sync_interval: 1800
          selectors:
            item: ".athing"
            title: ".titleline > a"
            url: ".titleline > a@href"
            score: ".score"
            meta: ".subtext"
          event_type: "content.feed.new_item"
          max_items: 30

        - site_id: "linkedin"
          name: "LinkedIn"
          login_url: "https://www.linkedin.com/login"
          feed_url: "https://www.linkedin.com/feed/"
          sync_interval: 3600
          login_selectors:
            username: "#username"
            password: "#password"
            submit: "button[type='submit']"
          selectors:
            item: ".feed-shared-update-v2"
            title: ".feed-shared-text span[dir='ltr']"
            author: ".feed-shared-actor__name"
          event_type: "content.feed.new_item"
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from connectors.browser.base_connector import BrowserBaseConnector
from connectors.browser.engine import BrowserEngine, CredentialVault, HumanEmulator, PageInteractor
from services.event_bus.bus import EventBus
from storage.database import DatabaseManager


class GenericBrowserConnector(BrowserBaseConnector):
    """
    A browser connector driven entirely by configuration.
    Provide CSS selectors and it extracts structured data from any site.
    """

    def __init__(self, event_bus: EventBus, db: DatabaseManager,
                 config: dict[str, Any],
                 browser_engine: Optional[BrowserEngine] = None,
                 credential_vault: Optional[CredentialVault] = None):
        # Override class-level constants from config BEFORE calling super(),
        # because BaseConnector.__init__ reads CONNECTOR_ID for subscriptions.
        self.CONNECTOR_ID = config.get("site_id", "generic")
        self.DISPLAY_NAME = config.get("name", "Generic Browser")
        self.SITE_ID = config.get("site_id", "generic")
        self.LOGIN_URL = config.get("login_url", "")
        self.SYNC_INTERVAL_SECONDS = config.get("sync_interval", 1800)

        super().__init__(event_bus, db, config, browser_engine, credential_vault)

        # All extraction behaviour is driven by these config values —
        # no custom code needed per site.
        self._feed_url = config.get("feed_url", "")         # Page to scrape
        self._selectors = config.get("selectors", {})        # CSS selectors for data extraction
        self._event_type = config.get("event_type", "content.feed.new_item")
        self._max_items = config.get("max_items", 30)        # Cap to prevent runaway scraping
        self._custom_login_selectors = config.get("login_selectors", {})
        self._requires_login = bool(config.get("login_url"))  # Skip login flow for public sites

    def get_login_selectors(self) -> dict[str, str]:
        if self._custom_login_selectors:
            return self._custom_login_selectors
        return super().get_login_selectors()

    async def authenticate(self) -> bool:
        if not self._requires_login:
            # No login needed — just start browser
            await self._browser_engine.start()
            self._context = await self._browser_engine.create_context(self.SITE_ID)
            self._page = await self._browser_engine.new_page(self._context)
            return True
        return await super().authenticate()

    async def browser_sync(self, page: Any, human: HumanEmulator,
                           interactor: PageInteractor) -> int:
        """Extract items from the configured feed URL using CSS selectors."""
        count = 0

        # Navigate to the configured feed page with rate limiting
        await self.navigate_with_rate_limit(page, self._feed_url)
        await human.wait_human(1.5, 3.0)

        # Scroll a couple of times to trigger lazy-loaded / infinite-scroll content
        for _ in range(2):
            await human.scroll(page, "down", 600)
            await human.wait_human(0.8, 1.5)

        # Build and run a JavaScript snippet that extracts structured data
        # from the page using the user-configured CSS selectors.
        item_selector = self._selectors.get("item", "article, .item, .post")
        field_selectors = {k: v for k, v in self._selectors.items() if k != "item"}

        js_code = self._build_extraction_js(item_selector, field_selectors)
        items = await page.evaluate(js_code)

        # Deduplicate: compute an MD5 hash of each item's content and compare
        # against previously seen hashes stored in the sync cursor.
        seen_hashes = self._get_seen_hashes()
        new_items = []

        for item in items[:self._max_items]:
            # Generate a stable hash for deduplication
            # SECURITY: md5 used only for content deduplication, not security
            item_hash = hashlib.md5(  # noqa: S324
                json.dumps(item, sort_keys=True).encode(),
                usedforsecurity=False,
            ).hexdigest()[:16]

            if item_hash not in seen_hashes:
                item["_hash"] = item_hash
                new_items.append(item)

        for item in new_items:
            payload = {
                "source_site": self.SITE_ID,
                "source_name": self.DISPLAY_NAME,
                **{k: v for k, v in item.items() if not k.startswith("_")},
            }

            await self.publish_event(
                self._event_type, payload,
                priority="low",
                metadata={"domain": "media", "source": self.SITE_ID},
            )
            count += 1

        # Update seen hashes
        if new_items:
            new_hashes = [i["_hash"] for i in new_items]
            self._update_seen_hashes(new_hashes)

        return count

    def _build_extraction_js(self, item_selector: str,
                              field_selectors: dict[str, str]) -> str:
        """Build JavaScript extraction code from CSS selectors.

        Generates a self-contained JS function that iterates over DOM elements
        matching item_selector and extracts the configured fields from each.
        """
        field_extractors = []
        for field_name, selector in field_selectors.items():
            # Support @attr syntax: "a@href" means get the href attribute of <a>
            # (as opposed to the default textContent extraction).
            if "@" in selector:
                css_part, attr = selector.rsplit("@", 1)
                field_extractors.append(
                    f'"{field_name}": (function() {{ '
                    f'const el = item.querySelector("{css_part}"); '
                    f'return el ? el.getAttribute("{attr}") || "" : ""; '
                    f'}})()'
                )
            else:
                field_extractors.append(
                    f'"{field_name}": (function() {{ '
                    f'const el = item.querySelector("{selector}"); '
                    f'return el ? el.textContent?.trim() || "" : ""; '
                    f'}})()'
                )

        fields_js = ",\n                        ".join(field_extractors)

        return f"""
            () => {{
                const items = [];
                const elements = document.querySelectorAll('{item_selector}');
                
                for (const item of elements) {{
                    try {{
                        const extracted = {{
                            {fields_js}
                        }};
                        
                        // Only include items with at least one non-empty field
                        if (Object.values(extracted).some(v => v && v.trim())) {{
                            items.push(extracted);
                        }}
                    }} catch (e) {{}}
                }}
                return items;
            }}
        """

    def _get_seen_hashes(self) -> set:
        """Load previously seen item hashes from the sync cursor."""
        cursor = self.get_sync_cursor()
        if cursor:
            try:
                return set(json.loads(cursor))
            except Exception:
                pass
        return set()

    def _update_seen_hashes(self, new_hashes: list[str]):
        """Persist new hashes to the sync cursor, capping at 2000 entries."""
        seen = self._get_seen_hashes()
        seen.update(new_hashes)
        # Keep only the most recent 2000 hashes to prevent unbounded growth
        self.set_sync_cursor(json.dumps(list(seen)[-2000:]))

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(f"{self.DISPLAY_NAME} connector is read-only")

    async def health_check(self) -> dict[str, Any]:
        if self._page:
            return {"status": "ok", "connector": self.CONNECTOR_ID, "mode": "browser"}
        return {"status": "not_started", "connector": self.CONNECTOR_ID}


# ===========================================================================
# Factory: Create connectors from YAML config
# ===========================================================================

def create_browser_connectors(
    event_bus: EventBus, db: DatabaseManager, configs: list[dict],
    browser_engine: Optional[BrowserEngine] = None,
    credential_vault: Optional[CredentialVault] = None,
) -> list[GenericBrowserConnector]:
    """
    Factory function to create multiple browser connectors from config.

    Each entry in *configs* becomes one GenericBrowserConnector instance.
    The shared browser_engine and credential_vault are injected so all
    generic connectors share a single Chromium process and credential store.

    Usage in settings.yaml:
        connectors:
          browser_sources:
            - site_id: "hackernews"
              name: "Hacker News"
              feed_url: "https://news.ycombinator.com/"
              selectors:
                item: ".athing"
                title: ".titleline > a"
                url: ".titleline > a@href"
    """
    connectors = []
    for config in configs:
        connector = GenericBrowserConnector(
            event_bus, db, config, browser_engine, credential_vault,
        )
        connectors.append(connector)
    return connectors

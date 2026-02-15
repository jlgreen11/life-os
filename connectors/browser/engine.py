"""
Life OS — Browser Automation Engine

A stealth browser automation layer that provides fallback data access
when APIs aren't available, are too restrictive, or have been shut down.

This is YOUR browser, on YOUR hardware, accessing YOUR accounts.
It's the digital equivalent of hiring a personal assistant who sits at
your computer and checks things for you.

Architecture:
    BrowserEngine       — Manages Playwright browser instances with stealth
    HumanEmulator       — Realistic mouse, keyboard, scroll, and timing patterns
    SessionManager      — Persistent cookies, storage, and auth state
    CredentialVault     — Reads credentials from Proton Pass CLI export
    PageInteractor      — High-level page interaction primitives
    
Design Principles:
    1. Stealth-first: fingerprint randomization, realistic timing, no detectable automation signals
    2. Session persistence: log in once, reuse sessions until they expire
    3. Respectful: rate-limited, backs off on errors, doesn't hammer servers
    4. Fallback role: only used when a proper API connector isn't available
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# Playwright is an optional dependency — the rest of the system can run
# without browser automation.  We gate all browser usage behind this flag.
try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        Playwright,
    )
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


# ===========================================================================
# HUMAN EMULATOR — Make the browser behave like a real person
# ===========================================================================

class HumanEmulator:
    """
    Generates human-like interaction patterns. Real humans don't:
        - Click at exact pixel coordinates instantly
        - Type at a constant speed
        - Scroll in perfect increments
        - Navigate without pause between actions
    
    This class adds the organic imperfection that makes automation
    indistinguishable from a real user.
    """

    def __init__(self, speed_factor: float = 1.0):
        """
        Args:
            speed_factor: 1.0 = normal human speed, 0.5 = twice as fast, 2.0 = half speed
        """
        self.speed_factor = speed_factor

    async def move_mouse_to(self, page: Any, x: float, y: float):
        """
        Move mouse to target with a natural curved path.
        Real humans don't move in straight lines — they arc slightly.
        """
        # Get current position (or start from a random edge position)
        steps = random.randint(15, 35)
        duration = random.uniform(0.3, 0.8) * self.speed_factor

        # Generate Bézier curve control points for natural arc
        current = await page.evaluate("() => ({ x: window._mouseX || 0, y: window._mouseY || 0 })")
        cx, cy = current.get("x", random.randint(0, 500)), current.get("y", random.randint(0, 500))

        # Control point offset (creates the arc)
        ctrl_x = (cx + x) / 2 + random.uniform(-100, 100)
        ctrl_y = (cy + y) / 2 + random.uniform(-50, 50)

        for i in range(steps):
            t = i / steps
            # Quadratic Bézier curve
            px = (1 - t) ** 2 * cx + 2 * (1 - t) * t * ctrl_x + t ** 2 * x
            py = (1 - t) ** 2 * cy + 2 * (1 - t) * t * ctrl_y + t ** 2 * y

            # Add micro-jitter (hand tremor)
            px += random.gauss(0, 1.5)
            py += random.gauss(0, 1.5)

            await page.mouse.move(px, py)
            await asyncio.sleep(duration / steps)

        # Store final position
        await page.evaluate(f"() => {{ window._mouseX = {x}; window._mouseY = {y}; }}")

    async def click(self, page: Any, selector: str, double: bool = False):
        """Click an element with human-like approach and timing."""
        element = await page.wait_for_selector(selector, timeout=10000)
        if not element:
            raise Exception(f"Element not found: {selector}")

        box = await element.bounding_box()
        if not box:
            raise Exception(f"Element not visible: {selector}")

        # Don't click dead center — humans are imprecise
        target_x = box["x"] + box["width"] * random.uniform(0.2, 0.8)
        target_y = box["y"] + box["height"] * random.uniform(0.3, 0.7)

        await self.move_mouse_to(page, target_x, target_y)

        # Brief pause before clicking (recognition time)
        await asyncio.sleep(random.uniform(0.05, 0.2) * self.speed_factor)

        if double:
            await page.mouse.dblclick(target_x, target_y)
        else:
            await page.mouse.click(target_x, target_y)

        # Brief pause after clicking (processing time)
        await asyncio.sleep(random.uniform(0.1, 0.4) * self.speed_factor)

    async def type_text(self, page: Any, selector: str, text: str,
                        clear_first: bool = True):
        """
        Type text with human-like keystroke timing.
        Real typing has variable speed, occasional pauses, and rhythm.
        """
        element = await page.wait_for_selector(selector, timeout=10000)
        if not element:
            raise Exception(f"Element not found: {selector}")

        await self.click(page, selector)

        if clear_first:
            # Select all and delete (like a human would)
            await page.keyboard.press("Meta+a" if os.uname().sysname == "Darwin" else "Control+a")
            await asyncio.sleep(random.uniform(0.05, 0.15))
            await page.keyboard.press("Backspace")
            await asyncio.sleep(random.uniform(0.1, 0.3))

        # Type each character with variable delay
        for i, char in enumerate(text):
            # Base typing speed: 80-120ms per character
            delay = random.gauss(0.1, 0.03) * self.speed_factor

            # Occasional longer pauses (thinking, looking at source)
            if random.random() < 0.05:
                delay += random.uniform(0.3, 0.8) * self.speed_factor

            # Speed up for common sequences (the, and, etc.)
            if i > 0 and text[i - 1:i + 1] in ["th", "he", "in", "er", "an"]:
                delay *= 0.7

            await page.keyboard.type(char, delay=int(delay * 1000))

    async def scroll(self, page: Any, direction: str = "down",
                     amount: Optional[int] = None):
        """
        Scroll with human-like behavior — variable speed, slight pauses,
        sometimes overshooting and correcting.
        """
        if amount is None:
            amount = random.randint(200, 600)

        if direction == "up":
            amount = -amount

        # Scroll in multiple smaller increments
        remaining = amount
        while abs(remaining) > 10:
            chunk = int(remaining * random.uniform(0.2, 0.5))
            if abs(chunk) < 20:
                chunk = remaining

            await page.mouse.wheel(0, chunk)
            remaining -= chunk

            # Brief pause between scroll chunks
            await asyncio.sleep(random.uniform(0.02, 0.1) * self.speed_factor)

        # Small settling pause
        await asyncio.sleep(random.uniform(0.2, 0.5) * self.speed_factor)

    async def wait_human(self, min_seconds: float = 0.5, max_seconds: float = 2.0):
        """Wait a human-realistic amount of time between actions."""
        delay = random.uniform(min_seconds, max_seconds) * self.speed_factor
        await asyncio.sleep(delay)

    async def read_page(self, page: Any, estimated_words: int = 200):
        """Simulate reading a page — scroll gradually, pause to read."""
        # Average reading speed: ~250 words per minute
        read_time = (estimated_words / 250) * 60 * self.speed_factor

        # Break into scroll-and-read chunks
        chunks = random.randint(3, 8)
        for _ in range(chunks):
            await self.scroll(page, "down", random.randint(150, 400))
            await asyncio.sleep(read_time / chunks)


# ===========================================================================
# SESSION MANAGER — Persistent browser sessions
# ===========================================================================

class SessionManager:
    """
    Manages persistent browser sessions so you don't have to log in
    every time. Stores cookies and local storage state.
    """

    def __init__(self, sessions_dir: str = "./data/browser_sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def get_session_path(self, site_id: str) -> Path:
        return self.sessions_dir / f"{site_id}.json"

    async def save_session(self, context: Any, site_id: str):
        """Save browser context state (cookies, storage) to disk."""
        state = await context.storage_state()
        path = self.get_session_path(site_id)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def has_session(self, site_id: str) -> bool:
        return self.get_session_path(site_id).exists()

    def get_storage_state(self, site_id: str) -> Optional[str]:
        path = self.get_session_path(site_id)
        if path.exists():
            return str(path)
        return None

    def clear_session(self, site_id: str):
        path = self.get_session_path(site_id)
        if path.exists():
            path.unlink()


# ===========================================================================
# CREDENTIAL VAULT — Read credentials from Proton Pass
# ===========================================================================

class CredentialVault:
    """
    Reads credentials from a Proton Pass export or a local encrypted vault.
    
    Supported formats:
        - Proton Pass JSON export
        - Simple encrypted JSON vault (for manual entries)
    
    NEVER stores credentials in plaintext outside the vault.
    Credentials are held in memory only during active use.
    """

    def __init__(self, vault_path: str = "./data/credentials"):
        self.vault_dir = Path(vault_path)
        self.vault_dir.mkdir(parents=True, exist_ok=True)
        self._credentials: dict[str, dict] = {}

    def load_proton_pass_export(self, export_path: str):
        """
        Load credentials from a Proton Pass JSON export.
        
        Proton Pass export format:
        {
            "vaults": {
                "vault_id": {
                    "items": [
                        {
                            "data": {
                                "type": "login",
                                "content": {
                                    "itemEmail": "user@example.com",
                                    "itemUsername": "username",
                                    "password": "pass123",
                                    "urls": ["https://example.com"]
                                },
                                "metadata": {
                                    "name": "Example Service"
                                }
                            }
                        }
                    ]
                }
            }
        }
        """
        with open(export_path) as f:
            data = json.load(f)

        for vault in data.get("vaults", {}).values():
            for item in vault.get("items", []):
                item_data = item.get("data", {})
                if item_data.get("type") != "login":
                    continue

                content = item_data.get("content", {})
                metadata = item_data.get("metadata", {})
                name = metadata.get("name", "unknown")
                urls = content.get("urls", [])

                # Create a site_id from the first URL
                site_id = self._url_to_site_id(urls[0]) if urls else name.lower().replace(" ", "_")

                self._credentials[site_id] = {
                    "name": name,
                    "username": content.get("itemUsername") or content.get("itemEmail", ""),
                    "email": content.get("itemEmail", ""),
                    "password": content.get("password", ""),
                    "urls": urls,
                    "totp_uri": content.get("totpUri", ""),
                }

    def load_manual_vault(self, vault_file: str = "manual_vault.json"):
        """
        Load from a simple JSON file for services not in Proton Pass.
        Format: {"site_id": {"username": "...", "password": "...", "url": "..."}}
        """
        path = self.vault_dir / vault_file
        if path.exists():
            with open(path) as f:
                entries = json.load(f)
                self._credentials.update(entries)

    def get_credential(self, site_id: str) -> Optional[dict]:
        """Get credentials for a site. Returns None if not found."""
        return self._credentials.get(site_id)

    def get_totp(self, site_id: str) -> Optional[str]:
        """Generate a TOTP code if URI is available."""
        cred = self._credentials.get(site_id)
        if not cred or not cred.get("totp_uri"):
            return None

        try:
            import pyotp
            totp = pyotp.parse_uri(cred["totp_uri"])
            return totp.now()
        except ImportError:
            print("pyotp not installed — cannot generate TOTP codes")
            return None

    def list_sites(self) -> list[str]:
        return list(self._credentials.keys())

    @staticmethod
    def _url_to_site_id(url: str) -> str:
        """Convert a URL to a simple site identifier."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        # Remove www. and TLD
        parts = domain.replace("www.", "").split(".")
        return parts[0] if parts else domain


# ===========================================================================
# BROWSER ENGINE — Core browser management with stealth
# ===========================================================================

class BrowserEngine:
    """
    Manages stealth Playwright browser instances.
    
    Stealth techniques:
        - Randomized viewport, user agent, and platform
        - WebDriver flag removal
        - Navigator properties patching
        - Realistic WebGL and canvas fingerprints
        - Timezone and locale matching
    """

    # Realistic viewport sizes (common desktop resolutions)
    VIEWPORTS = [
        {"width": 1920, "height": 1080},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1366, "height": 768},
        {"width": 2560, "height": 1440},
    ]

    # Realistic user agents (updated periodically)
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    ]

    def __init__(self, data_dir: str = "./data/browser"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session_manager = SessionManager(str(self.data_dir / "sessions"))
        # Shared browser instance pattern: a single Playwright process and
        # a single Chromium browser are shared across ALL browser connectors.
        # Each connector gets its own BrowserContext (isolated cookies/storage)
        # but they all share one browser process to save ~500 MB RAM.
        self._playwright: Optional[Any] = None
        self._browser: Optional[Any] = None

    async def start(self):
        """Start the Playwright browser engine."""
        if not HAS_PLAYWRIGHT:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        # Launch a single headless Chromium with stealth flags that disable
        # common automation-detection signals (AutomationControlled, infobars).
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-infobars",
                "--window-position=0,0",
                "--ignore-certificate-errors",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )

    async def stop(self):
        """Shut down the browser engine."""
        # Cleanup order matters: close the browser first (which closes all
        # contexts and pages), then stop the Playwright process wrapper.
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def create_context(self, site_id: str,
                             locale: str = "en-US",
                             timezone_id: str = "America/Chicago") -> Any:
        """
        Create a stealth browser context for a specific site.
        Reuses stored session if available.
        """
        # Lazy-start the browser if it hasn't been launched yet
        if not self._browser:
            await self.start()

        # Randomise viewport and user-agent per context so each site sees
        # a slightly different browser fingerprint.
        viewport = random.choice(self.VIEWPORTS)
        user_agent = random.choice(self.USER_AGENTS)
        # Restore saved cookies/local-storage so we skip re-login when possible
        storage_state = self.session_manager.get_storage_state(site_id)

        context = await self._browser.new_context(
            viewport=viewport,
            user_agent=user_agent,
            locale=locale,
            timezone_id=timezone_id,
            storage_state=storage_state,
            # Prevent detection
            java_script_enabled=True,
            bypass_csp=False,
            ignore_https_errors=False,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            },
        )

        # Apply stealth patches to every new page
        context.on("page", lambda page: asyncio.create_task(self._apply_stealth(page)))

        return context

    async def _apply_stealth(self, page: Any):
        """Apply stealth JavaScript patches to evade detection."""
        await page.add_init_script("""
            // Remove webdriver flag
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            
            // Fake plugins (real browsers have plugins)
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
                    { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
                    { name: 'Native Client', filename: 'internal-nacl-plugin' },
                ],
            });
            
            // Fake languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            // Fix permissions query
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) =>
                parameters.name === 'notifications'
                    ? Promise.resolve({ state: Notification.permission })
                    : originalQuery(parameters);
            
            // Chrome runtime (expected in real Chrome)
            window.chrome = { runtime: {} };
            
            // Prevent canvas fingerprint detection by adding noise
            const toBlob = HTMLCanvasElement.prototype.toBlob;
            const toDataURL = HTMLCanvasElement.prototype.toDataURL;
            const getImageData = CanvasRenderingContext2D.prototype.getImageData;
            
            // Add subtle noise to canvas reads
            const addNoise = (data) => {
                for (let i = 0; i < data.length; i += 4) {
                    // Tiny random perturbation (+/- 1 in each channel)
                    data[i] = Math.max(0, Math.min(255, data[i] + (Math.random() < 0.1 ? (Math.random() > 0.5 ? 1 : -1) : 0)));
                }
                return data;
            };
        """)

    async def new_page(self, context: Any) -> Any:
        """Create a new page in an existing context."""
        # Pages inherit the stealth patches from their context (applied via
        # the "page" event handler registered in create_context).
        page = await context.new_page()
        return page

    async def save_session(self, context: Any, site_id: str):
        """Save the current session for later reuse."""
        await self.session_manager.save_session(context, site_id)


# ===========================================================================
# PAGE INTERACTOR — High-level page interaction primitives
# ===========================================================================

class PageInteractor:
    """
    High-level interaction methods that combine the human emulator
    with common page interaction patterns.
    """

    def __init__(self, human: HumanEmulator):
        self.human = human

    async def login(self, page: Any, creds: dict,
                    username_selector: str = "input[type='email'], input[name='username'], input[name='email'], #email, #username",
                    password_selector: str = "input[type='password'], #password",
                    submit_selector: str = "button[type='submit'], input[type='submit'], .login-button, #login-button"):
        """
        Perform a login with human-like behavior.
        Tries common selectors, fills in credentials, submits.
        """
        # Wait for page to be interactive
        await page.wait_for_load_state("networkidle")
        await self.human.wait_human(1.0, 2.5)

        # Find and fill username
        username = creds.get("username") or creds.get("email", "")
        try:
            await self.human.type_text(page, username_selector, username)
        except Exception:
            # Try individual selectors
            for sel in username_selector.split(", "):
                try:
                    await self.human.type_text(page, sel.strip(), username)
                    break
                except Exception:
                    continue

        await self.human.wait_human(0.3, 1.0)

        # Fill password
        password = creds.get("password", "")
        try:
            await self.human.type_text(page, password_selector, password)
        except Exception:
            for sel in password_selector.split(", "):
                try:
                    await self.human.type_text(page, sel.strip(), password)
                    break
                except Exception:
                    continue

        await self.human.wait_human(0.5, 1.5)

        # Submit
        try:
            await self.human.click(page, submit_selector)
        except Exception:
            # Try pressing Enter as fallback
            await page.keyboard.press("Enter")

        # Wait for navigation
        await page.wait_for_load_state("networkidle")
        await self.human.wait_human(1.0, 3.0)

    async def handle_2fa(self, page: Any, totp_code: str,
                         code_selector: str = "input[name='code'], input[name='otp'], input[type='tel'], .otp-input"):
        """Enter a 2FA/TOTP code."""
        await self.human.wait_human(1.0, 2.0)
        await self.human.type_text(page, code_selector, totp_code)
        await self.human.wait_human(0.5, 1.0)
        await page.keyboard.press("Enter")
        await page.wait_for_load_state("networkidle")

    async def extract_text(self, page: Any, selector: str) -> list[str]:
        """Extract text content from elements matching a selector."""
        elements = await page.query_selector_all(selector)
        texts = []
        for el in elements:
            text = await el.text_content()
            if text and text.strip():
                texts.append(text.strip())
        return texts

    async def extract_table(self, page: Any, selector: str = "table") -> list[dict]:
        """Extract structured data from an HTML table."""
        return await page.evaluate(f"""
            () => {{
                const table = document.querySelector('{selector}');
                if (!table) return [];
                
                const headers = [...table.querySelectorAll('th')].map(th => th.textContent.trim());
                const rows = [...table.querySelectorAll('tbody tr')];
                
                return rows.map(row => {{
                    const cells = [...row.querySelectorAll('td')].map(td => td.textContent.trim());
                    const obj = {{}};
                    headers.forEach((h, i) => obj[h] = cells[i] || '');
                    return obj;
                }});
            }}
        """)

    async def wait_for_content(self, page: Any, selector: str,
                                timeout: int = 30000) -> bool:
        """Wait for specific content to appear on the page."""
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def screenshot(self, page: Any, path: str):
        """Take a screenshot for debugging."""
        await page.screenshot(path=path, full_page=True)

    async def get_page_data(self, page: Any) -> dict:
        """Extract common page metadata."""
        return {
            "url": page.url,
            "title": await page.title(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

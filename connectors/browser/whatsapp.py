"""
Life OS — WhatsApp Web Browser Connector

WhatsApp has no public API for personal accounts. The only way to access
your own messages programmatically is through WhatsApp Web.

This connector:
    1. Opens web.whatsapp.com (requires initial QR code scan)
    2. Monitors for new messages
    3. Extracts conversation history
    4. Can send messages on your behalf

First run requires you to scan the QR code with your phone.
After that, the session persists for weeks/months.

Configuration:
    connectors:
      whatsapp:
        mode: "browser"
        sync_interval: 10
        priority_contacts: ["Mom", "Partner"]
        max_conversations_per_sync: 10
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from connectors.browser.base_connector import BrowserBaseConnector
from connectors.browser.engine import HumanEmulator, PageInteractor

logger = logging.getLogger(__name__)


class WhatsAppConnector(BrowserBaseConnector):
    """
    Browser-only connector for WhatsApp Web. WhatsApp has no public API for
    personal accounts, so all access goes through the web UI. First run
    requires a QR code scan; after that the session persists for weeks.
    """

    CONNECTOR_ID = "whatsapp"
    DISPLAY_NAME = "WhatsApp"
    SITE_ID = "whatsapp"
    LOGIN_URL = "https://web.whatsapp.com"
    SYNC_INTERVAL_SECONDS = 10     # Poll frequently for near-real-time messages
    MIN_REQUEST_INTERVAL = 1.0     # WhatsApp Web is a SPA, so page loads are rare

    async def api_authenticate(self) -> bool:
        # WhatsApp has no personal API; always fall through to browser mode
        return False

    async def api_sync(self) -> int:
        raise NotImplementedError("WhatsApp has no personal API")

    def get_login_selectors(self) -> dict[str, str]:
        # WhatsApp Web uses QR code authentication, not username/password,
        # so we return an empty dict to skip the standard login form flow.
        return {}

    async def is_logged_in(self, page: Any) -> bool:
        """Check if WhatsApp Web is connected."""
        try:
            # If we can see the chat list, we're logged in
            chat_list = await page.query_selector('[aria-label="Chat list"], #pane-side')
            return chat_list is not None
        except Exception:
            return False

    async def authenticate(self) -> bool:
        """
        WhatsApp Web authentication requires QR code scan.
        We open the page and wait for the user to scan.
        """
        try:
            await self._browser_engine.start()
            self._context = await self._browser_engine.create_context(self.SITE_ID)
            self._page = await self._browser_engine.new_page(self._context)

            # Try to reuse an existing session (saved cookies from a previous run)
            await self._page.goto(self.LOGIN_URL, wait_until="networkidle")
            await self._human.wait_human(3.0, 5.0)

            if await self.is_logged_in(self._page):
                logger.info("[%s] Existing session valid", self.CONNECTOR_ID)
                return True

            # No valid session -- the user must scan the QR code with their phone.
            # We screenshot it to a known path so it can be viewed externally.
            logger.info(
                "[%s] Waiting for QR code scan — check screenshot at data/browser/whatsapp_qr.png",
                self.CONNECTOR_ID,
            )

            await self._interactor.screenshot(
                self._page, "data/browser/whatsapp_qr.png"
            )

            # Poll every ~5 seconds for up to 2 minutes (24 iterations)
            for _ in range(24):
                await self._human.wait_human(5.0, 5.0)
                if await self.is_logged_in(self._page):
                    logger.info("[%s] QR code scanned, connected!", self.CONNECTOR_ID)
                    # Save the authenticated session for future reuse
                    await self._browser_engine.save_session(self._context, self.SITE_ID)
                    return True

            logger.warning("[%s] QR code scan timed out", self.CONNECTOR_ID)
            return False

        except Exception as e:
            logger.error("[%s] Auth error: %s", self.CONNECTOR_ID, e)
            return False

    async def browser_sync(self, page: Any, human: HumanEmulator,
                           interactor: PageInteractor) -> int:
        """Scrape new messages from WhatsApp Web.

        Reads the sidebar for chats with unread badges, clicks into each
        conversation, and extracts the recent messages.
        """
        count = 0
        max_convos = self.config.get("max_conversations_per_sync", 10)

        # Step 1: Scan the sidebar for chats that have unread message badges
        unread_chats = await self._get_unread_chats(page)

        for chat in unread_chats[:max_convos]:
            try:
                # Step 2: Click the conversation to open it in the main panel
                await human.click(page, f'span[title="{chat["name"]}"]')
                await human.wait_human(0.5, 1.5)

                # Step 3: Extract the last 20 messages from the open conversation
                messages = await self._extract_messages(page)

                for msg in messages:
                    if msg.get("is_new"):
                        payload = {
                            "message_id": msg.get("id", ""),
                            "channel": "whatsapp",
                            "direction": "inbound",
                            "from_contact": chat["name"],
                            "body": msg["text"],
                            "body_plain": msg["text"],
                            "snippet": msg["text"][:150],
                            "timestamp": msg.get("time", ""),
                            "is_group": chat.get("is_group", False),
                            "group_name": chat["name"] if chat.get("is_group") else None,
                        }

                        await self.publish_event(
                            "message.received", payload,
                            priority=self._classify_priority(chat["name"]),
                            metadata={"related_contacts": [chat["name"]]},
                        )
                        count += 1

            except Exception as e:
                logger.warning("[%s] Error reading chat '%s': %s", self.CONNECTOR_ID, chat.get("name"), e)

        return count

    async def _get_unread_chats(self, page: Any) -> list[dict]:
        """Extract list of chats with unread messages."""
        return await page.evaluate("""
            () => {
                const chats = [];
                const chatElements = document.querySelectorAll('[aria-label="Chat list"] > div > div');
                
                for (const el of chatElements) {
                    const nameEl = el.querySelector('span[title]');
                    const unreadEl = el.querySelector('span[aria-label*="unread"]');
                    
                    if (nameEl && unreadEl) {
                        const name = nameEl.getAttribute('title');
                        const unreadText = unreadEl.getAttribute('aria-label') || '';
                        const unreadMatch = unreadText.match(/(\\d+)/);
                        const unreadCount = unreadMatch ? parseInt(unreadMatch[1]) : 0;
                        
                        if (unreadCount > 0) {
                            chats.push({
                                name: name,
                                unread_count: unreadCount,
                                is_group: false,  // Would need more logic to detect
                            });
                        }
                    }
                }
                return chats;
            }
        """)

    async def _extract_messages(self, page: Any) -> list[dict]:
        """Extract messages from the currently open conversation."""
        return await page.evaluate("""
            () => {
                const messages = [];
                const msgElements = document.querySelectorAll('.message-in, .message-out');
                
                // Get last 20 messages
                const recent = [...msgElements].slice(-20);
                
                for (const el of recent) {
                    const textEl = el.querySelector('.selectable-text');
                    const timeEl = el.querySelector('[data-pre-plain-text]');
                    
                    if (textEl) {
                        const text = textEl.textContent || '';
                        const prePlain = timeEl?.getAttribute('data-pre-plain-text') || '';
                        
                        messages.push({
                            text: text.trim(),
                            time: prePlain,
                            is_incoming: el.classList.contains('message-in'),
                            is_new: true,  // Simplified; would track seen message IDs
                        });
                    }
                }
                return messages;
            }
        """)

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a message through WhatsApp Web.

        Uses the search box to find the recipient, clicks into the chat,
        types the message with human-like keystrokes, and presses Enter.
        """
        if action == "send_message" and self._page:
            recipient = params["to"]
            message = params["message"]

            # Use WhatsApp's search to find the contact by name
            search_box = await self._page.wait_for_selector(
                '[aria-label="Search input textbox"], [title="Search input textbox"]'
            )
            if search_box:
                await self._human.type_text(
                    self._page,
                    '[aria-label="Search input textbox"], [title="Search input textbox"]',
                    recipient,
                )
                await self._human.wait_human(1.0, 2.0)

                # Click the contact result
                await self._human.click(
                    self._page,
                    f'span[title="{recipient}"]',
                )
                await self._human.wait_human(0.5, 1.0)

            # Type and send the message
            msg_box_selector = '[aria-label="Type a message"], [title="Type a message"], footer [contenteditable="true"]'
            await self._human.type_text(self._page, msg_box_selector, message)
            await self._human.wait_human(0.3, 0.8)
            await self._page.keyboard.press("Enter")

            await self.publish_event(
                "message.sent",
                {
                    "channel": "whatsapp",
                    "direction": "outbound",
                    "to_contact": recipient,
                    "body": message,
                    "body_plain": message,
                    "snippet": message[:150],
                },
            )

            return {"status": "sent", "to": recipient}

        raise ValueError(f"Unknown action: {action}")

    def _classify_priority(self, contact_name: str) -> str:
        """Assign "high" priority to contacts in the priority_contacts list,
        so the AI agent surfaces their messages immediately."""
        priority_contacts = self.config.get("priority_contacts", [])
        if contact_name in priority_contacts:
            return "high"
        return "normal"

    async def health_check(self) -> dict[str, Any]:
        if self._page:
            logged_in = await self.is_logged_in(self._page)
            return {"status": "ok" if logged_in else "session_expired",
                    "connector": self.CONNECTOR_ID, "mode": "browser"}
        return {"status": "not_started", "connector": self.CONNECTOR_ID}

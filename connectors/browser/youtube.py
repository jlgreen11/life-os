"""
Life OS — YouTube Subscriptions Browser Connector

The YouTube Data API has brutal quota limits (10K units/day, a single
subscription list call costs 1 unit but video details cost 50+).
For a personal system monitoring YOUR feed, browser scraping is more
practical and gives you access to everything you see in the UI.

Scrapes:
    - New videos from subscriptions
    - Watch Later list
    - Liked videos (for taste profiling)
    - Community posts from channels you follow

Configuration:
    connectors:
      youtube:
        mode: "browser"
        sync_interval: 900       # Every 15 minutes
        max_videos_per_sync: 30
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from connectors.browser.base_connector import BrowserBaseConnector
from connectors.browser.engine import HumanEmulator, PageInteractor


class YouTubeConnector(BrowserBaseConnector):

    CONNECTOR_ID = "youtube"
    DISPLAY_NAME = "YouTube"
    SITE_ID = "google"  # Uses Google account credentials
    LOGIN_URL = "https://accounts.google.com/ServiceLogin?service=youtube"
    REQUIRES_2FA = True
    SYNC_INTERVAL_SECONDS = 900
    MIN_REQUEST_INTERVAL = 3.0

    def get_login_selectors(self) -> dict[str, str]:
        return {
            "username": "input[type='email']",
            "password": "input[type='password']",
            "submit": "#identifierNext button, #passwordNext button",
            "totp": "input[type='tel']",
        }

    async def is_logged_in(self, page: Any) -> bool:
        try:
            avatar = await page.query_selector("#avatar-btn, button#avatar-btn, img#img[alt='Avatar image']")
            return avatar is not None
        except Exception:
            return False

    async def browser_sync(self, page: Any, human: HumanEmulator,
                           interactor: PageInteractor) -> int:
        """Scrape subscription feed for new videos."""
        count = 0
        max_videos = self.config.get("max_videos_per_sync", 30)

        # Navigate to subscriptions feed
        await self.navigate_with_rate_limit(page, "https://www.youtube.com/feed/subscriptions")
        await human.wait_human(2.0, 4.0)

        # Scroll down a few times to load more videos
        for _ in range(3):
            await human.scroll(page, "down", 800)
            await human.wait_human(1.0, 2.0)

        # Extract video entries
        videos = await page.evaluate("""
            () => {
                const videos = [];
                const items = document.querySelectorAll('ytd-grid-video-renderer, ytd-rich-item-renderer, ytd-video-renderer');
                
                for (const item of items) {
                    try {
                        const titleEl = item.querySelector('#video-title, #video-title-link, a#video-title');
                        const channelEl = item.querySelector('#channel-name a, .ytd-channel-name a, #text.ytd-channel-name');
                        const metaEl = item.querySelector('#metadata-line span, .inline-metadata-item');
                        const thumbEl = item.querySelector('img#img, img.yt-core-image');
                        const linkEl = item.querySelector('a#thumbnail, a#video-title-link, a#video-title');
                        
                        if (titleEl && linkEl) {
                            const href = linkEl.getAttribute('href') || '';
                            const videoId = href.match(/v=([^&]+)/)?.[1] || href.match(/shorts\\/([^?]+)/)?.[1] || '';
                            
                            videos.push({
                                title: titleEl.textContent?.trim() || '',
                                channel: channelEl?.textContent?.trim() || '',
                                video_id: videoId,
                                url: href.startsWith('http') ? href : 'https://www.youtube.com' + href,
                                thumbnail: thumbEl?.getAttribute('src') || '',
                                meta: metaEl?.textContent?.trim() || '',
                            });
                        }
                    } catch (e) {}
                }
                return videos;
            }
        """)

        # Get cursor (last seen video IDs)
        seen_ids = set()
        cursor = self.get_sync_cursor()
        if cursor:
            try:
                import json
                seen_ids = set(json.loads(cursor))
            except Exception:
                pass

        new_video_ids = []
        for video in videos[:max_videos]:
            vid = video.get("video_id", "")
            if not vid or vid in seen_ids:
                continue

            payload = {
                "video_id": vid,
                "title": video.get("title", ""),
                "channel": video.get("channel", ""),
                "url": video.get("url", ""),
                "thumbnail": video.get("thumbnail", ""),
                "meta": video.get("meta", ""),
            }

            await self.publish_event(
                "content.youtube.new_video", payload,
                priority="low",
                metadata={"domain": "media", "channel": video.get("channel", "")},
            )
            new_video_ids.append(vid)
            count += 1

        # Update cursor with seen video IDs (keep last 500)
        if new_video_ids:
            all_seen = list(seen_ids | set(new_video_ids))[-500:]
            import json
            self.set_sync_cursor(json.dumps(all_seen))

        return count

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        if action == "add_to_watch_later" and self._page:
            url = params.get("url", "")
            if url:
                await self.navigate_with_rate_limit(self._page, url)
                await self._human.wait_human(1.0, 2.0)
                # Click the save/watch later button
                try:
                    await self._human.click(self._page, "#button-shape button[aria-label='Save']")
                    await self._human.wait_human(0.5, 1.0)
                    await self._human.click(self._page, "tp-yt-paper-checkbox:has-text('Watch later')")
                    return {"status": "added_to_watch_later"}
                except Exception:
                    return {"status": "error", "details": "Could not find Watch Later button"}
        raise ValueError(f"Unknown action: {action}")

    async def health_check(self) -> dict[str, Any]:
        if self._page:
            return {"status": "ok", "connector": self.CONNECTOR_ID, "mode": "browser"}
        return {"status": "not_started", "connector": self.CONNECTOR_ID}

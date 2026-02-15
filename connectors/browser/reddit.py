"""
Life OS — Reddit Browser Connector

Reddit killed third-party apps by making their API absurdly expensive.
But old.reddit.com is lightweight and very scrapable. This connector
monitors your subscribed subreddits and extracts content you'd care about.

Uses old.reddit.com because:
    - Simpler DOM structure (easier to parse)
    - Faster page loads (less JavaScript)
    - More content per page
    - Less aggressive anti-bot measures

Scrapes:
    - Front page (personalized feed from your subscriptions)
    - Specific subreddits you flag as high-priority
    - Your inbox/messages
    - Saved posts

Configuration:
    connectors:
      reddit:
        mode: "browser"
        sync_interval: 600
        use_old_reddit: true
        priority_subreddits: ["homelab", "selfhosted", "privacy"]
        max_posts_per_sync: 25
"""

from __future__ import annotations

from typing import Any

from connectors.browser.base_connector import BrowserBaseConnector
from connectors.browser.engine import HumanEmulator, PageInteractor


class RedditConnector(BrowserBaseConnector):

    CONNECTOR_ID = "reddit"
    DISPLAY_NAME = "Reddit"
    SITE_ID = "reddit"
    LOGIN_URL = "https://old.reddit.com/login"
    SYNC_INTERVAL_SECONDS = 600
    MIN_REQUEST_INTERVAL = 3.0
    MAX_PAGES_PER_SYNC = 5

    def get_login_selectors(self) -> dict[str, str]:
        return {
            "username": "#user_login input[name='user']",
            "password": "#user_login input[name='passwd']",
            "submit": "#user_login button[type='submit']",
        }

    async def is_logged_in(self, page: Any) -> bool:
        try:
            # old.reddit.com shows username in the header when logged in
            user_link = await page.query_selector("span.user a.login-required, .user a[href*='/user/']")
            return user_link is not None
        except Exception:
            return False

    async def browser_sync(self, page: Any, human: HumanEmulator,
                           interactor: PageInteractor) -> int:
        count = 0
        max_posts = self.config.get("max_posts_per_sync", 25)

        # 1. Scrape the front page (personalized feed)
        await self.navigate_with_rate_limit(page, "https://old.reddit.com/")
        await human.wait_human(1.5, 3.0)

        posts = await self._extract_posts(page)

        # Filter to new posts only (using cursor)
        seen_ids = self._get_seen_ids()
        new_posts = [p for p in posts if p.get("id") not in seen_ids]

        for post in new_posts[:max_posts]:
            payload = {
                "post_id": post["id"],
                "title": post["title"],
                "subreddit": post["subreddit"],
                "author": post["author"],
                "score": post.get("score", 0),
                "url": post.get("url", ""),
                "permalink": post.get("permalink", ""),
                "comment_count": post.get("comments", 0),
                "is_self": post.get("is_self", False),
            }

            # Determine if this is from a priority subreddit
            priority_subs = self.config.get("priority_subreddits", [])
            is_priority = post["subreddit"].lower() in [s.lower() for s in priority_subs]

            await self.publish_event(
                "content.reddit.new_post", payload,
                priority="normal" if is_priority else "low",
                metadata={
                    "domain": "media",
                    "subreddit": post["subreddit"],
                    "topics": self._extract_topics(post["title"]),
                },
            )
            count += 1

        # Update seen IDs
        if new_posts:
            new_ids = [p["id"] for p in new_posts]
            self._update_seen_ids(new_ids)

        # 2. Check inbox for messages/replies
        await self.rate_limit_wait()
        await self.navigate_with_rate_limit(page, "https://old.reddit.com/message/unread/")
        await human.wait_human(1.0, 2.0)

        messages = await self._extract_messages(page)
        for msg in messages:
            await self.publish_event(
                "message.received",
                {
                    "channel": "reddit",
                    "direction": "inbound",
                    "from_contact": msg.get("author", ""),
                    "subject": msg.get("subject", ""),
                    "body": msg.get("body", ""),
                    "body_plain": msg.get("body", ""),
                    "snippet": msg.get("body", "")[:150],
                    "context_url": msg.get("context", ""),
                },
                priority="normal",
            )
            count += 1

        return count

    async def _extract_posts(self, page: Any) -> list[dict]:
        """Extract posts from the current old.reddit.com page."""
        return await page.evaluate("""
            () => {
                const posts = [];
                const things = document.querySelectorAll('.thing.link');
                
                for (const thing of things) {
                    try {
                        const id = thing.getAttribute('data-fullname') || '';
                        const titleEl = thing.querySelector('a.title');
                        const subEl = thing.querySelector('.subreddit');
                        const authorEl = thing.querySelector('.author');
                        const scoreEl = thing.querySelector('.score.unvoted');
                        const commentsEl = thing.querySelector('.comments');
                        const domain = thing.getAttribute('data-domain') || '';
                        
                        if (titleEl) {
                            posts.push({
                                id: id,
                                title: titleEl.textContent?.trim() || '',
                                url: titleEl.getAttribute('href') || '',
                                subreddit: subEl?.textContent?.trim().replace('/r/', '').replace('r/', '') || '',
                                author: authorEl?.textContent?.trim() || '',
                                score: parseInt(scoreEl?.getAttribute('title') || '0') || 0,
                                permalink: thing.getAttribute('data-permalink') || '',
                                comments: parseInt(commentsEl?.textContent?.match(/(\\d+)/)?.[1] || '0') || 0,
                                is_self: domain.startsWith('self.'),
                            });
                        }
                    } catch (e) {}
                }
                return posts;
            }
        """)

    async def _extract_messages(self, page: Any) -> list[dict]:
        """Extract unread messages from inbox."""
        return await page.evaluate("""
            () => {
                const messages = [];
                const things = document.querySelectorAll('.thing.message.unread');
                
                for (const thing of things) {
                    try {
                        const authorEl = thing.querySelector('.author');
                        const subjectEl = thing.querySelector('.subject a');
                        const bodyEl = thing.querySelector('.md');
                        const contextEl = thing.querySelector('a.bylink');
                        
                        messages.push({
                            id: thing.getAttribute('data-fullname') || '',
                            author: authorEl?.textContent?.trim() || '',
                            subject: subjectEl?.textContent?.trim() || '',
                            body: bodyEl?.textContent?.trim() || '',
                            context: contextEl?.getAttribute('href') || '',
                        });
                    } catch (e) {}
                }
                return messages;
            }
        """)

    def _get_seen_ids(self) -> set:
        cursor = self.get_sync_cursor()
        if cursor:
            try:
                import json
                return set(json.loads(cursor))
            except Exception:
                pass
        return set()

    def _update_seen_ids(self, new_ids: list[str]):
        import json
        seen = self._get_seen_ids()
        seen.update(new_ids)
        # Keep last 1000
        self.set_sync_cursor(json.dumps(list(seen)[-1000:]))

    def _extract_topics(self, title: str) -> list[str]:
        """Quick keyword extraction from post title."""
        stop_words = {"the", "a", "an", "is", "it", "to", "in", "for", "of", "and", "or", "on", "at", "i", "my", "me"}
        words = title.lower().split()
        return [w for w in words if len(w) > 3 and w not in stop_words][:5]

    async def execute(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("Reddit connector is read-only")

    async def health_check(self) -> dict[str, Any]:
        if self._page:
            logged_in = await self.is_logged_in(self._page)
            return {"status": "ok" if logged_in else "session_expired",
                    "connector": self.CONNECTOR_ID, "mode": "browser"}
        return {"status": "not_started", "connector": self.CONNECTOR_ID}

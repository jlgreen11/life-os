"""
Life OS — Connector Registry

Central registry of all connector types with their configuration schemas.
Used by the admin UI to generate config forms and by the runtime to
instantiate connectors dynamically.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ConnectorFieldDef:
    """Definition of a single configuration field for a connector."""
    name: str
    field_type: str = "string"  # string, password, integer, list, boolean
    required: bool = False
    sensitive: bool = False
    default: Any = None
    help_text: str = ""
    placeholder: str = ""


@dataclass
class ConnectorTypeDef:
    """Definition of a connector type with its config schema."""
    connector_id: str
    display_name: str
    description: str
    category: str  # "api" or "browser"
    module_path: str
    class_name: str
    config_fields: list[ConnectorFieldDef] = field(default_factory=list)


CONNECTOR_REGISTRY: dict[str, ConnectorTypeDef] = {

    # --- API Connectors ---

    "proton_mail": ConnectorTypeDef(
        connector_id="proton_mail",
        display_name="Proton Mail",
        description="Email sync via Proton Bridge (IMAP/SMTP)",
        category="api",
        module_path="connectors.proton_mail.connector",
        class_name="ProtonMailConnector",
        config_fields=[
            ConnectorFieldDef("imap_host", "string", required=True, default="127.0.0.1",
                              help_text="Proton Bridge IMAP host", placeholder="127.0.0.1"),
            ConnectorFieldDef("imap_port", "integer", required=True, default=1143,
                              help_text="Proton Bridge IMAP port", placeholder="1143"),
            ConnectorFieldDef("smtp_host", "string", default="127.0.0.1",
                              help_text="Proton Bridge SMTP host", placeholder="127.0.0.1"),
            ConnectorFieldDef("smtp_port", "integer", default=1025,
                              help_text="Proton Bridge SMTP port", placeholder="1025"),
            ConnectorFieldDef("username", "string", required=True,
                              help_text="Proton Mail address", placeholder="you@proton.me"),
            ConnectorFieldDef("password", "password", required=True, sensitive=True,
                              help_text="Proton Bridge password"),
            ConnectorFieldDef("sync_interval", "integer", default=30,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("folders", "list", default=["INBOX", "Sent"],
                              help_text="IMAP folders to sync (comma-separated)",
                              placeholder="INBOX, Sent"),
        ],
    ),

    "signal": ConnectorTypeDef(
        connector_id="signal",
        display_name="Signal Messenger",
        description="Messaging via signal-cli daemon",
        category="api",
        module_path="connectors.signal_msg.connector",
        class_name="SignalConnector",
        config_fields=[
            ConnectorFieldDef("socket_path", "string", required=True,
                              default="/tmp/signal-cli.sock",
                              help_text="Path to signal-cli socket",
                              placeholder="/tmp/signal-cli.sock"),
            ConnectorFieldDef("phone_number", "string", required=True,
                              help_text="Registered phone number", placeholder="+1XXXXXXXXXX"),
            ConnectorFieldDef("sync_interval", "integer", default=5,
                              help_text="Seconds between syncs"),
        ],
    ),

    "imessage": ConnectorTypeDef(
        connector_id="imessage",
        display_name="iMessage",
        description="macOS iMessage and SMS via chat.db and AppleScript",
        category="api",
        module_path="connectors.imessage.connector",
        class_name="iMessageConnector",
        config_fields=[
            ConnectorFieldDef("db_path", "string",
                              default="~/Library/Messages/chat.db",
                              help_text="Path to Messages database",
                              placeholder="~/Library/Messages/chat.db"),
            ConnectorFieldDef("sync_interval", "integer", default=5,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("include_sms", "boolean", default=True,
                              help_text="Include SMS/MMS messages (not just iMessage)"),
        ],
    ),

    "caldav": ConnectorTypeDef(
        connector_id="caldav",
        display_name="Calendar (CalDAV)",
        description="Calendar sync via CalDAV protocol",
        category="api",
        module_path="connectors.caldav.connector",
        class_name="CalDAVConnector",
        config_fields=[
            ConnectorFieldDef("url", "string", required=True,
                              help_text="CalDAV server URL",
                              placeholder="https://calendar.proton.me/api/calendars"),
            ConnectorFieldDef("username", "string", required=True,
                              help_text="CalDAV username", placeholder="you@proton.me"),
            ConnectorFieldDef("password", "password", required=True, sensitive=True,
                              help_text="CalDAV password"),
            ConnectorFieldDef("sync_interval", "integer", default=60,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("calendars", "list", default=["Personal"],
                              help_text="Calendar names to sync (comma-separated)",
                              placeholder="Personal, Work"),
        ],
    ),

    "finance": ConnectorTypeDef(
        connector_id="finance",
        display_name="Finance (Plaid)",
        description="Bank & transaction sync via Plaid API",
        category="api",
        module_path="connectors.finance.connector",
        class_name="FinanceConnector",
        config_fields=[
            ConnectorFieldDef("provider", "string", default="plaid",
                              help_text="Finance provider"),
            ConnectorFieldDef("client_id", "string", required=True, sensitive=True,
                              help_text="Plaid client ID"),
            ConnectorFieldDef("secret", "password", required=True, sensitive=True,
                              help_text="Plaid secret key"),
            ConnectorFieldDef("access_tokens", "list", required=True, sensitive=True,
                              help_text="Plaid access tokens (comma-separated)"),
            ConnectorFieldDef("sync_interval", "integer", default=3600,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("large_transaction_threshold", "integer", default=500,
                              help_text="Dollar amount to flag as large"),
        ],
    ),

    "home_assistant": ConnectorTypeDef(
        connector_id="home_assistant",
        display_name="Home Assistant",
        description="Home automation integration",
        category="api",
        module_path="connectors.home_assistant.connector",
        class_name="HomeAssistantConnector",
        config_fields=[
            ConnectorFieldDef("url", "string", required=True,
                              help_text="Home Assistant URL",
                              placeholder="http://homeassistant.local:8123"),
            ConnectorFieldDef("token", "password", required=True, sensitive=True,
                              help_text="Long-lived access token"),
            ConnectorFieldDef("sync_interval", "integer", default=30,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("watched_entities", "list",
                              help_text="Entity IDs to watch (comma-separated)",
                              placeholder="person.jay, sensor.temperature"),
        ],
    ),

    "google": ConnectorTypeDef(
        connector_id="google",
        display_name="Google (Gmail, Calendar, Contacts)",
        description="Gmail, Google Calendar, and Google Contacts via OAuth2",
        category="api",
        module_path="connectors.google.connector",
        class_name="GoogleConnector",
        config_fields=[
            ConnectorFieldDef("email_address", "string", required=True,
                              help_text="Google account email", placeholder="you@gmail.com"),
            ConnectorFieldDef("credentials_file", "string", required=True,
                              default="data/google_credentials.json",
                              help_text="Path to OAuth credentials.json from Google Cloud Console"),
            ConnectorFieldDef("token_file", "string",
                              default="data/google_token.json",
                              help_text="Path to store OAuth token (auto-generated)"),
            ConnectorFieldDef("sync_interval", "integer", default=30,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("calendars", "list", default=["primary"],
                              help_text="Calendar IDs to sync (comma-separated)",
                              placeholder="primary, work@group.calendar.google.com"),
            ConnectorFieldDef("gmail_labels", "list", default=["INBOX", "SENT"],
                              help_text="Gmail labels to sync (comma-separated)",
                              placeholder="INBOX, SENT"),
        ],
    ),

    # --- Browser Connectors ---

    "whatsapp": ConnectorTypeDef(
        connector_id="whatsapp",
        display_name="WhatsApp",
        description="Messaging via browser automation (scan QR on first run)",
        category="browser",
        module_path="connectors.browser.whatsapp",
        class_name="WhatsAppConnector",
        config_fields=[
            ConnectorFieldDef("sync_interval", "integer", default=10,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("priority_contacts", "list",
                              help_text="Priority contact names (comma-separated)",
                              placeholder="Mom, Partner"),
            ConnectorFieldDef("max_conversations_per_sync", "integer", default=10,
                              help_text="Max conversations to check per sync"),
        ],
    ),

    "youtube": ConnectorTypeDef(
        connector_id="youtube",
        display_name="YouTube",
        description="Subscription feed via browser automation",
        category="browser",
        module_path="connectors.browser.youtube",
        class_name="YouTubeConnector",
        config_fields=[
            ConnectorFieldDef("sync_interval", "integer", default=900,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("max_videos_per_sync", "integer", default=30,
                              help_text="Max videos to fetch per sync"),
        ],
    ),

    "reddit": ConnectorTypeDef(
        connector_id="reddit",
        display_name="Reddit",
        description="Subreddit feeds via browser automation",
        category="browser",
        module_path="connectors.browser.reddit",
        class_name="RedditConnector",
        config_fields=[
            ConnectorFieldDef("sync_interval", "integer", default=600,
                              help_text="Seconds between syncs"),
            ConnectorFieldDef("use_old_reddit", "boolean", default=True,
                              help_text="Use old.reddit.com (more reliable)"),
            ConnectorFieldDef("priority_subreddits", "list",
                              help_text="Subreddits to watch (comma-separated)",
                              placeholder="homelab, selfhosted, privacy"),
            ConnectorFieldDef("max_posts_per_sync", "integer", default=25,
                              help_text="Max posts to fetch per sync"),
        ],
    ),
}


def get_connector_class(connector_id: str):
    """Lazily import and return the connector class for a given ID."""
    typedef = CONNECTOR_REGISTRY.get(connector_id)
    if not typedef:
        raise ValueError(f"Unknown connector type: {connector_id}")

    module = importlib.import_module(typedef.module_path)
    return getattr(module, typedef.class_name)

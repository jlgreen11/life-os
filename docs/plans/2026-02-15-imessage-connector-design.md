# iMessage Connector Design

## Overview

Add a bidirectional iMessage connector to Life OS that reads messages from macOS's `~/Library/Messages/chat.db` and sends messages via AppleScript. Supports iMessage, SMS/MMS, 1-on-1, and group chats.

## Approach

**chat.db + AppleScript** — the simplest native macOS integration:

- **Read**: Poll `~/Library/Messages/chat.db` (read-only SQLite) every 5 seconds using `message.ROWID` as sync cursor
- **Send**: Execute AppleScript via `osascript` subprocess to send through Messages.app

Alternatives considered and rejected:
- **Shortcuts CLI** — more complex setup, less programmatic control
- **Browser automation** — messages.apple.com is limited and fragile

## Architecture

Standard `BaseConnector` subclass, same pattern as Signal connector.

### Sync (Inbound)

Query `message` joined with `handle` (sender phone/email) and `chat` (group info):

- **Apple epoch**: timestamps are nanoseconds since 2001-01-01. Conversion: `unix_ts = (apple_ns / 1e9) + 978307200`
- **Service type**: `handle.service` distinguishes `"iMessage"` vs `"SMS"` — both ingested
- **Group chats**: detected via `chat.chat_identifier` prefix and `chat.display_name`
- **Direction**: `message.is_from_me` flag
- **Cursor**: `max(message.ROWID)` — monotonically increasing, restart-safe

### Execute (Outbound)

AppleScript via subprocess:
```applescript
tell application "Messages"
    set targetBuddy to buddy "<recipient>" of (service 1 whose service type is iMessage)
    send "<message>" to targetBuddy
end tell
```

Recipient can be phone number or email (both valid iMessage identifiers).

### Event Payloads

Same format as Signal — `message.received` / `message.sent`:
```python
{
    "message_id": "<guid>",
    "channel": "imessage",
    "direction": "inbound",
    "from_address": "+1234567890",
    "body": "...",
    "snippet": "...",
    "is_group": True/False,
    "group_name": "Family Chat",
    "service_type": "iMessage",  # or "SMS"
}
```

### Contact Sync

Extract unique handles from `chat.db`, cross-reference with `contact_identifiers` in `entities.db`. Add `"imessage"` to the contact's `channels` dict. Same dedup logic as Signal (phone map lookup, then name matching).

## Prerequisites

- **Full Disk Access** granted to Python/Terminal in System Settings (for `chat.db` reads)
- **Automation permission** for Messages.app (prompted on first AppleScript send)

## Files to Create/Modify

| File | Action |
|------|--------|
| `connectors/imessage/__init__.py` | Create (empty) |
| `connectors/imessage/connector.py` | Create (main connector class) |
| `connectors/registry.py` | Add `"imessage"` entry to `CONNECTOR_REGISTRY` |
| `models/core.py` | Add `IMESSAGE = "imessage"` to `SourceType` enum |

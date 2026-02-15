"""
Life OS — Connectors Package

Pluggable integration layer for external services. Each connector follows
the BaseConnector lifecycle (authenticate -> sync -> execute -> health_check)
and publishes events to the NATS event bus.

Available connectors:
    - proton_mail: Email via Proton Mail Bridge (IMAP/SMTP)
    - signal_msg: Signal messaging via signal-cli
    - caldav: Calendar sync via CalDAV protocol
    - finance: Financial data via Plaid API
    - home_assistant: Smart home via Home Assistant API
    - browser/*: Web scraping via Playwright (Reddit, YouTube, WhatsApp, generic)
"""

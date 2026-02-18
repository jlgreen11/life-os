"""
Life OS — Shared Marketing/Automated-Sender Filter

Single authoritative implementation of the marketing/automated-sender detection
logic used across the system.  Previously, three separate copies of this logic
existed in:

  - services/signal_extractor/relationship.py  (_is_marketing_or_noreply)
  - services/prediction_engine/engine.py       (_is_marketing_or_noreply)
  - services/behavioral_accuracy_tracker/tracker.py (_is_automated_sender)

Because those copies evolved independently, they diverged: the relationship
extractor was missing the financial (Fidelity, Schwab, PayPal …) and
retail/hospitality (Customerservice, Reservations, WorldofHyatt …) patterns
added to the prediction engine in iterations 171 and 178.  Those omissions
allowed automated senders to accumulate in the relationships profile,
generating opportunity predictions that could never be fulfilled — a root
cause of the 19% opportunity accuracy rate observed in production.

Design decision — pattern scope:
    This module contains the PREDICTION ENGINE's pattern set as the baseline
    (the patterns present in engine.py before this refactor), PLUS the
    financial/retail patterns that were previously missing from the relationship
    extractor.  Patterns that existed only in the relationship extractor and
    not in the engine (e.g. "team@", "ens@", "@mail.") are intentionally kept
    in the relationship extractor's own _is_marketing_or_noreply wrapper, which
    calls this shared function and can apply additional extractor-specific
    strictness via the `extra_localparts` / `extra_domain_patterns` parameters.

    The prediction engine and behavioral accuracy tracker both use this module
    with default parameters, preserving their existing (less strict) behavior.

Usage:

    from services.signal_extractor.marketing_filter import is_marketing_or_noreply

    # Standard call (prediction engine / tracker level strictness)
    if is_marketing_or_noreply(from_address, payload):
        return  # skip this contact

    # Address-only check (no payload, e.g. from a stored contact record)
    if is_marketing_or_noreply(address):
        continue

    # Extra-strict call (relationship extractor level)
    if is_marketing_or_noreply(address, payload,
                               extra_localparts=("team@", "ens@"),
                               extra_domain_patterns=("@mail.",)):
        continue
"""
from __future__ import annotations


def is_marketing_or_noreply(
    from_addr: str,
    payload: dict | None = None,
    *,
    extra_localparts: tuple[str, ...] = (),
    extra_domain_patterns: tuple[str, ...] = (),
) -> bool:
    """Return True if the address belongs to a marketing or automated sender.

    Filters out:
    - No-reply and automated system senders (noreply@, mailer-daemon@, etc.)
    - Bulk/marketing email local-parts (newsletter@, updates@, service@, …)
    - Financial/brokerage automated systems (fidelity@, schwab@, paypal@, …)
    - Retail/hospitality transactional senders (customerservice@, reservations@, …)
    - Embedded notification patterns anywhere in the local-part (-notification,
      news, no.reply, …)
    - Marketing subdomain patterns in the domain part (@email., @comms., @alerts., …)
    - Third-party email marketing platform domains (sendgrid.net, mailchimp.com, …)
    - Brand self-mailer pattern: company@company.com
    - Emails whose body/snippet contains an "Unsubscribe" link

    This is the SINGLE SOURCE OF TRUTH for marketing detection.  All services
    that need to distinguish human contacts from automated senders must import
    and call this function.  Do NOT add a local copy of this logic.

    Args:
        from_addr: The email address to evaluate (any case).
        payload:   Optional event payload dict.  When provided, the body and
                   snippet are checked for unsubscribe indicators.  Pass None
                   (or omit) for address-only checks, e.g. when evaluating a
                   contact address retrieved from a stored profile.
        extra_localparts: Additional local-part prefixes to block (e.g.
                   ``("team@", "ens@")``).  Used by callers that need stricter
                   filtering than the baseline set.
        extra_domain_patterns: Additional domain substring patterns to block
                   (e.g. ``("@mail.",)``).  Used by callers that need stricter
                   domain filtering.

    Returns:
        True if the address is a marketing or automated sender; False if it
        could plausibly be a real human contact.

    Examples:
        >>> is_marketing_or_noreply("noreply@example.com")
        True
        >>> is_marketing_or_noreply("alice@gmail.com")
        False
        >>> is_marketing_or_noreply("schwab@mail.schwab.com")
        True
        >>> is_marketing_or_noreply("john@company.com", {"body": "Unsubscribe"})
        True
        >>> is_marketing_or_noreply("team@startup.io",
        ...                         extra_localparts=("team@",))
        True
    """
    if not from_addr:
        return False

    addr_lower = from_addr.lower()

    # -----------------------------------------------------------------------
    # 1. No-reply and automated system senders
    #
    # Covers all common separators (-, _, none) and + modifier variants.
    # Examples: noreply@, no-reply@, no_reply@, no-reply+ID@domain.com
    # -----------------------------------------------------------------------
    noreply_patterns = (
        "no-reply@", "no_reply@", "noreply@",
        "do-not-reply@", "do_not_reply@", "donotreply@",
        "no-reply+", "no_reply+", "noreply+",   # catches no-reply+ID@domain.com
        "mailer-daemon@", "postmaster@", "daemon@",
        "auto-reply@", "auto_reply@", "autoreply@",
        "automated@", "automation@",
    )
    if any(pattern in addr_lower for pattern in noreply_patterns):
        return True

    # -----------------------------------------------------------------------
    # 2. Bulk-sender local-parts (startswith check on the full address)
    #
    # Must match at the START of the address string to avoid false positives
    # like john.email@company.com or sarah.reply@startup.io.
    #
    # Both singular and plural forms are listed (notification / notifications).
    #
    # Financial/brokerage entries were added in iteration 171 and the
    # retail/hospitality entries in iteration 178.  These are present in the
    # prediction engine but were previously MISSING from the relationship
    # extractor, allowing automated senders to pollute the relationships
    # profile and generate unfulfillable opportunity predictions.
    # -----------------------------------------------------------------------
    bulk_localpart_patterns = (
        # Core marketing / notification patterns
        "newsletter@", "notifications@", "notification@",
        "updates@", "update@", "digest@",
        "mailer@", "bulk@", "promo@", "marketing@",
        "reply@", "email@", "news@",
        "offers@", "offer@", "deals@",
        "hello@", "info@", "support@", "help@",
        "service@", "services@",
        "discover@",
        "alert@", "alerts@",
        "contactus@",
        # Transactional / automated senders
        "orders@", "order@", "receipts@", "receipt@",
        "auto-confirm@", "autoconfirm@", "confirmation@", "confirm@",
        "shipment-tracking@", "shipping@", "delivery@",
        "accountservice@", "account-service@", "account@",
        "yourhealth@", "youraccount@",
        "smartoption@", "quickalert@",
        # Organisational bulk senders
        "communications@", "development@", "fundraising@",
        # Loyalty / rewards programmes
        "rewards@", "loyalty@",
        # Financial / brokerage automated systems (iteration 171)
        #
        # These generate high-volume transactional emails (trade confirmations,
        # statements, alerts) that should NEVER trigger follow-up predictions.
        # Previously missing from the relationship extractor.
        "fidelity.investments@", "fidelity@",
        "schwab@", "vanguard@", "etrade@", "td.ameritrade@", "merrilledge@",
        "robinhood@", "wealthfront@", "betterment@",
        "chase.alerts@", "bankofamerica.alerts@", "citi.alerts@",
        "wellsfargo.alerts@", "usbank.alerts@", "pnc.alerts@",
        "paypal@", "venmo@", "stripe@", "square@",
        "coinbase@", "binance@", "kraken@",
        "experian@", "equifax@", "transunion@", "creditkarma@",
        # Retail / hospitality transactional senders (iteration 178)
        #
        # Production data showed 135 unresolved opportunity predictions for
        # automated senders at hotel chains, airlines, retailers, and services.
        # Previously missing from the relationship extractor.
        "customerservice@",   # e.g., Customerservice@nationalcar.com
        "reservations@",      # e.g., reservations@nationalcar.com
        "onlineservice@",     # e.g., onlineservice@fedex.com
        "return@",            # e.g., return@amazon.com
        "tracking@",          # e.g., tracking@shipstation.com
        "transaction@",       # e.g., transaction@info.samsclub.com
        "online.account@",    # e.g., online.account@marriott.com
        "guestservices@",     # e.g., guestservices@boxoffice.axs.com
        "drivers@",           # e.g., drivers@chargepoint.com
        "gaming@",            # e.g., gaming@nvgaming.nvidia.com
        "messenger@",         # e.g., messenger@messaging.squareup.com
        "tickets@",           # e.g., tickets@transactions.axs.com
        "walgreens@",         # e.g., walgreens@eml.walgreens.com
        "rei@",               # e.g., rei@alerts.rei.com
        "applecash@",         # e.g., applecash@insideapple.apple.com
        "worldofhyatt@",      # e.g., worldofhyatt@loyalty.hyatt.com
        "disneycruiseline@",  # e.g., disneycruiseline@vacations.disneydestinations.com
        # E-delivery / transactional confirmation local-part prefixes.
        # These are financial/brokerage document delivery systems that always
        # send statements, trade confirmations, and tax forms — never human mail.
        # Pattern: eDelivery@etradefrommorganstanley.com, econfirm@schwab.com
        "edelivery@", "econfirm@", "enotice@",
    ) + extra_localparts

    if any(addr_lower.startswith(pattern) for pattern in bulk_localpart_patterns):
        return True

    # -----------------------------------------------------------------------
    # 3. Embedded notification patterns anywhere in the local-part
    #
    # Catches HOA-Notifications@, user-notifications@, lafconews@, etc.
    # Also catches dot-separated no-reply variants like no.reply.alerts@chase.com
    # -----------------------------------------------------------------------
    local_part = addr_lower.split("@")[0] if "@" in addr_lower else addr_lower

    embedded_notification_patterns = (
        "-notification", "-notifications", "-alert", "-alerts",
        "-update", "-updates", "-digest",
        "news",         # lafconews@, morningnews@, dailynews@, …
        "no.reply",     # no.reply.alerts@chase.com
        "do.not.reply", # do.not.reply@domain.com
    )
    if any(pattern in local_part for pattern in embedded_notification_patterns):
        return True

    # -----------------------------------------------------------------------
    # 4. Marketing subdomain patterns in the full address (domain portion)
    #
    # Patterns starting with @ match subdomains after the user's own @, e.g.
    # "@email." matches user@email.domain.com.  Compound patterns without @
    # match anywhere in the domain string.
    #
    # Note: @mail. is NOT included here.  It was removed from the prediction
    # engine in iteration 160 because it incorrectly blocked @gmail.com,
    # @hotmail.com, @protonmail.com.  The relationship extractor can add it
    # back via extra_domain_patterns=("@mail.",) if needed.
    # -----------------------------------------------------------------------
    marketing_domain_patterns = (
        "@news-", "@email.", "@reply.", "@mailing.",
        "@newsletters.", "@promo.", "@marketing.",
        "@em.", "@mg.",
        "@m.",
        "@engage.", "@iluv.", "@e.", "@e2.",
        "@comms.", "@communications.",
        "@attn.",
        "@notification.", "@notifications.",
        "@txn.", "@transactional.",
        "@deals.", "@offers.", "@promo-",
        "@campaigns.", "@campaign.",
        "@blast.", "@bulk.",
        "@lists.", "@list.",
        "@messages.", "@message.",
        "@care.",
        "@mcmap.",
        "@soslprospect.",
        # Retail/hospitality subdomains (iteration 178)
        "@alerts.",        # e.g., @alerts.rei.com, @alerts.chase.com
        "@loyalty.",       # e.g., @loyalty.hyatt.com
        "@vacations.",     # e.g., @vacations.disneydestinations.com
        "@transactions.",  # e.g., @transactions.axs.com
        "@eml.",           # e.g., @eml.walgreens.com
        "@insideapple.",   # e.g., @insideapple.apple.com
        "@card.",          # e.g., @card.southwest.com
        "@odysseymail.",   # e.g., @odysseymail.tylertech.cloud
        "@mc.",            # e.g., @mc.ihg.com
        "@eg.",            # e.g., @eg.vrbo.com
        # @mail. subdomain — catches financial/brokerage bulk mailers that use
        # a dedicated "mail." subdomain (mail.fidelity.com, mail.schwab.com,
        # mail.instagram.com).  This does NOT false-positive on gmail.com,
        # hotmail.com, or protonmail.com because those addresses contain
        # "@gmail.com", "@hotmail.com", "@protonmail.com" — none of which
        # contain the substring "@mail." (the @ anchors to the user boundary).
        "@mail.",          # e.g., benefitscenter@mail.fidelity.com
        # Airline/travel automated mailer subdomains
        "@ifly.",          # e.g., passenger@ifly.southwest.com (Southwest Airlines)
        "@trx.",           # e.g., user@trx.company.com (transactional subdomain)
        "@shareholderdocs.",  # e.g., Fidelity.Investments@shareholderdocs.fidelity.com
        "@investordelivery.",  # e.g., user@investordelivery.company.com
        # Compound subdomain patterns (no leading @; match anywhere in domain)
        "info.email.",     # e.g., @info.email.aa.com
        "points-mail.",    # e.g., points@points-mail.ihg.com (IHG loyalty)
    ) + extra_domain_patterns

    if any(pattern in addr_lower for pattern in marketing_domain_patterns):
        return True

    # -----------------------------------------------------------------------
    # 5. Third-party email marketing / transactional platform domains
    #
    # Checks whether the domain ENDS WITH any of these suffixes.  Also checks
    # for apex-domain matches (e.g. bytebytego@substack.com).
    # -----------------------------------------------------------------------
    domain = addr_lower.split("@")[1] if "@" in addr_lower else ""

    marketing_service_patterns = (
        ".e2ma.net",
        ".sendgrid.net",
        ".mailchimp.com",
        ".constantcontact.com",
        ".hubspot.com",
        ".marketo.com",
        ".pardot.com",
        ".eloqua.com",
        ".sailthru.com",
        ".responsys.net",
        ".exacttarget.com",
        ".smtp2go.com",
        ".postmarkapp.com",
        ".mandrillapp.com",
        ".amazonses.com",
        ".sparkpostmail.com",
        ".sendinblue.com",
        ".intercom-mail.com",
        ".customer.io",
        ".iterable.com",
        ".klaviyo.com",
        # Specific automated domains (iteration 178)
        "proxyvote.com",        # Proxy voting (Broadridge Financial)
        "playatmcd.com",        # McDonald's Monopoly promo
        "facebookmail.com",     # Facebook automated notifications
        ".smg.com",             # Service Management Group surveys
        ".ms.aa.com",           # American Airlines info subdomain
        # Newsletter/creator platforms — every address @substack.com or
        # @beehiiv.com is an automated publication, never a human replying.
        # Both apex and subdomain variants are caught: "substack.com" matches
        # both "bytebytego@substack.com" and "letter@publication.substack.com"
        # via str.endswith() because "publication.substack.com" ends with
        # "substack.com" as well.
        "substack.com",         # Newsletter platform (e.g. bytebytego@substack.com)
        "beehiiv.com",          # Newsletter platform (e.g. author@newsletter.beehiiv.com)
    )
    if any(domain.endswith(pattern) for pattern in marketing_service_patterns):
        return True

    # -----------------------------------------------------------------------
    # 6. Brand self-mailer pattern: company@company.com
    #
    # When the local-part matches the domain name it's almost always marketing.
    # Length check (> 3) avoids false positives like me@me.com.
    # -----------------------------------------------------------------------
    if "@" in addr_lower and "." in domain:
        local_normalized = local_part.replace("-", "").replace("_", "").replace(".", "")
        domain_base = domain.split(".")[0].replace("-", "").replace("_", "")
        if local_normalized == domain_base and len(local_normalized) > 3:
            return True

    # -----------------------------------------------------------------------
    # 7. Unsubscribe link in body / snippet
    #
    # Marketing emails are legally required to include unsubscribe links.
    # Only checked when a payload is supplied.
    # -----------------------------------------------------------------------
    if payload:
        text = " ".join(filter(None, [
            payload.get("body_plain", ""),
            payload.get("snippet", ""),
            payload.get("body", ""),
        ])).lower()
        if "unsubscribe" in text:
            return True

    return False

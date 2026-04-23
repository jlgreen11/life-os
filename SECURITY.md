# Security Policy

Life OS processes personal data: email, messages, calendar, location, device
proximity. Security reports are taken seriously.

## Reporting a vulnerability

**Do not open a public GitHub issue for security problems.**

Report privately via one of:
- GitHub Security Advisories — <https://github.com/jlgreen11/life-os/security/advisories/new>
- DM on GitHub to [@jlgreen11](https://github.com/jlgreen11)

Please include:
- A description of the issue
- Steps to reproduce
- Affected version / commit SHA
- Impact assessment (what can an attacker do?)

I aim to acknowledge reports within 72 hours and give a first remediation
estimate within 1 week.

## Scope

In scope:
- Code in this repository
- Any hosted service I operate under this project's name (none at time of writing)

Out of scope:
- Third-party services Life OS connects to (Proton, Google, Plaid, etc.) —
  report to those vendors directly
- The operator's own Mac Mini or network configuration
- Social-engineering the operator

## Privacy model (brief)

- Local-first: all user data lives in local SQLite + LanceDB on the operator's
  machine. No cloud sync by default.
- Encrypted credentials: connector passwords and API keys are Fernet-encrypted
  at rest.
- PII shield: when the optional cloud AI path runs, PII is tokenized on the
  client before leaving the device; real values are restored on response. The
  cloud model never sees raw PII.

See `README.md` and `DESIGN.md` for broader context.

//
//  WebSocketEvent.swift
//  Life OS — Typed envelope for v2 `/ws` push events
//
//  The v2 backend's WebSocket fan-out lives in
//  `core/moment/broadcaster.py`. The web client (HTMX) consumes pre-
//  rendered HTML fragments out-of-band-swap style; the iOS client
//  needs typed JSON it can route into ViewModels without parsing
//  HTML. Until the broadcaster grows a JSON channel (planned for
//  Phase 2 alongside APNs), this file is the single source of truth
//  for the iOS-side wire contract — server changes must keep the
//  envelope in lockstep with the cases here.
//
//  Wire envelope
//  -------------
//  Every push is a single text frame containing one JSON object:
//
//      { "type": "<event-name>", "data": { ...payload... } }
//
//  The `type` discriminator drives decoding; anything else (heartbeat
//  echoes, server-side debug strings) is ignored. Unknown `type`
//  values surface as `WebSocketEvent.unknown(String)` so a client
//  rolled out before a server-side new event type still functions —
//  call sites handle .unknown silently rather than throwing.
//
//  Timestamps are Unix epoch SECONDS (matches the rest of the v2
//  contract — see `Moment.swift`). Use `WebSocketEvent.decoder()`
//  to get a JSONDecoder pre-configured with `.secondsSince1970`.
//
//  Heartbeats
//  ----------
//  The client sends `{"type": "ping"}` text frames every 30s. The
//  server treats them as no-ops (the `/ws` endpoint blocks on
//  `receive_text` and discards anything inbound). The pings exist
//  purely to keep intermediate proxies (carrier NAT, corporate
//  middleboxes) from idle-timing the socket out from underneath us.
//

import Foundation

// MARK: - Event payloads

/// `moment.state_changed` payload.
///
/// Emitted whenever the state machine in `core/moment/state.py`
/// successfully transitions a Moment. Carries the from/to states so
/// the iOS UI can animate accordingly without re-fetching the
/// Moment object — the change ID lets a stale-cache client refresh
/// itself if the from-state doesn't match its current snapshot.
struct MomentStateChange: Decodable, Equatable {
    let id: String
    let fromState: MomentState?
    let toState: MomentState
    let ts: Date
    let annotation: String?

    enum CodingKeys: String, CodingKey {
        case id
        case fromState = "from_state"
        case toState = "to_state"
        case ts
        case annotation
    }
}

/// `connector.status_changed` payload.
///
/// Emitted by `storage/repos/connectors.py` when a connector flips
/// online/offline or records a fresh `last_error`. The Settings tab
/// uses these to update the status dot live without polling.
struct ConnectorStatusChange: Decodable, Equatable {
    let id: String
    let status: String
    let lastSyncAt: Date?
    let lastError: String?

    enum CodingKeys: String, CodingKey {
        case id
        case status
        case lastSyncAt = "last_sync_at"
        case lastError = "last_error"
    }
}

// MARK: - Event enum

/// One push from `/ws`. Discriminator is the JSON `type` field.
///
/// `.unknown` is the forward-compat sink: a client decoded against
/// an older event vocabulary still routes the message instead of
/// crashing. Call sites pattern-match the cases they care about and
/// drop the rest.
enum WebSocketEvent: Decodable, Equatable {
    case momentCreated(Moment)
    case momentStateChanged(MomentStateChange)
    case connectorStatusChanged(ConnectorStatusChange)
    case unknown(type: String)

    private enum Envelope: String, CodingKey {
        case type
        case data
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: Envelope.self)
        let type = try c.decode(String.self, forKey: .type)
        switch type {
        case "moment.created":
            let moment = try c.decode(Moment.self, forKey: .data)
            self = .momentCreated(moment)
        case "moment.state_changed":
            let change = try c.decode(MomentStateChange.self, forKey: .data)
            self = .momentStateChanged(change)
        case "connector.status_changed":
            let change = try c.decode(ConnectorStatusChange.self, forKey: .data)
            self = .connectorStatusChanged(change)
        default:
            // Forward-compat: don't fail decoding for unknown types
            // (see file header). The string is preserved so call
            // sites can log if they want.
            self = .unknown(type: type)
        }
    }

    /// JSONDecoder pre-configured for the v2 WebSocket wire format.
    ///
    /// Mirrors `Moment.decoder()` so the date strategy lines up
    /// across REST + WebSocket payloads. Call sites typically use
    /// this rather than constructing their own.
    static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .secondsSince1970
        return d
    }
}

//
//  APIClient.swift
//  Life OS — REST client for the v2 API surface
//
//  One method per locked REST endpoint per engineering plan
//  § "14-endpoint API contract" plus the iOS-compat shim at
//  `/api/{context,briefing,feedback,preferences,status}` served by
//  `api/routes/context.py`.
//
//  Timestamps are Unix epoch seconds (Int) on the wire — both encoders
//  and decoders here stay at the default (Int round-trip) because the
//  Moment type is the only payload that uses `Date` fields, and it
//  carries its own `.secondsSince1970` decoder via `Moment.decoder()`.
//
//  Testability: the actor accepts an injected `URLSession` so tests
//  can pass a session wired to a `URLProtocol` mock. Production code
//  uses the default `.default` configuration.
//

import Foundation

actor APIClient {
    /// Base URL prefix (e.g. `"http://lifeos.local:8080"`), trailing
    /// slash stripped on init. Every call composes the full URL as
    /// `baseURL + path`; `path` always carries its leading slash.
    private let baseURL: String
    private let session: URLSession
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(baseURL: String, session: URLSession? = nil) {
        self.baseURL = baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if let session = session {
            self.session = session
        } else {
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 30
            config.timeoutIntervalForResource = 60
            self.session = URLSession(configuration: config)
        }
    }

    // MARK: - Now tab

    /// `GET /api/now` — pending · scheduled · done-today.
    ///
    /// Returns the full `MomentFeed` (three bounded lists). The
    /// server-side limits are 20/10/10 (engineering plan § "GET
    /// /api/now"); this client does not paginate — callers re-hit the
    /// endpoint after an action to refresh.
    func getNow() async throws -> MomentFeed {
        return try await get("/api/now", decoder: Moment.decoder())
    }

    /// `POST /api/moments/{id}/accept` — transition SUGGESTED → ACCEPTED.
    ///
    /// The outbox dispatch (email draft, calendar entry, …) is
    /// enqueued server-side in the same transaction as the state
    /// change and becomes eligible after the 3-second Undo grace.
    func acceptMoment(id: String, annotation: String? = nil) async throws -> Moment {
        let body = MomentActionBody(annotation: annotation)
        return try await post("/api/moments/\(id)/accept", body: body, decoder: Moment.decoder())
    }

    /// `POST /api/moments/{id}/dismiss` — transition SUGGESTED → DISMISSED.
    /// Terminal state. Feeds signal=0.0 into the feedback-weight EWMA.
    func dismissMoment(id: String, annotation: String? = nil) async throws -> Moment {
        let body = MomentActionBody(annotation: annotation)
        return try await post("/api/moments/\(id)/dismiss", body: body, decoder: Moment.decoder())
    }

    /// `POST /api/moments/{id}/snooze` — transition SUGGESTED → SNOOZED.
    ///
    /// ``snoozeUntil`` is Unix epoch seconds. A value past the
    /// Moment's ``expiresAt`` is coerced server-side to EXPIRED; the
    /// returned Moment carries the post-transition state so the UI
    /// can redraw accordingly.
    func snoozeMoment(id: String, snoozeUntil: Int, annotation: String? = nil) async throws -> Moment {
        let body = MomentActionBody(snoozeUntil: snoozeUntil, annotation: annotation)
        return try await post("/api/moments/\(id)/snooze", body: body, decoder: Moment.decoder())
    }

    /// `POST /api/moments/{id}/edit` — replace ``proposed_action.params``.
    ///
    /// Used by the inline draft editor: the user tweaks a generated
    /// message before accepting. State stays SUGGESTED. ``params`` is
    /// a full replace (not a merge) — clients must send the complete
    /// payload.
    func editMoment(id: String, actionParams: [String: AnyCodable]) async throws -> Moment {
        let body = MomentActionBody(actionParams: actionParams)
        return try await post("/api/moments/\(id)/edit", body: body, decoder: Moment.decoder())
    }

    /// `POST /api/moments/{id}/undo` — reverse the last terminal
    /// transition within the 3-second grace window. Returns 410 if
    /// the window has elapsed.
    func undoMoment(id: String) async throws -> Moment {
        return try await postEmpty("/api/moments/\(id)/undo", decoder: Moment.decoder())
    }

    // MARK: - You tab

    /// `GET /api/you` — self-portrait (observed months, interactions,
    /// when-at-best, how-you-write, routines, drifting). Empty-state
    /// installs round-trip with every list at ``[]``.
    func getYou() async throws -> SelfPortrait {
        return try await get("/api/you")
    }

    // MARK: - People tab

    /// `GET /api/people` — roster with YOU pinned to the top.
    ///
    /// ``query`` is a case-insensitive substring filter on
    /// name/contact-id; ``page`` is 1-indexed; ``pageSize`` matches
    /// the server-side `MAX_PAGE_SIZE`. Empty filter means "no filter".
    func getPeople(query: String? = nil, page: Int = 1, pageSize: Int? = nil) async throws -> PeopleList {
        var items: [URLQueryItem] = []
        if let q = query, !q.isEmpty {
            items.append(URLQueryItem(name: "q", value: q))
        }
        if page != 1 {
            items.append(URLQueryItem(name: "page", value: String(page)))
        }
        if let size = pageSize {
            items.append(URLQueryItem(name: "page_size", value: String(size)))
        }
        return try await get("/api/people", queryItems: items)
    }

    /// `GET /api/people/{id}` — per-contact dossier.
    ///
    /// 404 for unknown contacts. The dossier is agent-facing — it
    /// surfaces comm-template, cadence sparkline, recent topics, and
    /// the predicted-next-contact text from the relationship producer.
    func getContact(id: String) async throws -> ContactDossier {
        return try await get("/api/people/\(id)")
    }

    // MARK: - Settings tab — Connectors

    /// `GET /api/connectors` — one row per registered connector.
    ///
    /// Fernet-encrypted credentials are never returned on this path —
    /// only status-level fields (enabled flag, last-sync timestamp,
    /// last-error string).
    func getConnectors() async throws -> [Connector] {
        return try await get("/api/connectors")
    }

    /// `PATCH /api/connectors/{id}` — update enabled / config / secrets.
    ///
    /// Any of the three body fields are optional; omitting a field
    /// keeps the current value. Plaintext secrets on the wire are
    /// acceptable because the v2 API ships Tailscale-only in Phase 1
    /// and the repo re-encrypts with Fernet server-side.
    func updateConnector(id: String, update: ConnectorConfigUpdate) async throws -> Connector {
        return try await patch("/api/connectors/\(id)", body: update)
    }

    /// `POST /api/connectors/{id}/test` — dry-run sync against a
    /// connector. Response shape is free-form (per-connector
    /// diagnostics); the sensible default is
    /// `{"ok": Bool, "message": String, "details": {...}}`.
    func testConnector(id: String) async throws -> [String: AnyCodable] {
        return try await postEmpty("/api/connectors/\(id)/test")
    }

    // MARK: - Health + status

    /// `GET /api/health` — deep-health multi-key payload. Unlike the
    /// iOS-compat ``getStatus``, this surfaces per-connector strings,
    /// scheduler heartbeat, producer activity, and pending-moment
    /// count so operators see the failure mode at a glance.
    func getHealth() async throws -> HealthStatus {
        return try await get("/api/health")
    }

    /// `GET /api/status` — iOS launch-time smoke check. Flat 4-key
    /// shape (ok, ts, event_count, moment_count). Used by the iOS
    /// app on app-open before any tab loads.
    func getStatus() async throws -> StatusSmoke {
        return try await get("/api/status")
    }

    // MARK: - Legacy proxies (iOS compat shim)

    /// `GET /api/briefing` — daily briefing via the v2 AI engine.
    /// Keeps the v1 `{briefing, generated_at, error?}` shape so the
    /// iOS dashboard can render without a paired release. ``briefing``
    /// is nullable; ``error`` carries a diagnostic when present.
    func getBriefing() async throws -> BriefingResponse {
        return try await get("/api/briefing")
    }

    /// `POST /api/feedback` — persist free-text feedback as an
    /// append-only `system.user.feedback` event. The decision loop
    /// behind the v1 feedback collector is deliberately NOT
    /// re-implemented (CEO plan § "Killed Insights"), but the audit
    /// trail survives so future loops can re-derive signal.
    func submitFeedback(message: String) async throws {
        let body = ["message": message]
        let _: EmptyResponse = try await post("/api/feedback", body: body)
    }

    /// `POST /api/preferences` — upsert one `(key, value)` preference
    /// row. ``value`` accepts any JSON-serialisable scalar or object
    /// (v2 stores as JSON text regardless of source type).
    func updatePreference(key: String, value: AnyCodable) async throws {
        let body = PreferenceUpsert(key: key, value: value)
        let _: EmptyResponse = try await post("/api/preferences", body: body)
    }

    // MARK: - Context pipeline (mobile-sensor ingestion)

    /// `POST /api/context/event` — single mobile context event.
    /// Called by `LocationManager` / `DeviceDiscovery` on state changes.
    func submitContextEvent(_ event: ContextEvent) async throws {
        let _: EmptyResponse = try await post("/api/context/event", body: event)
    }

    /// `POST /api/context/batch` — batched mobile context events.
    /// Called by `BackgroundTaskManager` during background refresh.
    func submitContextBatch(_ events: [ContextEvent]) async throws {
        let body = ContextBatch(events: events)
        let _: EmptyResponse = try await post("/api/context/batch", body: body)
    }

    /// `GET /api/context/summary` — rolling summary of recent context.
    /// Free-form dict shape (per-install surface) — return as a
    /// decoded JSON object so callers can inspect individual keys
    /// without churning a shared schema every install.
    func getContextSummary() async throws -> [String: AnyCodable] {
        return try await get("/api/context/summary")
    }

    // MARK: - HTTP helpers

    private func get<T: Decodable>(
        _ path: String,
        queryItems: [URLQueryItem] = [],
        decoder: JSONDecoder? = nil
    ) async throws -> T {
        let url = try buildURL(path: path, queryItems: queryItems)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return try decodeResponse(T.self, from: data, path: path, decoder: decoder)
    }

    private func post<T: Decodable>(
        _ path: String,
        body: some Encodable,
        decoder: JSONDecoder? = nil
    ) async throws -> T {
        let url = try buildURL(path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try encoder.encode(body)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return try decodeResponse(T.self, from: data, path: path, decoder: decoder)
    }

    /// POST with no body — used for `undo` + `connector/test` where
    /// the endpoint keys off the URL path alone.
    private func postEmpty<T: Decodable>(
        _ path: String,
        decoder: JSONDecoder? = nil
    ) async throws -> T {
        let url = try buildURL(path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return try decodeResponse(T.self, from: data, path: path, decoder: decoder)
    }

    private func patch<T: Decodable>(
        _ path: String,
        body: some Encodable,
        decoder: JSONDecoder? = nil
    ) async throws -> T {
        let url = try buildURL(path: path)
        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try encoder.encode(body)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return try decodeResponse(T.self, from: data, path: path, decoder: decoder)
    }

    private func buildURL(path: String, queryItems: [URLQueryItem] = []) throws -> URL {
        guard var components = URLComponents(string: "\(baseURL)\(path)") else {
            throw APIError.invalidResponse
        }
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        guard let url = components.url else {
            throw APIError.invalidResponse
        }
        return url
    }

    /// Decodes a response, wrapping DecodingError with endpoint
    /// context so debug logs point at the failing route.
    private func decodeResponse<T: Decodable>(
        _ type: T.Type,
        from data: Data,
        path: String,
        decoder overrideDecoder: JSONDecoder?
    ) throws -> T {
        // `EmptyResponse` is the zero-payload sentinel — the server
        // returns either `{}` or an empty body for write acks. Handle
        // the truly empty case up front so callers don't need to hack
        // a placeholder byte into their handlers.
        if type == EmptyResponse.self, data.isEmpty {
            return EmptyResponse() as! T
        }
        let d = overrideDecoder ?? decoder
        do {
            return try d.decode(type, from: data)
        } catch let error as DecodingError {
            let preview = String(data: data.prefix(500), encoding: .utf8) ?? "<non-UTF8 data>"
            print("APIClient decode error on \(path): \(error)\nResponse body preview: \(preview)")
            throw APIError.decodingError(error)
        }
    }

    private func validateResponse(_ response: URLResponse) throws {
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            throw APIError.httpError(statusCode: httpResponse.statusCode)
        }
    }
}

// MARK: - Errors & helpers

enum APIError: LocalizedError, Equatable {
    case invalidResponse
    case httpError(statusCode: Int)
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidResponse: return "Invalid response from server"
        case .httpError(let code): return "HTTP error \(code)"
        case .decodingError(let error): return "Decoding error: \(error.localizedDescription)"
        }
    }

    static func == (lhs: APIError, rhs: APIError) -> Bool {
        switch (lhs, rhs) {
        case (.invalidResponse, .invalidResponse): return true
        case (.httpError(let a), .httpError(let b)): return a == b
        case (.decodingError, .decodingError): return true
        default: return false
        }
    }
}

struct EmptyResponse: Codable, Equatable {}

/// Upsert-body for `POST /api/preferences`. ``value`` is open-shape
/// (AnyCodable) so callers can send scalars, arrays, or nested
/// objects without re-deriving a schema per setting.
private struct PreferenceUpsert: Codable {
    let key: String
    let value: AnyCodable
}

/// Wrapper for `POST /api/context/batch` — matches the Pydantic
/// `ContextBatchIn {events: [ContextEventIn]}` shape on the server.
private struct ContextBatch: Codable {
    let events: [ContextEvent]
}

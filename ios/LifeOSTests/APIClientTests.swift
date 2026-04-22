//
//  APIClientTests.swift
//  Life OS — APIClient unit tests against a URLProtocol mock
//
//  One test per locked v2 endpoint (per NEXT_TASKS.md Category A
//  "Regenerate APIClient.swift against v2 endpoints"). Assertions:
//
//  - URL + method + body shape match the server-side contract in
//    `api/schemas.py` and `api/routes/*`
//  - Response decoding round-trips a realistic fixture
//  - HTTP errors surface as `APIError.httpError`
//
//  The tests use a custom `URLProtocol` subclass registered on a
//  `URLSessionConfiguration.ephemeral` session so we intercept every
//  request the actor makes without touching the network.
//

import XCTest
@testable import LifeOS

// MARK: - URLProtocol mock

/// Intercepts every request on sessions whose config lists it in
/// `protocolClasses`. Tests push a `(request -> (status, headers, body))`
/// handler into the static slot before each call; the handler fires
/// synchronously inside `startLoading`, so the test can assert on the
/// captured request after awaiting the API call.
final class MockURLProtocol: URLProtocol {
    typealias Handler = (URLRequest) throws -> (HTTPURLResponse, Data)

    static var handler: Handler?
    static var lastRequest: URLRequest?
    static var lastBody: Data?

    static func reset() {
        handler = nil
        lastRequest = nil
        lastBody = nil
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        // `URLRequest.httpBody` can be nil when the body was set via
        // an `InputStream` (URLSession does this for non-trivial
        // bodies). Capture from the stream when that happens so
        // tests can inspect the encoded JSON reliably.
        MockURLProtocol.lastRequest = request
        if let body = request.httpBody {
            MockURLProtocol.lastBody = body
        } else if let stream = request.httpBodyStream {
            MockURLProtocol.lastBody = Self.drain(stream)
        } else {
            MockURLProtocol.lastBody = nil
        }

        guard let handler = MockURLProtocol.handler else {
            let err = NSError(domain: "MockURLProtocol", code: -1, userInfo: [
                NSLocalizedDescriptionKey: "handler not set",
            ])
            client?.urlProtocol(self, didFailWithError: err)
            return
        }
        do {
            let (response, data) = try handler(request)
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}

    private static func drain(_ stream: InputStream) -> Data {
        var data = Data()
        stream.open()
        defer { stream.close() }
        let bufSize = 4096
        let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: bufSize)
        defer { buf.deallocate() }
        while stream.hasBytesAvailable {
            let read = stream.read(buf, maxLength: bufSize)
            if read <= 0 { break }
            data.append(buf, count: read)
        }
        return data
    }
}

// MARK: - Helpers

/// Build an APIClient whose session routes every request through
/// `MockURLProtocol`. The base URL matches `http://stub.local` so
/// captured request URLs are stable across test runs.
private func makeClient(baseURL: String = "http://stub.local") -> APIClient {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [MockURLProtocol.self]
    let session = URLSession(configuration: config)
    return APIClient(baseURL: baseURL, session: session)
}

private func respond(
    status: Int = 200,
    json: String,
    contentType: String = "application/json"
) {
    MockURLProtocol.handler = { request in
        let data = Data(json.utf8)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": contentType]
        )!
        return (response, data)
    }
}

private func respondEmpty(status: Int = 200) {
    MockURLProtocol.handler = { request in
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "application/json"]
        )!
        return (response, Data())
    }
}

// MARK: - Now-tab endpoints

final class APIClientNowTests: XCTestCase {

    override func tearDown() {
        MockURLProtocol.reset()
        super.tearDown()
    }

    func test_getNow_hitsApiNowAndDecodesMomentFeed() async throws {
        let body = #"""
        {
          "pending": [],
          "scheduled": [],
          "done": []
        }
        """#
        respond(json: body)

        let client = makeClient()
        let feed = try await client.getNow()

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/now")
        XCTAssertEqual(MockURLProtocol.lastRequest?.httpMethod, "GET")
        XCTAssertTrue(feed.pending.isEmpty)
        XCTAssertTrue(feed.scheduled.isEmpty)
        XCTAssertTrue(feed.done.isEmpty)
    }

    func test_acceptMoment_postsToAcceptPathWithAnnotation() async throws {
        respond(json: momentJSON(state: "accepted"))

        let client = makeClient()
        let moment = try await client.acceptMoment(id: "M1", annotation: "note")

        let req = try XCTUnwrap(MockURLProtocol.lastRequest)
        XCTAssertEqual(req.url?.path, "/api/moments/M1/accept")
        XCTAssertEqual(req.httpMethod, "POST")
        XCTAssertEqual(req.value(forHTTPHeaderField: "Content-Type"), "application/json")
        let decoded = try JSONSerialization.jsonObject(with: MockURLProtocol.lastBody ?? Data()) as? [String: Any]
        XCTAssertEqual(decoded?["annotation"] as? String, "note")
        XCTAssertEqual(moment.state, .accepted)
    }

    func test_dismissMoment_postsToDismissPath() async throws {
        respond(json: momentJSON(state: "dismissed"))

        let client = makeClient()
        let moment = try await client.dismissMoment(id: "M2")

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/moments/M2/dismiss")
        XCTAssertEqual(MockURLProtocol.lastRequest?.httpMethod, "POST")
        XCTAssertEqual(moment.state, .dismissed)
    }

    func test_snoozeMoment_postsSnoozeUntilInBody() async throws {
        respond(json: momentJSON(state: "snoozed", snoozeUntil: 1777300000))

        let client = makeClient()
        let moment = try await client.snoozeMoment(id: "M3", snoozeUntil: 1777300000)

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/moments/M3/snooze")
        let decoded = try JSONSerialization.jsonObject(with: MockURLProtocol.lastBody ?? Data()) as? [String: Any]
        XCTAssertEqual(decoded?["snooze_until"] as? Int, 1777300000)
        XCTAssertEqual(moment.state, .snoozed)
    }

    func test_editMoment_sendsActionParamsUnderSnakeCaseKey() async throws {
        respond(json: momentJSON(state: "suggested"))

        let client = makeClient()
        _ = try await client.editMoment(
            id: "M4",
            actionParams: ["body": AnyCodable("hello")]
        )

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/moments/M4/edit")
        let decoded = try JSONSerialization.jsonObject(with: MockURLProtocol.lastBody ?? Data()) as? [String: Any]
        let params = decoded?["action_params"] as? [String: Any]
        XCTAssertEqual(params?["body"] as? String, "hello")
    }

    func test_undoMoment_postsEmptyBody() async throws {
        respond(json: momentJSON(state: "suggested"))

        let client = makeClient()
        _ = try await client.undoMoment(id: "M5")

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/moments/M5/undo")
        XCTAssertEqual(MockURLProtocol.lastRequest?.httpMethod, "POST")
        XCTAssertTrue((MockURLProtocol.lastBody ?? Data()).isEmpty)
    }
}

// MARK: - You-tab + People-tab

final class APIClientYouPeopleTests: XCTestCase {

    override func tearDown() {
        MockURLProtocol.reset()
        super.tearDown()
    }

    func test_getYou_decodesSelfPortraitDefaults() async throws {
        let body = #"""
        {
          "observed_months": 6,
          "interactions_count": 120,
          "confidence_pct": 72,
          "when_at_best": ["mornings before standup"],
          "how_you_write": [],
          "your_routines": [],
          "drifting": []
        }
        """#
        respond(json: body)

        let client = makeClient()
        let you = try await client.getYou()

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/you")
        XCTAssertEqual(you.observedMonths, 6)
        XCTAssertEqual(you.whenAtBest, ["mornings before standup"])
    }

    func test_getPeople_encodesQueryParams() async throws {
        let body = peopleListJSON()
        respond(json: body)

        let client = makeClient()
        _ = try await client.getPeople(query: "sam", page: 2, pageSize: 10)

        let url = try XCTUnwrap(MockURLProtocol.lastRequest?.url)
        let components = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let items = components.queryItems ?? []
        let dict = Dictionary(uniqueKeysWithValues: items.map { ($0.name, $0.value ?? "") })
        XCTAssertEqual(url.path, "/api/people")
        XCTAssertEqual(dict["q"], "sam")
        XCTAssertEqual(dict["page"], "2")
        XCTAssertEqual(dict["page_size"], "10")
    }

    func test_getPeople_omitsQueryWhenDefault() async throws {
        respond(json: peopleListJSON())

        let client = makeClient()
        _ = try await client.getPeople()

        let url = try XCTUnwrap(MockURLProtocol.lastRequest?.url)
        XCTAssertEqual(url.path, "/api/people")
        // Default call should not add any query items; the server
        // applies its own defaults.
        XCTAssertNil(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems)
    }

    func test_getContact_decodesDossier() async throws {
        let body = #"""
        {
          "contact_id": "c1",
          "name": "Sam",
          "last_contact_ts": 1777200000,
          "usual_cadence_days": 4,
          "comm_template": "warm, short",
          "cadence_sparkline": [1, 0, 1, 0, 1, 0, 0],
          "recent_topics": ["climbing"],
          "predicted_next": "tomorrow evening"
        }
        """#
        respond(json: body)

        let client = makeClient()
        let dossier = try await client.getContact(id: "c1")

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/people/c1")
        XCTAssertEqual(dossier.name, "Sam")
        XCTAssertEqual(dossier.cadenceSparkline.count, 7)
    }
}

// MARK: - Connectors

final class APIClientConnectorTests: XCTestCase {

    override func tearDown() {
        MockURLProtocol.reset()
        super.tearDown()
    }

    func test_getConnectors_decodesList() async throws {
        let body = #"""
        [
          {"id": "proton", "kind": "imap", "enabled": true, "status": "ok", "last_sync_at": 1777200000, "last_error": null},
          {"id": "imessage", "kind": "imessage", "enabled": false, "status": "unknown", "last_sync_at": null, "last_error": null}
        ]
        """#
        respond(json: body)

        let client = makeClient()
        let connectors = try await client.getConnectors()

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/connectors")
        XCTAssertEqual(connectors.count, 2)
        XCTAssertEqual(connectors[0].id, "proton")
        XCTAssertEqual(connectors[0].status, "ok")
    }

    func test_updateConnector_usesPatchWithEnabledField() async throws {
        let body = #"""
        {"id": "proton", "kind": "imap", "enabled": false, "status": "ok", "last_sync_at": null, "last_error": null}
        """#
        respond(json: body)

        let client = makeClient()
        _ = try await client.updateConnector(
            id: "proton",
            update: ConnectorConfigUpdate(enabled: false)
        )

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/connectors/proton")
        XCTAssertEqual(MockURLProtocol.lastRequest?.httpMethod, "PATCH")
        let decoded = try JSONSerialization.jsonObject(with: MockURLProtocol.lastBody ?? Data()) as? [String: Any]
        XCTAssertEqual(decoded?["enabled"] as? Bool, false)
    }

    func test_testConnector_postsToTestPathAndDecodesFreeFormBody() async throws {
        let body = #"""
        {"ok": true, "message": "authenticated", "details": {"folders": 14}}
        """#
        respond(json: body)

        let client = makeClient()
        let result = try await client.testConnector(id: "proton")

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/connectors/proton/test")
        XCTAssertEqual(MockURLProtocol.lastRequest?.httpMethod, "POST")
        XCTAssertEqual(result["ok"]?.value as? Bool, true)
    }
}

// MARK: - Health + smoke status

final class APIClientHealthTests: XCTestCase {

    override func tearDown() {
        MockURLProtocol.reset()
        super.tearDown()
    }

    func test_getHealth_decodesDeepHealthPayload() async throws {
        let body = #"""
        {
          "ok": true,
          "ts": 1777200000,
          "connectors": {"proton": "ok", "imessage": "ok"},
          "db_last_write_ts": 1777199990,
          "scheduler_heartbeat_ts": 1777199995,
          "producer_activity": {"cadence": 3, "relationship": 1},
          "pending_moments": 2,
          "notes": []
        }
        """#
        respond(json: body)

        let client = makeClient()
        let health = try await client.getHealth()

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/health")
        XCTAssertTrue(health.ok)
        XCTAssertEqual(health.connectors["proton"], "ok")
        XCTAssertEqual(health.pendingMoments, 2)
    }

    func test_getStatus_decodesSmokePayload() async throws {
        let body = #"""
        {"ok": true, "ts": 1777200000, "event_count": 10000, "moment_count": 42}
        """#
        respond(json: body)

        let client = makeClient()
        let status = try await client.getStatus()

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/status")
        XCTAssertEqual(status.eventCount, 10000)
        XCTAssertEqual(status.momentCount, 42)
    }

    func test_httpError_surfacesAsAPIError() async throws {
        respondEmpty(status: 503)

        let client = makeClient()
        do {
            _ = try await client.getStatus()
            XCTFail("expected APIError.httpError")
        } catch let APIError.httpError(code) {
            XCTAssertEqual(code, 503)
        } catch {
            XCTFail("expected APIError.httpError; got \(error)")
        }
    }
}

// MARK: - Legacy proxies + preferences + context

final class APIClientLegacyProxyTests: XCTestCase {

    override func tearDown() {
        MockURLProtocol.reset()
        super.tearDown()
    }

    func test_getBriefing_decodesLegacyShape() async throws {
        let body = #"""
        {"briefing": "Three moments today.", "generated_at": "2026-04-22T12:00:00+00:00", "error": null}
        """#
        respond(json: body)

        let client = makeClient()
        let briefing = try await client.getBriefing()

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/briefing")
        XCTAssertEqual(briefing.briefing, "Three moments today.")
        XCTAssertNil(briefing.error)
    }

    func test_submitFeedback_postsMessageShape() async throws {
        respond(json: #"{"status": "received", "event_id": "evt-1"}"#)

        let client = makeClient()
        try await client.submitFeedback(message: "loving the Now tab")

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/feedback")
        let decoded = try JSONSerialization.jsonObject(with: MockURLProtocol.lastBody ?? Data()) as? [String: Any]
        XCTAssertEqual(decoded?["message"] as? String, "loving the Now tab")
    }

    func test_updatePreference_sendsKeyValuePayload() async throws {
        respond(json: #"{"status": "updated"}"#)

        let client = makeClient()
        try await client.updatePreference(key: "quiet_hours_start", value: AnyCodable("22:00"))

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/preferences")
        let decoded = try JSONSerialization.jsonObject(with: MockURLProtocol.lastBody ?? Data()) as? [String: Any]
        XCTAssertEqual(decoded?["key"] as? String, "quiet_hours_start")
        XCTAssertEqual(decoded?["value"] as? String, "22:00")
    }

    func test_submitContextEvent_postsToContextEventPath() async throws {
        respond(json: #"{"status": "received", "event_id": "evt-2"}"#)

        let client = makeClient()
        let event = ContextEvent(
            type: "context.location",
            source: "ios_app",
            timestamp: "2026-04-22T12:00:00Z",
            payload: ContextPayload(
                latitude: 37.7749,
                longitude: -122.4194,
                altitude: nil,
                horizontalAccuracy: nil,
                speed: nil,
                placeName: "Mission",
                placeType: "neighbourhood",
                deviceName: nil,
                deviceType: nil,
                signalStrength: nil,
                isConnected: nil,
                localTime: nil,
                timezone: nil,
                dayOfWeek: nil,
                isWeekend: nil,
                activity: nil,
                confidence: nil
            ),
            metadata: nil
        )
        try await client.submitContextEvent(event)

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/context/event")
        XCTAssertEqual(MockURLProtocol.lastRequest?.httpMethod, "POST")
    }

    func test_submitContextBatch_wrapsEventsUnderEventsKey() async throws {
        respond(json: #"{"status": "received", "count": 1, "event_ids": ["evt-3"]}"#)

        let client = makeClient()
        let event = ContextEvent(
            type: "context.time",
            source: "ios_app",
            timestamp: "2026-04-22T12:00:00Z",
            payload: ContextPayload(
                latitude: nil, longitude: nil, altitude: nil, horizontalAccuracy: nil,
                speed: nil, placeName: nil, placeType: nil, deviceName: nil,
                deviceType: nil, signalStrength: nil, isConnected: nil,
                localTime: "12:00", timezone: "UTC", dayOfWeek: "Tue",
                isWeekend: false, activity: nil, confidence: nil
            ),
            metadata: nil
        )
        try await client.submitContextBatch([event])

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/context/batch")
        let decoded = try JSONSerialization.jsonObject(with: MockURLProtocol.lastBody ?? Data()) as? [String: Any]
        let events = decoded?["events"] as? [[String: Any]]
        XCTAssertEqual(events?.count, 1)
        XCTAssertEqual(events?.first?["type"] as? String, "context.time")
    }

    func test_getContextSummary_decodesFreeForm() async throws {
        respond(json: #"{"type": "context_summary", "content": "x", "locations": [], "devices": []}"#)

        let client = makeClient()
        let summary = try await client.getContextSummary()

        XCTAssertEqual(MockURLProtocol.lastRequest?.url?.path, "/api/context/summary")
        XCTAssertEqual(summary["content"]?.value as? String, "x")
    }
}

// MARK: - Fixture helpers

private func momentJSON(state: String, snoozeUntil: Int? = nil) -> String {
    let snoozeLine: String
    if let s = snoozeUntil {
        snoozeLine = String(s)
    } else {
        snoozeLine = "null"
    }
    return """
    {
      "id": "m-1",
      "created_at": 1777200000,
      "expires_at": 1777300000,
      "insight": "sample",
      "evidence": [],
      "evidence_hash": "sha256:abc",
      "proposed_action": {"kind": "draft_message", "params": {}},
      "source_insight_type": "cadence",
      "state": "\(state)",
      "scheduled_for": null,
      "context_trigger": null,
      "snooze_until": \(snoozeLine),
      "confidence": 0.5,
      "feedback_weight": 1.0,
      "state_history": []
    }
    """
}

private func peopleListJSON() -> String {
    return #"""
    {
      "you": {
        "observed_months": 0,
        "interactions_count": 0,
        "confidence_pct": 0,
        "when_at_best": [],
        "how_you_write": [],
        "your_routines": [],
        "drifting": []
      },
      "needs_attention": [],
      "active_this_week": [],
      "total": 0,
      "query": null
    }
    """#
}

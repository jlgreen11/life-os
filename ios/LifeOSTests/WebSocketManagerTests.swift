//
//  WebSocketManagerTests.swift
//  Life OS — v2 WebSocketManager unit tests
//
//  Three behaviours under test, each via the `WebSocketTransport`
//  injection seam (no real sockets are opened):
//
//   1. Decode + route — every supported `WebSocketEvent` case round-
//      trips from a representative JSON fixture and lands on the
//      manager's `onEvent` callback.
//   2. Reconnect — after a transport-side failure, the manager
//      schedules a reconnect using exponential backoff and gives up
//      after `maxReconnectAttempts`.
//   3. Heartbeat — once connected, the manager emits a `{"type":
//      "ping"}` text frame on the configured cadence.
//
//  All callbacks are routed onto a synchronous test queue
//  (`DispatchQueue.main` — the test runs on `XCTestCase`'s main loop)
//  so assertions can read state without explicit waits.
//

import XCTest
@testable import LifeOS

// MARK: - Fake transport

/// In-memory `WebSocketTransport` that hands out `FakeWebSocketTask`
/// instances. Tests grab the task via `tasks.last` and drive it.
final class FakeWebSocketTransport: WebSocketTransport {
    private(set) var tasks: [FakeWebSocketTask] = []
    private(set) var madeURLs: [URL] = []

    func makeTask(url: URL) -> WebSocketTaskProtocol {
        let task = FakeWebSocketTask()
        tasks.append(task)
        madeURLs.append(url)
        return task
    }

    var lastTask: FakeWebSocketTask? { tasks.last }
}

/// Test-time `URLSessionWebSocketTask` stand-in. Captures send calls
/// and lets tests deliver receive results synchronously.
final class FakeWebSocketTask: WebSocketTaskProtocol {
    private(set) var resumeCount = 0
    private(set) var cancelled = false
    private(set) var sentMessages: [URLSessionWebSocketTask.Message] = []
    private var pendingReceive: ((Result<URLSessionWebSocketTask.Message, Error>) -> Void)?

    func resume() {
        resumeCount += 1
    }

    func cancel(with closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        cancelled = true
        // Drain any pending receive — the manager treats that as a
        // failure and may schedule a reconnect (which we don't want
        // when the caller asked for a clean disconnect; the manager
        // tracks `isClosedByCaller` to suppress that).
        if let cb = pendingReceive {
            pendingReceive = nil
            let err = NSError(domain: "FakeWebSocketTask", code: -1)
            cb(.failure(err))
        }
    }

    func send(
        _ message: URLSessionWebSocketTask.Message,
        completionHandler: @escaping @Sendable (Error?) -> Void
    ) {
        sentMessages.append(message)
        completionHandler(nil)
    }

    func receive(
        completionHandler: @escaping @Sendable (Result<URLSessionWebSocketTask.Message, Error>) -> Void
    ) {
        // Tests only ever have one receive in flight at a time —
        // overwriting the slot matches the production flow where the
        // manager re-arms `receive()` after each delivery.
        pendingReceive = completionHandler
    }

    // MARK: Test driver

    /// Deliver a text frame to whatever receive callback the manager
    /// has parked on the task.
    func deliver(text: String) {
        let cb = pendingReceive
        pendingReceive = nil
        cb?(.success(.string(text)))
    }

    /// Surface a transport error, mimicking a dropped connection.
    func deliver(error: Error) {
        let cb = pendingReceive
        pendingReceive = nil
        cb?(.failure(error))
    }

    var sentTextFrames: [String] {
        sentMessages.compactMap { msg in
            if case .string(let s) = msg { return s } else { return nil }
        }
    }
}

// MARK: - Helpers

private func makeManager(
    transport: FakeWebSocketTransport,
    configuration: WebSocketManager.Configuration = .default
) -> WebSocketManager {
    return WebSocketManager(
        baseURL: "http://stub.local",
        transport: transport,
        configuration: configuration,
        callbackQueue: .main
    )
}

/// Spin the run loop briefly so async dispatches drain. The manager
/// hops onto its serial `queue` for state updates and then back to
/// `.main` for callbacks; one short pump is enough.
private func pump(seconds: TimeInterval = 0.05) {
    let exp = XCTestExpectation(description: "pump")
    DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { exp.fulfill() }
    XCTWaiter().wait(for: [exp], timeout: 1.0)
}

// MARK: - Decode + route

final class WebSocketManagerDecodeTests: XCTestCase {

    func test_decodesMomentCreatedAndRoutesToCallback() throws {
        let transport = FakeWebSocketTransport()
        let mgr = makeManager(transport: transport)

        var received: [WebSocketEvent] = []
        mgr.onEvent = { received.append($0) }
        mgr.connect()
        pump()

        let task = try XCTUnwrap(transport.lastTask)
        task.deliver(text: momentCreatedJSON)
        pump()

        XCTAssertEqual(received.count, 1)
        guard case .momentCreated(let moment) = received.first else {
            return XCTFail("expected .momentCreated, got \(String(describing: received.first))")
        }
        XCTAssertEqual(moment.id, "m-1")
        XCTAssertEqual(moment.state, .suggested)
        XCTAssertEqual(moment.sourceInsightType, .cadence)
    }

    func test_decodesMomentStateChangedAndRoutesToCallback() throws {
        let transport = FakeWebSocketTransport()
        let mgr = makeManager(transport: transport)

        var received: [WebSocketEvent] = []
        mgr.onEvent = { received.append($0) }
        mgr.connect()
        pump()

        let task = try XCTUnwrap(transport.lastTask)
        task.deliver(text: momentStateChangedJSON)
        pump()

        guard case .momentStateChanged(let change) = received.first else {
            return XCTFail("expected .momentStateChanged")
        }
        XCTAssertEqual(change.id, "m-1")
        XCTAssertEqual(change.fromState, .suggested)
        XCTAssertEqual(change.toState, .accepted)
        XCTAssertEqual(change.annotation, "user accepted")
    }

    func test_decodesConnectorStatusChangedAndRoutesToCallback() throws {
        let transport = FakeWebSocketTransport()
        let mgr = makeManager(transport: transport)

        var received: [WebSocketEvent] = []
        mgr.onEvent = { received.append($0) }
        mgr.connect()
        pump()

        let task = try XCTUnwrap(transport.lastTask)
        task.deliver(text: connectorStatusChangedJSON)
        pump()

        guard case .connectorStatusChanged(let change) = received.first else {
            return XCTFail("expected .connectorStatusChanged")
        }
        XCTAssertEqual(change.id, "proton")
        XCTAssertEqual(change.status, "ok")
        XCTAssertNil(change.lastError)
    }

    func test_unknownEventTypeDoesNotInvokeCallbackButKeepsConnectionAlive() throws {
        let transport = FakeWebSocketTransport()
        let mgr = makeManager(transport: transport)

        var received: [WebSocketEvent] = []
        mgr.onEvent = { received.append($0) }
        mgr.connect()
        pump()

        let task = try XCTUnwrap(transport.lastTask)
        task.deliver(text: #"{"type": "bogus.future_event", "data": {}}"#)
        pump()

        XCTAssertTrue(received.isEmpty, "unknown events should be silently ignored")
        XCTAssertFalse(task.cancelled, "unknown events should not tear down the socket")
    }

    func test_malformedJSONDoesNotCrashAndKeepsConnectionAlive() throws {
        let transport = FakeWebSocketTransport()
        let mgr = makeManager(transport: transport)

        var received: [WebSocketEvent] = []
        mgr.onEvent = { received.append($0) }
        mgr.connect()
        pump()

        let task = try XCTUnwrap(transport.lastTask)
        task.deliver(text: "not json at all")
        pump()

        XCTAssertTrue(received.isEmpty)
        XCTAssertFalse(task.cancelled)
    }
}

// MARK: - Reconnect logic

final class WebSocketManagerReconnectTests: XCTestCase {

    func test_reconnectsAfterTransportError() throws {
        let transport = FakeWebSocketTransport()
        var config = WebSocketManager.Configuration.default
        config.reconnectInitialDelay = 0.05
        config.reconnectMaxDelay = 0.05
        config.maxReconnectAttempts = 3
        let mgr = makeManager(transport: transport, configuration: config)
        mgr.connect()
        pump()

        XCTAssertEqual(transport.tasks.count, 1)

        // First task fails — manager should schedule a reconnect.
        let firstTask = try XCTUnwrap(transport.lastTask)
        firstTask.deliver(error: NSError(domain: "test", code: 0))
        pump(seconds: 0.2)

        XCTAssertGreaterThanOrEqual(transport.tasks.count, 2,
            "manager should re-create a task on reconnect")
    }

    func test_disconnectStopsAutomaticReconnect() throws {
        let transport = FakeWebSocketTransport()
        var config = WebSocketManager.Configuration.default
        config.reconnectInitialDelay = 0.05
        let mgr = makeManager(transport: transport, configuration: config)
        mgr.connect()
        pump()

        let firstTask = try XCTUnwrap(transport.lastTask)
        // Mark connected so the manager really enters the reconnect
        // path on the next failure.
        firstTask.deliver(text: momentCreatedJSON)
        pump()

        mgr.disconnect()
        pump(seconds: 0.2)

        XCTAssertTrue(firstTask.cancelled)
        // No additional task should have been spun up — disconnect
        // forbids automatic reconnect.
        XCTAssertEqual(transport.tasks.count, 1)
    }

    func test_backoffDelayCurveDoublesUntilCap() {
        var config = WebSocketManager.Configuration.default
        config.reconnectInitialDelay = 1
        config.reconnectMaxDelay = 8
        config.reconnectMultiplier = 2

        XCTAssertEqual(nextReconnectDelay(attempt: 1, configuration: config), 1)
        XCTAssertEqual(nextReconnectDelay(attempt: 2, configuration: config), 2)
        XCTAssertEqual(nextReconnectDelay(attempt: 3, configuration: config), 4)
        XCTAssertEqual(nextReconnectDelay(attempt: 4, configuration: config), 8)
        // Beyond the cap, delay clamps — it does not keep doubling.
        XCTAssertEqual(nextReconnectDelay(attempt: 7, configuration: config), 8)
    }
}

// MARK: - Heartbeat

final class WebSocketManagerHeartbeatTests: XCTestCase {

    func test_pingsTransportOnceConnectedOnConfiguredCadence() throws {
        let transport = FakeWebSocketTransport()
        var config = WebSocketManager.Configuration.default
        config.heartbeatInterval = 0.05
        let mgr = makeManager(transport: transport, configuration: config)
        mgr.connect()
        pump()

        let task = try XCTUnwrap(transport.lastTask)
        // Trigger the connected transition with a real frame so the
        // heartbeat timer kicks in.
        task.deliver(text: momentCreatedJSON)
        pump(seconds: 0.2)

        let pings = task.sentTextFrames.filter { $0.contains("ping") }
        XCTAssertGreaterThanOrEqual(pings.count, 1, "expected at least one heartbeat ping")
    }

    func test_doesNotPingBeforeFirstSuccessfulReceive() throws {
        let transport = FakeWebSocketTransport()
        var config = WebSocketManager.Configuration.default
        config.heartbeatInterval = 0.05
        let mgr = makeManager(transport: transport, configuration: config)
        mgr.connect()
        pump(seconds: 0.2)

        let task = try XCTUnwrap(transport.lastTask)
        XCTAssertTrue(task.sentTextFrames.isEmpty,
            "no heartbeats should fire before the manager confirms the socket is open")
    }
}

// MARK: - Connection-change callback

final class WebSocketManagerConnectionChangeTests: XCTestCase {

    func test_firesOnConnectionChangeTrueOnFirstReceive() throws {
        let transport = FakeWebSocketTransport()
        let mgr = makeManager(transport: transport)
        var transitions: [Bool] = []
        mgr.onConnectionChange = { transitions.append($0) }
        mgr.connect()
        pump()

        let task = try XCTUnwrap(transport.lastTask)
        task.deliver(text: momentCreatedJSON)
        pump()

        XCTAssertEqual(transitions, [true])
    }

    func test_firesOnConnectionChangeFalseAfterDisconnect() throws {
        let transport = FakeWebSocketTransport()
        let mgr = makeManager(transport: transport)
        var transitions: [Bool] = []
        mgr.onConnectionChange = { transitions.append($0) }
        mgr.connect()
        pump()
        let task = try XCTUnwrap(transport.lastTask)
        task.deliver(text: momentCreatedJSON)
        pump()

        mgr.disconnect()
        pump()

        XCTAssertEqual(transitions, [true, false])
    }
}

// MARK: - Fixtures

private let momentCreatedJSON = #"""
{
  "type": "moment.created",
  "data": {
    "id": "m-1",
    "created_at": 1777200000,
    "expires_at": 1777300000,
    "insight": "live push fixture",
    "evidence": [],
    "evidence_hash": "sha256:abc",
    "proposed_action": {"kind": "draft_message", "params": {}},
    "source_insight_type": "cadence",
    "state": "suggested",
    "scheduled_for": null,
    "context_trigger": null,
    "snooze_until": null,
    "confidence": 0.6,
    "feedback_weight": 1.0,
    "state_history": []
  }
}
"""#

private let momentStateChangedJSON = #"""
{
  "type": "moment.state_changed",
  "data": {
    "id": "m-1",
    "from_state": "suggested",
    "to_state": "accepted",
    "ts": 1777200500,
    "annotation": "user accepted"
  }
}
"""#

private let connectorStatusChangedJSON = #"""
{
  "type": "connector.status_changed",
  "data": {
    "id": "proton",
    "status": "ok",
    "last_sync_at": 1777200000,
    "last_error": null
  }
}
"""#

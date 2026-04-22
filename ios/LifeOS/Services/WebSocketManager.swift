//
//  WebSocketManager.swift
//  Life OS — v2 WebSocket client
//
//  Talks to the v2 backend's `/ws` endpoint (see
//  `api/routes/websocket.py` + `core/moment/broadcaster.py`). Decodes
//  incoming text frames as `WebSocketEvent` JSON envelopes (see
//  `Models/WebSocketEvent.swift`), routes them to a single callback,
//  and keeps the socket healthy via:
//
//    • exponential-backoff reconnect on drop (capped delay)
//    • 30s heartbeat ping while connected (defeats idle proxies)
//
//  Testability seams
//  -----------------
//  Production callers use `init(baseURL:)` which wires a default
//  `URLSession`. Tests inject a custom `WebSocketTransport` to drive
//  the receive loop and capture sends without touching the network.
//
//  Threading
//  ---------
//  All public callbacks (`onEvent`, `onConnectionChange`) fire on the
//  main queue so SwiftUI/Combine call sites can update `@Published`
//  state without an extra hop. Internal state mutations stay on the
//  manager's serial queue.
//

import Foundation

// MARK: - Transport abstraction

/// One in-flight WebSocket session — abstracted so tests can sub in
/// a fake. Mirrors the slice of `URLSessionWebSocketTask` we use.
/// Sendable so it can cross queue boundaries inside the manager
/// without tripping Swift 6 sendability checks.
protocol WebSocketTaskProtocol: AnyObject, Sendable {
    func resume()
    func cancel(with closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?)
    func send(_ message: URLSessionWebSocketTask.Message, completionHandler: @escaping @Sendable (Error?) -> Void)
    func receive(completionHandler: @escaping @Sendable (Result<URLSessionWebSocketTask.Message, Error>) -> Void)
}

extension URLSessionWebSocketTask: WebSocketTaskProtocol {}

/// Builds a transport-side task per connect attempt. Tests inject
/// their own factory so they control the receive/send loop end to end.
protocol WebSocketTransport {
    func makeTask(url: URL) -> WebSocketTaskProtocol
}

/// Default URLSession-backed transport. Holds a single session
/// configured to keep WebSocket frames alive on cellular without
/// aggressive timeouts.
final class URLSessionWebSocketTransport: NSObject, WebSocketTransport, URLSessionWebSocketDelegate {
    private let session: URLSession
    var onOpen: (() -> Void)?
    var onClose: ((URLSessionWebSocketTask.CloseCode) -> Void)?

    override init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.waitsForConnectivity = true
        self.session = URLSession(configuration: config)
        super.init()
    }

    func makeTask(url: URL) -> WebSocketTaskProtocol {
        return session.webSocketTask(with: url)
    }

    // URLSessionWebSocketDelegate hooks (delegate is set on the task,
    // not the session, so we leave these unused in the default path —
    // close detection runs through receive() error surfacing instead).
    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didOpenWithProtocol protocol: String?
    ) {
        onOpen?()
    }

    func urlSession(
        _ session: URLSession,
        webSocketTask: URLSessionWebSocketTask,
        didCloseWith closeCode: URLSessionWebSocketTask.CloseCode,
        reason: Data?
    ) {
        onClose?(closeCode)
    }
}

// MARK: - Manager

/// Connects to `<baseURL>/ws`, decodes typed events, and surfaces
/// them via `onEvent`. Reconnects with exponential backoff and pings
/// every 30s while connected.
///
/// Lifecycle:
///
///     let mgr = WebSocketManager(baseURL: "http://lifeos.local:8080")
///     mgr.onEvent = { event in /* route into ViewModel */ }
///     mgr.connect()
///     // ...
///     mgr.disconnect()
final class WebSocketManager: @unchecked Sendable {

    // MARK: Configuration

    /// Defaults are tuned for the local-first deployment (Tailscale-
    /// only, ~1ms RTT to the Mac Mini). The cap of 60s on backoff
    /// matches typical mobile-radio sleep cycles — beyond that, the
    /// OS will likely re-foreground the app and trigger a manual
    /// reconnect anyway.
    struct Configuration {
        var heartbeatInterval: TimeInterval = 30
        var reconnectInitialDelay: TimeInterval = 1
        var reconnectMaxDelay: TimeInterval = 60
        var reconnectMultiplier: Double = 2
        var maxReconnectAttempts: Int = 10

        static let `default` = Configuration()
    }

    // MARK: State

    private let baseURL: String
    private let transport: WebSocketTransport
    private let configuration: Configuration
    private let queue = DispatchQueue(label: "WebSocketManager")
    private let callbackQueue: DispatchQueue
    private let decoder: JSONDecoder

    private var task: WebSocketTaskProtocol?
    private var heartbeatTimer: DispatchSourceTimer?
    private var reconnectWork: DispatchWorkItem?
    private var reconnectAttempts = 0
    private var isConnected = false
    private var isClosedByCaller = false

    /// Fires for every successfully-decoded event from `/ws`. Always
    /// invoked on `callbackQueue` (default: main).
    var onEvent: ((WebSocketEvent) -> Void)?

    /// Fires whenever the open/closed state flips. Always invoked on
    /// `callbackQueue` (default: main).
    var onConnectionChange: ((Bool) -> Void)?

    // MARK: Init

    /// Production initializer — wires `URLSessionWebSocketTransport`.
    convenience init(baseURL: String) {
        self.init(
            baseURL: baseURL,
            transport: URLSessionWebSocketTransport(),
            configuration: .default,
            callbackQueue: .main
        )
    }

    /// Test-only initializer — inject a custom transport, tweak
    /// reconnect timings, redirect callbacks to a synchronous queue.
    init(
        baseURL: String,
        transport: WebSocketTransport,
        configuration: Configuration = .default,
        callbackQueue: DispatchQueue = .main,
        decoder: JSONDecoder = WebSocketEvent.decoder()
    ) {
        let cleaned = baseURL
            .replacingOccurrences(of: "http://", with: "ws://")
            .replacingOccurrences(of: "https://", with: "wss://")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.baseURL = cleaned
        self.transport = transport
        self.configuration = configuration
        self.callbackQueue = callbackQueue
        self.decoder = decoder
    }

    // MARK: Public API

    /// Open a fresh connection. Cancels any in-flight reconnect timer
    /// and starts the receive loop. Idempotent — calling while
    /// already connected is a no-op.
    func connect() {
        queue.async { [weak self] in
            guard let self = self else { return }
            self.isClosedByCaller = false
            guard self.task == nil else { return }
            guard let url = URL(string: "\(self.baseURL)/ws") else { return }

            self.reconnectWork?.cancel()
            self.reconnectWork = nil

            let task = self.transport.makeTask(url: url)
            self.task = task
            task.resume()
            self.startReceiveLoop(on: task)
            // Optimistic open — the transport raises receive errors
            // if the handshake actually fails. The first successful
            // receive(/) callback is when we mark ourselves connected.
        }
    }

    /// Close cleanly and prevent automatic reconnect. Pair with
    /// `connect()` when the user backgrounds the app.
    func disconnect() {
        queue.async { [weak self] in
            guard let self = self else { return }
            self.isClosedByCaller = true
            self.tearDownLocked(notify: true, dueToError: false)
        }
    }

    /// Send a raw text frame (used by heartbeats; exposed so future
    /// upstream-message features — e.g. presence — can hook in).
    func send(_ text: String) {
        queue.async { [weak self] in
            guard let self = self, let task = self.task else { return }
            task.send(.string(text)) { _ in }
        }
    }

    // MARK: - Receive loop

    private func startReceiveLoop(on task: WebSocketTaskProtocol) {
        task.receive { [weak self, weak task] result in
            guard let self = self, let task = task else { return }
            self.queue.async {
                // Stale callback after teardown — drop it.
                guard self.task === task else { return }

                switch result {
                case .success(let message):
                    if !self.isConnected {
                        self.markConnectedLocked()
                    }
                    self.handleMessageLocked(message)
                    self.startReceiveLoop(on: task)
                case .failure:
                    self.tearDownLocked(notify: true, dueToError: true)
                }
            }
        }
    }

    private func handleMessageLocked(_ message: URLSessionWebSocketTask.Message) {
        let data: Data?
        switch message {
        case .string(let text):
            data = text.data(using: .utf8)
        case .data(let bytes):
            data = bytes
        @unknown default:
            data = nil
        }
        guard let payload = data else { return }
        do {
            let event = try decoder.decode(WebSocketEvent.self, from: payload)
            // Skip the synthetic .unknown bucket so call sites only
            // see events they meaningfully handle. (Logging the type
            // could go here if we ever need server-vs-client version
            // visibility.)
            if case .unknown = event { return }
            let cb = onEvent
            callbackQueue.async {
                cb?(event)
            }
        } catch {
            // Malformed frame — likely a server bug or out-of-band
            // debug payload. Don't tear down the connection over it.
            print("WebSocket decode error: \(error)")
        }
    }

    // MARK: - Connection state

    private func markConnectedLocked() {
        guard !isConnected else { return }
        isConnected = true
        reconnectAttempts = 0
        startHeartbeatLocked()
        let cb = onConnectionChange
        callbackQueue.async {
            cb?(true)
        }
    }

    private func tearDownLocked(notify: Bool, dueToError: Bool) {
        let wasConnected = isConnected
        isConnected = false
        stopHeartbeatLocked()
        task?.cancel(with: .goingAway, reason: nil)
        task = nil

        if wasConnected, notify {
            let cb = onConnectionChange
            callbackQueue.async {
                cb?(false)
            }
        }

        if dueToError, !isClosedByCaller {
            scheduleReconnectLocked()
        }
    }

    // MARK: - Heartbeat

    private func startHeartbeatLocked() {
        stopHeartbeatLocked()
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(
            deadline: .now() + configuration.heartbeatInterval,
            repeating: configuration.heartbeatInterval
        )
        timer.setEventHandler { [weak self] in
            self?.sendHeartbeatLocked()
        }
        timer.resume()
        heartbeatTimer = timer
    }

    private func stopHeartbeatLocked() {
        heartbeatTimer?.cancel()
        heartbeatTimer = nil
    }

    private func sendHeartbeatLocked() {
        guard let task = task else { return }
        // Matches the no-op shape on the server: any text is fine,
        // but keep it parseable so `WebSocketEvent.unknown("ping")`
        // is what a (test-only) loopback would see.
        task.send(.string(#"{"type": "ping"}"#)) { _ in }
    }

    // MARK: - Reconnect

    private func scheduleReconnectLocked() {
        guard reconnectAttempts < configuration.maxReconnectAttempts else { return }
        reconnectAttempts += 1
        let delay = nextReconnectDelay(attempt: reconnectAttempts, configuration: configuration)
        let work = DispatchWorkItem { [weak self] in
            self?.connect()
        }
        reconnectWork = work
        queue.asyncAfter(deadline: .now() + delay, execute: work)
    }
}

// MARK: - Free function for unit testing

/// Computes the delay for reconnect attempt `attempt` (1-indexed).
/// Pulled out so the backoff curve can be exercised without standing
/// up a full `WebSocketManager` instance.
func nextReconnectDelay(
    attempt: Int,
    configuration: WebSocketManager.Configuration
) -> TimeInterval {
    guard attempt > 0 else { return configuration.reconnectInitialDelay }
    let exponent = Double(attempt - 1)
    let raw = configuration.reconnectInitialDelay * pow(configuration.reconnectMultiplier, exponent)
    return min(raw, configuration.reconnectMaxDelay)
}

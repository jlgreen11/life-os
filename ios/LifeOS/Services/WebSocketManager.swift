import Foundation

final class WebSocketManager: NSObject {
    private let baseURL: String
    private let apiKey: String?
    private var webSocketTask: URLSessionWebSocketTask?
    private var session: URLSession?
    private var isConnected = false
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 10
    private let connectionTimeout: TimeInterval = 10
    private var connectionTimeoutWork: DispatchWorkItem?

    var onMessage: ((WebSocketMessage) -> Void)?
    var onConnectionChange: ((Bool) -> Void)?

    init(baseURL: String, apiKey: String? = nil) {
        let cleaned = baseURL
            .replacingOccurrences(of: "http://", with: "ws://")
            .replacingOccurrences(of: "https://", with: "wss://")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        self.baseURL = cleaned
        let trimmedKey = apiKey?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.apiKey = (trimmedKey?.isEmpty == false) ? trimmedKey : nil
        super.init()
        self.session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    func connect() {
        // URLSessionWebSocketTask can't reliably set custom handshake headers,
        // so the API key is passed as a query parameter (?api_key=...).
        var urlString = "\(baseURL)/ws"
        if let key = apiKey, let encoded = key.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) {
            urlString += "?api_key=\(encoded)"
        }
        guard let url = URL(string: urlString) else { return }
        webSocketTask = session?.webSocketTask(with: url)
        webSocketTask?.resume()
        reconnectAttempts = 0
        receiveMessage()

        // Schedule a connection timeout — if not connected after 10s, trigger reconnection
        connectionTimeoutWork?.cancel()
        let timeoutWork = DispatchWorkItem { [weak self] in
            guard let self = self, !self.isConnected else { return }
            print("WebSocket connection timeout after \(self.connectionTimeout)s")
            self.handleDisconnect()
        }
        connectionTimeoutWork = timeoutWork
        DispatchQueue.main.asyncAfter(deadline: .now() + connectionTimeout, execute: timeoutWork)
    }

    func disconnect() {
        connectionTimeoutWork?.cancel()
        connectionTimeoutWork = nil
        isConnected = false
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        onConnectionChange?(false)
    }

    func send(_ message: String) {
        let wsMessage = URLSessionWebSocketTask.Message.string(message)
        webSocketTask?.send(wsMessage) { error in
            if let error = error {
                print("WebSocket send error: \(error)")
            }
        }
    }

    private func receiveMessage() {
        webSocketTask?.receive { [weak self] result in
            guard let self = self else { return }
            switch result {
            case .success(let message):
                switch message {
                case .string(let text):
                    self.handleMessage(text)
                case .data(let data):
                    if let text = String(data: data, encoding: .utf8) {
                        self.handleMessage(text)
                    }
                @unknown default:
                    break
                }
                self.receiveMessage()
            case .failure(let error):
                print("WebSocket receive error: \(error)")
                self.handleDisconnect()
            }
        }
    }

    private func handleMessage(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }
        do {
            let message = try JSONDecoder().decode(WebSocketMessage.self, from: data)
            DispatchQueue.main.async {
                self.onMessage?(message)
            }
        } catch {
            print("WebSocket decode error: \(error)")
        }
    }

    private func handleDisconnect() {
        isConnected = false
        DispatchQueue.main.async {
            self.onConnectionChange?(false)
        }
        guard reconnectAttempts < maxReconnectAttempts else { return }
        reconnectAttempts += 1
        let delay = pow(2.0, Double(min(reconnectAttempts, 6)))
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.connect()
        }
    }
}

extension WebSocketManager: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        connectionTimeoutWork?.cancel()
        connectionTimeoutWork = nil
        isConnected = true
        reconnectAttempts = 0
        DispatchQueue.main.async {
            self.onConnectionChange?(true)
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        handleDisconnect()
    }
}

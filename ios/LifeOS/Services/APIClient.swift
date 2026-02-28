import Foundation

actor APIClient {
    private let baseURL: String
    private let session: URLSession
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(baseURL: String) {
        self.baseURL = baseURL.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)
    }

    // MARK: - Health & Status

    func getHealth() async throws -> HealthResponse {
        return try await get("/health")
    }

    func getStatus() async throws -> StatusResponse {
        return try await get("/api/status")
    }

    // MARK: - Command Interface

    func sendCommand(_ command: String) async throws -> CommandResponse {
        // The backend CommandRequest schema expects "text", not "command".
        // Using "command" returns a 422 Unprocessable Entity and silently fails.
        let body = ["text": command]
        return try await post("/api/command", body: body)
    }

    // MARK: - Briefing

    func getBriefing() async throws -> Briefing {
        return try await get("/api/briefing")
    }

    // MARK: - Notifications

    func getNotifications() async throws -> [LifeOSNotification] {
        return try await get("/api/notifications")
    }

    // MARK: - Tasks

    func getTasks() async throws -> [LifeOSTask] {
        return try await get("/api/tasks")
    }

    func createTask(_ title: String, priority: String = "normal") async throws -> LifeOSTask {
        // The correct endpoint is /api/tasks (plural). /api/task returns 404.
        let body: [String: Any] = ["title": title, "priority": priority]
        return try await post("/api/tasks", body: body)
    }

    // MARK: - Search

    func search(_ query: String) async throws -> CommandResponse {
        let body = ["query": query]
        return try await post("/api/search", body: body)
    }

    // MARK: - Feedback

    func submitFeedback(_ feedback: FeedbackPayload) async throws {
        let _: EmptyResponse = try await post("/api/feedback", body: feedback)
    }

    // MARK: - Context Events (phone sensor data -> backend)

    func submitContextEvent(_ event: ContextEvent) async throws {
        let _: EmptyResponse = try await post("/api/context/event", body: event)
    }

    func submitContextBatch(_ events: [ContextEvent]) async throws {
        let body = ["events": events]
        let _: EmptyResponse = try await post("/api/context/batch", body: body)
    }

    func getContextSummary() async throws -> CommandResponse {
        return try await get("/api/context/summary")
    }

    // MARK: - Preferences

    func updatePreferences(_ prefs: [String: String]) async throws {
        let _: EmptyResponse = try await post("/api/preferences", body: prefs)
    }

    // MARK: - HTTP Helpers

    private func get<T: Decodable>(_ path: String) async throws -> T {
        let url = URL(string: "\(baseURL)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return try decoder.decode(T.self, from: data)
    }

    private func post<T: Decodable>(_ path: String, body: some Encodable) async throws -> T {
        let url = URL(string: "\(baseURL)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = try encoder.encode(body)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return try decoder.decode(T.self, from: data)
    }

    private func post(_ path: String, body: [String: Any]) async throws -> Data {
        let url = URL(string: "\(baseURL)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return data
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

// MARK: - Errors & Helpers

enum APIError: LocalizedError {
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
}

struct EmptyResponse: Codable {}

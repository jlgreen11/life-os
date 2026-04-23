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

    /// Create a new task. Only `title` is required; all other fields are optional.
    ///
    /// Server schema (TaskCreateRequest in web/schemas.py):
    ///   - title: str (required)
    ///   - description: Optional[str]
    ///   - domain: Optional[str]
    ///   - priority: str = "normal"
    ///   - due_date: Optional[str]
    func createTask(
        _ title: String,
        description: String? = nil,
        domain: String? = nil,
        priority: String = "normal",
        dueDate: String? = nil
    ) async throws -> [String: Any] {
        // The correct endpoint is /api/tasks (plural). /api/task returns 404.
        // Server returns {"task_id": "<id>"}.
        var body: [String: Any] = ["title": title, "priority": priority]
        if let description = description {
            body["description"] = description
        }
        if let domain = domain {
            body["domain"] = domain
        }
        if let dueDate = dueDate {
            // JSON key is snake_case to match Pydantic model field name
            body["due_date"] = dueDate
        }
        let data = try await post("/api/tasks", body: body)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw APIError.invalidResponse
        }
        return json
    }

    /// Mark a task as completed.
    ///
    /// Server endpoint: POST /api/tasks/{task_id}/complete
    /// Returns {"status": "completed"} on success.
    func completeTask(_ taskId: String) async throws {
        let _: Data = try await post("/api/tasks/\(taskId)/complete", body: [String: Any]())
    }

    /// Partially update a task. Only supplied fields are changed.
    ///
    /// Server schema (TaskUpdateRequest in web/schemas.py):
    ///   - status: Optional[str]
    ///   - priority: Optional[str]
    ///   - due_date: Optional[str]
    ///   - title: Optional[str]
    func updateTask(
        _ taskId: String,
        title: String? = nil,
        status: String? = nil,
        priority: String? = nil,
        dueDate: String? = nil
    ) async throws {
        var body: [String: Any] = [:]
        if let title = title {
            body["title"] = title
        }
        if let status = status {
            body["status"] = status
        }
        if let priority = priority {
            body["priority"] = priority
        }
        if let dueDate = dueDate {
            body["due_date"] = dueDate
        }
        // PATCH /api/tasks/{task_id} — server returns {"status": "updated"}
        let _: Data = try await patch("/api/tasks/\(taskId)", body: body)
    }

    // MARK: - Search

    /// Semantic vector search across ingested events.
    ///
    /// Server schema (SearchRequest in web/schemas.py):
    ///   - query: str (required) — matches "query" key sent here
    ///   - limit: int = 10 (optional, not sent — server default used)
    ///   - filters: Optional[dict] (optional, not sent)
    func search(_ query: String) async throws -> CommandResponse {
        let body = ["query": query]
        return try await post("/api/search", body: body)
    }

    // MARK: - Feedback

    /// Submit explicit user feedback.
    ///
    /// NOTE: Schema mismatch — FeedbackPayload sends {type, target_id, target_type, value}
    /// but the server's FeedbackRequest (web/schemas.py) expects only {message: str}.
    /// The server will return 422 unless FeedbackPayload is updated to match.
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

    /// Update user preferences.
    ///
    /// NOTE: The server endpoint is PUT /api/preferences (not POST), and expects
    /// PreferenceUpdate schema: {key: str, value: Any} — a single key-value pair.
    /// This method sends a POST with a flat dict, which will return 405 Method Not Allowed.
    /// To fix: change to PUT and send one key-value pair at a time.
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
        return try decodeResponse(T.self, from: data, path: path)
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
        return try decodeResponse(T.self, from: data, path: path)
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

    private func patch(_ path: String, body: [String: Any]) async throws -> Data {
        let url = URL(string: "\(baseURL)\(path)")!
        var request = URLRequest(url: url)
        request.httpMethod = "PATCH"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await session.data(for: request)
        try validateResponse(response)
        return data
    }

    /// Decodes a response, wrapping DecodingError with endpoint context for debuggability.
    private func decodeResponse<T: Decodable>(_ type: T.Type, from data: Data, path: String) throws -> T {
        do {
            return try decoder.decode(type, from: data)
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

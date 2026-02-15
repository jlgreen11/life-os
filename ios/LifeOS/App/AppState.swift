import SwiftUI
import Combine

@MainActor
final class AppState: ObservableObject {
    @Published var isConnected = false
    @Published var serverURL: String {
        didSet { UserDefaults.standard.set(serverURL, forKey: "serverURL") }
    }
    @Published var notifications: [LifeOSNotification] = []
    @Published var tasks: [LifeOSTask] = []
    @Published var currentTab: Tab = .dashboard

    enum Tab: String, CaseIterable {
        case dashboard = "Dashboard"
        case chat = "Assistant"
        case context = "Context"
        case settings = "Settings"
    }

    private var apiClient: APIClient?
    private var webSocket: WebSocketManager?
    private var cancellables = Set<AnyCancellable>()

    init() {
        self.serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? "http://localhost:8080"
        setupAPIClient()
    }

    func setupAPIClient() {
        apiClient = APIClient(baseURL: serverURL)
        webSocket?.disconnect()
        webSocket = WebSocketManager(baseURL: serverURL)
        webSocket?.onMessage = { [weak self] message in
            Task { @MainActor in
                self?.handleWebSocketMessage(message)
            }
        }
        checkConnection()
    }

    func checkConnection() {
        guard let client = apiClient else { return }
        Task {
            do {
                let status = try await client.getHealth()
                isConnected = status.status == "ok"
                if isConnected {
                    webSocket?.connect()
                    await refreshData()
                }
            } catch {
                isConnected = false
            }
        }
    }

    func refreshData() async {
        guard let client = apiClient else { return }
        do {
            async let fetchedNotifications = client.getNotifications()
            async let fetchedTasks = client.getTasks()
            let (notifs, taskList) = try await (fetchedNotifications, fetchedTasks)
            self.notifications = notifs
            self.tasks = taskList
        } catch {
            print("Failed to refresh data: \(error)")
        }
    }

    func sendCommand(_ command: String) async -> CommandResponse? {
        guard let client = apiClient else { return nil }
        do {
            return try await client.sendCommand(command)
        } catch {
            print("Command failed: \(error)")
            return nil
        }
    }

    func submitFeedback(_ feedback: FeedbackPayload) async {
        guard let client = apiClient else { return }
        do {
            try await client.submitFeedback(feedback)
        } catch {
            print("Feedback failed: \(error)")
        }
    }

    func submitContextEvent(_ event: ContextEvent) async {
        guard let client = apiClient else { return }
        do {
            try await client.submitContextEvent(event)
        } catch {
            print("Context event submission failed: \(error)")
        }
    }

    private func handleWebSocketMessage(_ message: WebSocketMessage) {
        switch message.type {
        case "notification":
            if let notif = message.notification {
                notifications.insert(notif, at: 0)
            }
        case "task_update":
            if let task = message.task {
                if let idx = tasks.firstIndex(where: { $0.id == task.id }) {
                    tasks[idx] = task
                } else {
                    tasks.insert(task, at: 0)
                }
            }
        case "prediction":
            if let notif = message.notification {
                notifications.insert(notif, at: 0)
            }
        default:
            break
        }
    }
}

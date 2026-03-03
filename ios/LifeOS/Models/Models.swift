import Foundation
import CoreLocation

// MARK: - API Response Models

struct HealthResponse: Codable {
    let status: String
    let uptime: Double?
}

struct CommandResponse: Codable {
    let type: String
    let content: String
    let suggestions: [String]?
    let actions: [ActionItem]?
}

struct ActionItem: Codable, Identifiable {
    let id: String
    let label: String
    let type: String
}

struct StatusResponse: Codable {
    let eventCount: Int?
    let vectorStoreReady: Bool?
    let connectors: [String: ConnectorStatus]?

    enum CodingKeys: String, CodingKey {
        case eventCount = "event_count"
        case vectorStoreReady = "vector_store_ready"
        case connectors
    }
}

struct ConnectorStatus: Codable {
    let connected: Bool
    let lastSync: String?

    enum CodingKeys: String, CodingKey {
        case connected
        case lastSync = "last_sync"
    }
}

// MARK: - Notification Model

struct LifeOSNotification: Codable, Identifiable {
    let id: String
    let title: String
    let body: String
    let priority: String
    let source: String?
    let timestamp: String
    let status: String
    let actions: [ActionItem]?

    var priorityColor: String {
        switch priority {
        case "critical": return "red"
        case "high": return "orange"
        case "normal": return "blue"
        case "low": return "gray"
        default: return "gray"
        }
    }

    var relativeTime: String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = formatter.date(from: timestamp) else {
            // Try without fractional seconds
            formatter.formatOptions = [.withInternetDateTime]
            guard let date = formatter.date(from: timestamp) else { return timestamp }
            return RelativeDateTimeFormatter().localizedString(for: date, relativeTo: Date())
        }
        return RelativeDateTimeFormatter().localizedString(for: date, relativeTo: Date())
    }
}

// MARK: - Task Model

struct LifeOSTask: Codable, Identifiable {
    let id: String
    let title: String
    let description: String?
    let status: String
    let priority: String
    let domain: String?
    let dueDate: String?
    let relatedContacts: [String]?
    let source: String?

    enum CodingKeys: String, CodingKey {
        case id, title, description, status, priority, domain, source
        case dueDate = "due_date"
        case relatedContacts = "related_contacts"
    }

    var isComplete: Bool { status == "completed" }

    var statusIcon: String {
        switch status {
        case "completed": return "checkmark.circle.fill"
        case "in_progress": return "arrow.triangle.2.circlepath"
        case "pending": return "circle"
        default: return "circle"
        }
    }
}

// MARK: - Briefing Model

struct Briefing: Codable {
    let summary: String
    let sections: [BriefingSection]?
}

struct BriefingSection: Codable, Identifiable {
    var id: String { title }
    let title: String
    let content: String
    let priority: String?
}

// MARK: - Context Event (sent from phone to backend)

struct ContextEvent: Codable {
    let type: String
    let source: String
    let timestamp: String
    let payload: ContextPayload
    let metadata: ContextMetadata?
}

struct ContextPayload: Codable {
    // Location fields
    let latitude: Double?
    let longitude: Double?
    let altitude: Double?
    let horizontalAccuracy: Double?
    let speed: Double?
    let placeName: String?
    let placeType: String?

    // Device discovery fields
    let deviceName: String?
    let deviceType: String?
    let signalStrength: Int?
    let isConnected: Bool?

    // Time context fields
    let localTime: String?
    let timezone: String?
    let dayOfWeek: String?
    let isWeekend: Bool?

    // Activity fields
    let activity: String?
    let confidence: Double?

    enum CodingKeys: String, CodingKey {
        case latitude, longitude, altitude, speed, activity, confidence
        case horizontalAccuracy = "horizontal_accuracy"
        case placeName = "place_name"
        case placeType = "place_type"
        case deviceName = "device_name"
        case deviceType = "device_type"
        case signalStrength = "signal_strength"
        case isConnected = "is_connected"
        case localTime = "local_time"
        case timezone
        case dayOfWeek = "day_of_week"
        case isWeekend = "is_weekend"
    }
}

struct ContextMetadata: Codable {
    let deviceModel: String?
    let osVersion: String?
    let batteryLevel: Float?
    let networkType: String?
    let appState: String?

    enum CodingKeys: String, CodingKey {
        case deviceModel = "device_model"
        case osVersion = "os_version"
        case batteryLevel = "battery_level"
        case networkType = "network_type"
        case appState = "app_state"
    }
}

// MARK: - Feedback

struct FeedbackPayload: Codable {
    let type: String
    let targetId: String?
    let targetType: String?
    let value: String?

    enum CodingKeys: String, CodingKey {
        case type
        case targetId = "target_id"
        case targetType = "target_type"
        case value
    }
}

// MARK: - Chat

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    let content: String
    let timestamp: Date
    let suggestions: [String]?
    let actions: [ActionItem]?

    enum Role {
        case user
        case assistant
        case system
    }
}

// MARK: - WebSocket

struct WebSocketMessage: Codable {
    let type: String
    let notification: LifeOSNotification?
    let task: LifeOSTask?
    let data: [String: String]?
}

// MARK: - Nearby Device

struct NearbyDevice: Identifiable, Equatable {
    let id: String
    let name: String
    let type: DeviceType
    var signalStrength: Int
    var lastSeen: Date
    var isConnected: Bool
    var attributedTo: String?

    enum DeviceType: String, Codable {
        case bluetooth = "bluetooth"
        case wifi = "wifi"
        case bonjour = "bonjour"
        case unknown = "unknown"
    }

    var signalLabel: String {
        switch signalStrength {
        case -50...0: return "Strong"
        case -70...(-51): return "Good"
        case -90...(-71): return "Weak"
        default: return "Very Weak"
        }
    }

    var signalIcon: String {
        switch signalStrength {
        case -50...0: return "wifi"
        case -70...(-51): return "wifi"
        case -90...(-71): return "wifi.exclamationmark"
        default: return "wifi.slash"
        }
    }
}

// MARK: - Location Context

struct LocationContext: Equatable {
    let coordinate: CLLocationCoordinate2D
    let altitude: Double
    let speed: Double
    let horizontalAccuracy: Double
    let timestamp: Date
    var placeName: String?
    var placeType: String?
    var wifiSSID: String?

    static func == (lhs: LocationContext, rhs: LocationContext) -> Bool {
        lhs.coordinate.latitude == rhs.coordinate.latitude &&
        lhs.coordinate.longitude == rhs.coordinate.longitude &&
        lhs.timestamp == rhs.timestamp
    }
}

// MARK: - Context Snapshot

struct ContextSnapshot {
    let timestamp: Date
    let location: LocationContext?
    let nearbyDevices: [NearbyDevice]
    let timeContext: TimeContext
    let batteryLevel: Float
    let networkType: String

    struct TimeContext {
        let localTime: Date
        let timezone: TimeZone
        let dayOfWeek: String
        let isWeekend: Bool
        let hourOfDay: Int
        let partOfDay: String

        init() {
            let now = Date()
            let calendar = Calendar.current
            self.localTime = now
            self.timezone = .current
            let weekday = calendar.component(.weekday, from: now)
            let formatter = DateFormatter()
            formatter.dateFormat = "EEEE"
            self.dayOfWeek = formatter.string(from: now)
            self.isWeekend = weekday == 1 || weekday == 7
            self.hourOfDay = calendar.component(.hour, from: now)
            switch hourOfDay {
            case 5..<12: self.partOfDay = "morning"
            case 12..<17: self.partOfDay = "afternoon"
            case 17..<21: self.partOfDay = "evening"
            default: self.partOfDay = "night"
            }
        }
    }
}

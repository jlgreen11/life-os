//
//  Models.swift
//  Life OS — context-pipeline value types
//
//  These types are the wire shape for `POST /api/context/event` and
//  `POST /api/context/batch` (see `APIClient.submitContextEvent` /
//  `submitContextBatch`) plus the in-memory snapshot the
//  `ContextEngine` aggregates from `LocationManager` and
//  `DeviceDiscovery`.
//
//  v2 API response/request types live in `Models/APITypes.swift` and
//  `Models/Moment.swift`; this file is intentionally narrow to the
//  context-ingestion surface so the iOS app keeps shipping mobile
//  sensor data while the rest of the v2 UI is being scaffolded.
//

import Foundation
import CoreLocation

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
        case bluetooth
        case wifi
        case bonjour
        case unknown
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
        lhs.coordinate.latitude == rhs.coordinate.latitude
            && lhs.coordinate.longitude == rhs.coordinate.longitude
            && lhs.timestamp == rhs.timestamp
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

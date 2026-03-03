import Foundation
import CoreLocation
import Combine
import UIKit

/// Central engine that aggregates all contextual signals (location, devices, time, activity)
/// and periodically sends context snapshots to the Life OS backend for AI learning.
@MainActor
final class ContextEngine: ObservableObject {
    @Published var locationManager = LocationManager()
    @Published var deviceDiscovery = DeviceDiscovery()
    @Published var latestSnapshot: ContextSnapshot?
    @Published var isCollecting = false
    @Published var contextEventCount = 0

    private var apiClient: APIClient?
    private var snapshotTimer: Timer?
    private var eventBuffer: [ContextEvent] = []
    private var cancellables = Set<AnyCancellable>()

    /// Interval between automatic context snapshots. Settable via SettingsView slider.
    /// Persisted to UserDefaults under "contextSnapshotInterval".
    var snapshotInterval: TimeInterval = UserDefaults.standard.double(forKey: "contextSnapshotInterval").nonZeroOrDefault(300)
    private let batchSize = 10
    private let maxBufferSize = 100

    init() {
        setupObservers()
    }

    func configure(serverURL: String) {
        apiClient = APIClient(baseURL: serverURL)
    }

    func startCollecting() {
        guard !isCollecting else { return }
        isCollecting = true

        locationManager.requestPermission()
        locationManager.startTracking()
        deviceDiscovery.startDiscovery()

        // Periodic snapshot collection
        snapshotTimer = Timer.scheduledTimer(withTimeInterval: snapshotInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.captureAndSendSnapshot()
            }
        }

        // Capture initial snapshot
        Task { await captureAndSendSnapshot() }
    }

    func stopCollecting() {
        isCollecting = false
        locationManager.stopTracking()
        deviceDiscovery.stopDiscovery()
        snapshotTimer?.invalidate()
        snapshotTimer = nil
        flushBuffer()
    }

    // MARK: - Snapshot Capture

    func captureAndSendSnapshot() async {
        let snapshot = captureSnapshot()
        latestSnapshot = snapshot
        let events = buildContextEvents(from: snapshot)
        for event in events {
            bufferEvent(event)
        }
        await flushBufferIfNeeded()
    }

    func captureSnapshot() -> ContextSnapshot {
        let batteryLevel = UIDevice.current.batteryLevel
        UIDevice.current.isBatteryMonitoringEnabled = true

        let networkType = getNetworkType()

        return ContextSnapshot(
            timestamp: Date(),
            location: locationManager.currentLocation,
            nearbyDevices: deviceDiscovery.nearbyDevices,
            timeContext: ContextSnapshot.TimeContext(),
            batteryLevel: batteryLevel,
            networkType: networkType
        )
    }

    // MARK: - Event Building

    private func buildContextEvents(from snapshot: ContextSnapshot) -> [ContextEvent] {
        var events: [ContextEvent] = []
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime]
        let timestamp = isoFormatter.string(from: snapshot.timestamp)
        let metadata = buildMetadata(snapshot)

        // Location event
        if let loc = snapshot.location {
            events.append(ContextEvent(
                type: "context.location",
                source: "ios_app",
                timestamp: timestamp,
                payload: ContextPayload(
                    latitude: loc.coordinate.latitude,
                    longitude: loc.coordinate.longitude,
                    altitude: loc.altitude,
                    horizontalAccuracy: loc.horizontalAccuracy,
                    speed: loc.speed,
                    placeName: loc.placeName,
                    placeType: loc.placeType,
                    deviceName: nil, deviceType: nil, signalStrength: nil, isConnected: nil,
                    localTime: nil, timezone: nil, dayOfWeek: nil, isWeekend: nil,
                    activity: nil, confidence: nil
                ),
                metadata: metadata
            ))
        }

        // Device events - only send when devices change significantly
        for device in snapshot.nearbyDevices {
            events.append(ContextEvent(
                type: "context.device_nearby",
                source: "ios_app",
                timestamp: timestamp,
                payload: ContextPayload(
                    latitude: nil, longitude: nil, altitude: nil, horizontalAccuracy: nil, speed: nil,
                    placeName: nil, placeType: nil,
                    deviceName: device.name,
                    deviceType: device.type.rawValue,
                    signalStrength: device.signalStrength,
                    isConnected: device.isConnected,
                    localTime: nil, timezone: nil, dayOfWeek: nil, isWeekend: nil,
                    activity: nil, confidence: nil
                ),
                metadata: metadata
            ))
        }

        // Time context event
        let tc = snapshot.timeContext
        let timeFormatter = DateFormatter()
        timeFormatter.dateFormat = "HH:mm:ss"
        events.append(ContextEvent(
            type: "context.time",
            source: "ios_app",
            timestamp: timestamp,
            payload: ContextPayload(
                latitude: nil, longitude: nil, altitude: nil, horizontalAccuracy: nil, speed: nil,
                placeName: nil, placeType: nil,
                deviceName: nil, deviceType: nil, signalStrength: nil, isConnected: nil,
                localTime: timeFormatter.string(from: tc.localTime),
                timezone: tc.timezone.identifier,
                dayOfWeek: tc.dayOfWeek,
                isWeekend: tc.isWeekend,
                activity: tc.partOfDay, confidence: nil
            ),
            metadata: metadata
        ))

        return events
    }

    private func buildMetadata(_ snapshot: ContextSnapshot) -> ContextMetadata {
        ContextMetadata(
            deviceModel: UIDevice.current.model,
            osVersion: UIDevice.current.systemVersion,
            batteryLevel: snapshot.batteryLevel,
            networkType: snapshot.networkType,
            appState: UIApplication.shared.applicationState == .active ? "foreground" : "background"
        )
    }

    // MARK: - Event Buffer

    private func bufferEvent(_ event: ContextEvent) {
        eventBuffer.append(event)
        contextEventCount += 1
        if eventBuffer.count > maxBufferSize {
            eventBuffer.removeFirst(eventBuffer.count - maxBufferSize)
        }
    }

    private func flushBufferIfNeeded() async {
        guard eventBuffer.count >= batchSize else { return }
        await flushBuffer()
    }

    private func flushBuffer() {
        guard !eventBuffer.isEmpty, let client = apiClient else { return }
        let batch = eventBuffer
        eventBuffer.removeAll()
        Task {
            do {
                try await client.submitContextBatch(batch)
            } catch {
                // Re-buffer on failure (at front, capped)
                await MainActor.run {
                    self.eventBuffer.insert(contentsOf: batch, at: 0)
                    if self.eventBuffer.count > self.maxBufferSize {
                        self.eventBuffer = Array(self.eventBuffer.prefix(self.maxBufferSize))
                    }
                }
                print("Failed to flush context buffer: \(error)")
            }
        }
    }

    // MARK: - Helpers

    private func getNetworkType() -> String {
        // Simplified - in production use NWPathMonitor
        return "unknown"
    }

    /// Updates the snapshot interval and restarts the timer if currently collecting.
    /// Persists the new value to UserDefaults.
    func updateSnapshotInterval(_ interval: TimeInterval) {
        snapshotInterval = interval
        UserDefaults.standard.set(interval, forKey: "contextSnapshotInterval")

        // Restart the timer with the new interval if actively collecting
        if isCollecting {
            snapshotTimer?.invalidate()
            snapshotTimer = Timer.scheduledTimer(withTimeInterval: snapshotInterval, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    await self?.captureAndSendSnapshot()
                }
            }
        }
    }

    private func setupObservers() {
        // Listen for significant location changes to trigger immediate context capture
        NotificationCenter.default.publisher(for: .didEnterMonitoredRegion)
            .sink { [weak self] _ in
                Task { @MainActor in
                    await self?.captureAndSendSnapshot()
                }
            }
            .store(in: &cancellables)

        NotificationCenter.default.publisher(for: .didExitMonitoredRegion)
            .sink { [weak self] _ in
                Task { @MainActor in
                    await self?.captureAndSendSnapshot()
                }
            }
            .store(in: &cancellables)

        // App lifecycle
        NotificationCenter.default.publisher(for: UIApplication.willResignActiveNotification)
            .sink { [weak self] _ in
                self?.flushBuffer()
            }
            .store(in: &cancellables)
    }
}

// MARK: - Helpers

private extension Double {
    /// Returns self if non-zero, otherwise returns the provided default value.
    /// Useful for UserDefaults which returns 0.0 for unset keys.
    func nonZeroOrDefault(_ defaultValue: Double) -> Double {
        self != 0 ? self : defaultValue
    }
}

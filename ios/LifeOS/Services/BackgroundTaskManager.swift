import Foundation
import BackgroundTasks
import UIKit

/// Manages background task scheduling for context collection when the app is not active.
final class BackgroundTaskManager {
    static let shared = BackgroundTaskManager()
    static let contextRefreshIdentifier = "com.lifeos.context-refresh"
    static let contextProcessingIdentifier = "com.lifeos.context-processing"

    private init() {}

    func registerBackgroundTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.contextRefreshIdentifier,
            using: nil
        ) { task in
            self.handleContextRefresh(task: task as! BGAppRefreshTask)
        }

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.contextProcessingIdentifier,
            using: nil
        ) { task in
            self.handleContextProcessing(task: task as! BGProcessingTask)
        }
    }

    func scheduleContextRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: Self.contextRefreshIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15 minutes
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Could not schedule context refresh: \(error)")
        }
    }

    func scheduleContextProcessing() {
        let request = BGProcessingTaskRequest(identifier: Self.contextProcessingIdentifier)
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = false
        request.earliestBeginDate = Date(timeIntervalSinceNow: 60 * 60) // 1 hour
        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Could not schedule context processing: \(error)")
        }
    }

    private func handleContextRefresh(task: BGAppRefreshTask) {
        scheduleContextRefresh() // Reschedule

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1

        let operation = ContextRefreshOperation()

        task.expirationHandler = {
            queue.cancelAllOperations()
        }

        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        queue.addOperation(operation)
    }

    private func handleContextProcessing(task: BGProcessingTask) {
        scheduleContextProcessing() // Reschedule

        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1

        let operation = ContextProcessingOperation()

        task.expirationHandler = {
            queue.cancelAllOperations()
        }

        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        queue.addOperation(operation)
    }
}

// MARK: - Background Operations

/// Quick context snapshot during app refresh
final class ContextRefreshOperation: Operation {
    override func main() {
        guard !isCancelled else { return }

        let serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? "http://localhost:8080"
        let client = APIClient(baseURL: serverURL)

        let timeContext = ContextSnapshot.TimeContext()
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime]

        let event = ContextEvent(
            type: "context.background_refresh",
            source: "ios_app_background",
            timestamp: isoFormatter.string(from: Date()),
            payload: ContextPayload(
                latitude: nil, longitude: nil, altitude: nil, horizontalAccuracy: nil, speed: nil,
                placeName: nil, placeType: nil,
                deviceName: nil, deviceType: nil, signalStrength: nil, isConnected: nil,
                localTime: formatter.string(from: timeContext.localTime),
                timezone: timeContext.timezone.identifier,
                dayOfWeek: timeContext.dayOfWeek,
                isWeekend: timeContext.isWeekend,
                activity: "background_refresh", confidence: nil
            ),
            metadata: ContextMetadata(
                deviceModel: UIDevice.current.model,
                osVersion: UIDevice.current.systemVersion,
                batteryLevel: UIDevice.current.batteryLevel,
                networkType: "unknown",
                appState: "background"
            )
        )

        let semaphore = DispatchSemaphore(value: 0)
        Task {
            try? await client.submitContextEvent(event)
            semaphore.signal()
        }
        semaphore.wait()
    }
}

/// Longer context processing during background processing
final class ContextProcessingOperation: Operation {
    override func main() {
        guard !isCancelled else { return }
        // Process accumulated context data, send batched events
        // This runs when the device is plugged in or has sufficient battery
        let serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? "http://localhost:8080"
        let client = APIClient(baseURL: serverURL)

        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.formatOptions = [.withInternetDateTime]

        let event = ContextEvent(
            type: "context.background_processing",
            source: "ios_app_background",
            timestamp: isoFormatter.string(from: Date()),
            payload: ContextPayload(
                latitude: nil, longitude: nil, altitude: nil, horizontalAccuracy: nil, speed: nil,
                placeName: nil, placeType: nil,
                deviceName: nil, deviceType: nil, signalStrength: nil, isConnected: nil,
                localTime: nil, timezone: TimeZone.current.identifier,
                dayOfWeek: nil, isWeekend: nil,
                activity: "background_processing", confidence: nil
            ),
            metadata: ContextMetadata(
                deviceModel: UIDevice.current.model,
                osVersion: UIDevice.current.systemVersion,
                batteryLevel: UIDevice.current.batteryLevel,
                networkType: "unknown",
                appState: "background"
            )
        )

        let semaphore = DispatchSemaphore(value: 0)
        Task {
            try? await client.submitContextEvent(event)
            semaphore.signal()
        }
        semaphore.wait()
    }
}

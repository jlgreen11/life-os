import Foundation
import CoreLocation
import Combine

final class LocationManager: NSObject, ObservableObject {
    private let locationManager = CLLocationManager()
    private let geocoder = CLGeocoder()

    @Published var currentLocation: LocationContext?
    @Published var authorizationStatus: CLAuthorizationStatus = .notDetermined
    @Published var locationHistory: [LocationContext] = []
    @Published var significantLocations: [String: Int] = [:]  // place name -> visit count

    private var lastGeocode: Date?
    private let geocodeInterval: TimeInterval = 60  // throttle reverse geocoding

    override init() {
        super.init()
        locationManager.delegate = self
        locationManager.desiredAccuracy = kCLLocationAccuracyBest
        locationManager.distanceFilter = 50  // meters
        locationManager.allowsBackgroundLocationUpdates = true
        locationManager.pausesLocationUpdatesAutomatically = true
        locationManager.showsBackgroundLocationIndicator = true
        authorizationStatus = locationManager.authorizationStatus
    }

    func requestPermission() {
        locationManager.requestAlwaysAuthorization()
    }

    func startTracking() {
        locationManager.startUpdatingLocation()
        locationManager.startMonitoringSignificantLocationChanges()
    }

    func stopTracking() {
        locationManager.stopUpdatingLocation()
        locationManager.stopMonitoringSignificantLocationChanges()
    }

    func startMonitoringRegion(center: CLLocationCoordinate2D, radius: CLLocationDistance, identifier: String) {
        let region = CLCircularRegion(center: center, radius: radius, identifier: identifier)
        region.notifyOnEntry = true
        region.notifyOnExit = true
        locationManager.startMonitoring(for: region)
    }

    private func reverseGeocode(_ location: CLLocation) {
        let now = Date()
        if let last = lastGeocode, now.timeIntervalSince(last) < geocodeInterval { return }
        lastGeocode = now

        geocoder.reverseGeocodeLocation(location) { [weak self] placemarks, error in
            guard let self = self, let placemark = placemarks?.first else { return }
            DispatchQueue.main.async {
                var ctx = self.currentLocation
                ctx?.placeName = [placemark.name, placemark.locality]
                    .compactMap { $0 }
                    .joined(separator: ", ")
                ctx?.placeType = self.classifyPlace(placemark)
                self.currentLocation = ctx

                if let name = ctx?.placeName {
                    self.significantLocations[name, default: 0] += 1
                }
            }
        }
    }

    private func classifyPlace(_ placemark: CLPlacemark) -> String {
        if let types = placemark.areasOfInterest, !types.isEmpty {
            return types.first ?? "unknown"
        }
        if placemark.name?.lowercased().contains("home") == true { return "home" }
        if placemark.name?.lowercased().contains("work") == true { return "work" }
        return "other"
    }
}

extension LocationManager: CLLocationManagerDelegate {
    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        DispatchQueue.main.async {
            self.authorizationStatus = manager.authorizationStatus
        }
        if manager.authorizationStatus == .authorizedAlways ||
           manager.authorizationStatus == .authorizedWhenInUse {
            startTracking()
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }

        let context = LocationContext(
            coordinate: location.coordinate,
            altitude: location.altitude,
            speed: max(0, location.speed),
            horizontalAccuracy: location.horizontalAccuracy,
            timestamp: location.timestamp,
            placeName: nil,
            placeType: nil,
            wifiSSID: nil
        )

        DispatchQueue.main.async {
            self.currentLocation = context
            self.locationHistory.append(context)
            // Keep last 1000 entries
            if self.locationHistory.count > 1000 {
                self.locationHistory.removeFirst(self.locationHistory.count - 1000)
            }
        }

        reverseGeocode(location)
    }

    func locationManager(_ manager: CLLocationManager, didEnterRegion region: CLRegion) {
        NotificationCenter.default.post(
            name: .didEnterMonitoredRegion,
            object: nil,
            userInfo: ["region": region.identifier]
        )
    }

    func locationManager(_ manager: CLLocationManager, didExitRegion region: CLRegion) {
        NotificationCenter.default.post(
            name: .didExitMonitoredRegion,
            object: nil,
            userInfo: ["region": region.identifier]
        )
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error.localizedDescription)")
    }
}

extension Notification.Name {
    static let didEnterMonitoredRegion = Notification.Name("didEnterMonitoredRegion")
    static let didExitMonitoredRegion = Notification.Name("didExitMonitoredRegion")
}

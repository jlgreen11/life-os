import Foundation
import CoreBluetooth
import Network
import Combine

final class DeviceDiscovery: NSObject, ObservableObject {
    @Published var nearbyDevices: [NearbyDevice] = []
    @Published var isScanning = false

    private var centralManager: CBCentralManager?
    private var browser: NWBrowser?
    private var discoveredPeripherals: [String: (peripheral: CBPeripheral, rssi: Int, lastSeen: Date)] = [:]
    private var cleanupTimer: Timer?

    // Known device mappings (user can configure)
    private var deviceAttributions: [String: String] = [:]  // device name -> person/place

    override init() {
        super.init()
        loadAttributions()
    }

    // MARK: - Bluetooth Scanning

    func startBluetoothScan() {
        centralManager = CBCentralManager(delegate: self, queue: .global(qos: .utility))
    }

    func stopBluetoothScan() {
        centralManager?.stopScan()
        isScanning = false
    }

    // MARK: - Bonjour/Network Discovery

    func startNetworkDiscovery() {
        let params = NWParameters()
        params.includePeerToPeer = true
        browser = NWBrowser(for: .bonjour(type: "_services._dns-sd._udp", domain: nil), using: params)
        browser?.stateUpdateHandler = { state in
            print("Browser state: \(state)")
        }
        browser?.browseResultsChangedHandler = { [weak self] results, changes in
            self?.handleBrowseResults(results)
        }
        browser?.start(queue: .global(qos: .utility))
    }

    func stopNetworkDiscovery() {
        browser?.cancel()
        browser = nil
    }

    // MARK: - Combined Start/Stop

    func startDiscovery() {
        startBluetoothScan()
        startNetworkDiscovery()
        startCleanupTimer()
    }

    func stopDiscovery() {
        stopBluetoothScan()
        stopNetworkDiscovery()
        cleanupTimer?.invalidate()
    }

    // MARK: - Device Attribution

    func attributeDevice(_ deviceId: String, to person: String) {
        deviceAttributions[deviceId] = person
        saveAttributions()
        if let idx = nearbyDevices.firstIndex(where: { $0.id == deviceId }) {
            nearbyDevices[idx].attributedTo = person
        }
    }

    func removeAttribution(_ deviceId: String) {
        deviceAttributions.removeValue(forKey: deviceId)
        saveAttributions()
        if let idx = nearbyDevices.firstIndex(where: { $0.id == deviceId }) {
            nearbyDevices[idx].attributedTo = nil
        }
    }

    // MARK: - Private Helpers

    private func handleBrowseResults(_ results: Set<NWBrowser.Result>) {
        for result in results {
            let name: String
            switch result.endpoint {
            case .service(let svcName, _, _, _):
                name = svcName
            default:
                name = "Unknown Service"
            }

            let device = NearbyDevice(
                id: "bonjour-\(name)",
                name: name,
                type: .bonjour,
                signalStrength: -50,  // bonjour doesn't give RSSI
                lastSeen: Date(),
                isConnected: false,
                attributedTo: deviceAttributions[name]
            )
            updateDevice(device)
        }
    }

    private func updateDevice(_ device: NearbyDevice) {
        DispatchQueue.main.async {
            if let idx = self.nearbyDevices.firstIndex(where: { $0.id == device.id }) {
                self.nearbyDevices[idx].signalStrength = device.signalStrength
                self.nearbyDevices[idx].lastSeen = device.lastSeen
            } else {
                self.nearbyDevices.append(device)
            }
        }
    }

    private func startCleanupTimer() {
        cleanupTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            self?.cleanupStaleDevices()
        }
    }

    private func cleanupStaleDevices() {
        let cutoff = Date().addingTimeInterval(-120)  // 2 minutes
        DispatchQueue.main.async {
            self.nearbyDevices.removeAll { $0.lastSeen < cutoff }
        }
    }

    private func loadAttributions() {
        if let data = UserDefaults.standard.dictionary(forKey: "deviceAttributions") as? [String: String] {
            deviceAttributions = data
        }
    }

    private func saveAttributions() {
        UserDefaults.standard.set(deviceAttributions, forKey: "deviceAttributions")
    }
}

// MARK: - Bluetooth Delegate

extension DeviceDiscovery: CBCentralManagerDelegate {
    func centralManagerDidUpdateState(_ central: CBCentralManager) {
        switch central.state {
        case .poweredOn:
            DispatchQueue.main.async { self.isScanning = true }
            central.scanForPeripherals(
                withServices: nil,
                options: [CBCentralManagerScanOptionAllowDuplicatesKey: true]
            )
        case .poweredOff, .unauthorized, .unsupported:
            DispatchQueue.main.async { self.isScanning = false }
        default:
            break
        }
    }

    func centralManager(_ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
                         advertisementData: [String: Any], rssi RSSI: NSNumber) {
        let id = peripheral.identifier.uuidString
        let name = peripheral.name ?? advertisementData[CBAdvertisementDataLocalNameKey] as? String ?? "Unknown"
        let rssiValue = RSSI.intValue

        // Skip very weak signals (likely noise)
        guard rssiValue > -100 else { return }

        discoveredPeripherals[id] = (peripheral, rssiValue, Date())

        let device = NearbyDevice(
            id: id,
            name: name,
            type: .bluetooth,
            signalStrength: rssiValue,
            lastSeen: Date(),
            isConnected: peripheral.state == .connected,
            attributedTo: deviceAttributions[name] ?? deviceAttributions[id]
        )
        updateDevice(device)
    }
}

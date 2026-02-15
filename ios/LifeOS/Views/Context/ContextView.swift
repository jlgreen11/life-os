import SwiftUI
import MapKit

struct ContextView: View {
    @EnvironmentObject var contextEngine: ContextEngine
    @State private var selectedSection: ContextSection = .overview
    @State private var showAttributionSheet = false
    @State private var selectedDevice: NearbyDevice?

    enum ContextSection: String, CaseIterable {
        case overview = "Overview"
        case location = "Location"
        case devices = "Devices"
        case timeline = "Timeline"
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Section picker
                Picker("Section", selection: $selectedSection) {
                    ForEach(ContextSection.allCases, id: \.self) { section in
                        Text(section.rawValue).tag(section)
                    }
                }
                .pickerStyle(.segmented)
                .padding()

                ScrollView {
                    switch selectedSection {
                    case .overview:
                        overviewSection
                    case .location:
                        locationSection
                    case .devices:
                        devicesSection
                    case .timeline:
                        timelineSection
                    }
                }
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Context")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(contextEngine.isCollecting ? Color.green : Color.gray)
                            .frame(width: 6, height: 6)
                        Text(contextEngine.isCollecting ? "Live" : "Paused")
                            .font(.caption)
                    }
                }
            }
            .sheet(item: $selectedDevice) { device in
                DeviceAttributionSheet(device: device, discovery: contextEngine.deviceDiscovery)
            }
        }
    }

    // MARK: - Overview

    private var overviewSection: some View {
        VStack(spacing: 16) {
            // Stats grid
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                statCard(
                    icon: "location.fill",
                    label: "Location",
                    value: contextEngine.locationManager.currentLocation?.placeName ?? "Unknown",
                    color: .blue
                )
                statCard(
                    icon: "antenna.radiowaves.left.and.right",
                    label: "Nearby Devices",
                    value: "\(contextEngine.deviceDiscovery.nearbyDevices.count)",
                    color: .purple
                )
                statCard(
                    icon: "clock",
                    label: "Time of Day",
                    value: contextEngine.latestSnapshot?.timeContext.partOfDay.capitalized ?? "...",
                    color: .orange
                )
                statCard(
                    icon: "arrow.up.doc",
                    label: "Events Sent",
                    value: "\(contextEngine.contextEventCount)",
                    color: .green
                )
            }

            // People nearby
            if !attributedDevices.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Label("People Nearby", systemImage: "person.2")
                        .font(.headline)
                    ForEach(attributedDevices) { device in
                        HStack {
                            Image(systemName: "person.circle.fill")
                                .foregroundStyle(.blue)
                            Text(device.attributedTo ?? device.name)
                                .font(.body)
                            Spacer()
                            Text(device.signalLabel)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 2)
                    }
                }
                .padding()
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            // AI is learning banner
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "brain")
                        .foregroundStyle(.pink)
                    Text("AI Learning")
                        .font(.headline)
                }
                Text("Life OS passively learns from your context data: which locations you visit, when you're there, who's around, and your daily patterns. This builds your personal model without ever asking questions.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding()
    }

    // MARK: - Location

    private var locationSection: some View {
        VStack(spacing: 16) {
            // Map
            if let loc = contextEngine.locationManager.currentLocation {
                Map {
                    Marker("You", coordinate: loc.coordinate)
                }
                .frame(height: 250)
                .clipShape(RoundedRectangle(cornerRadius: 12))

                // Location details
                VStack(alignment: .leading, spacing: 8) {
                    Label("Current Location", systemImage: "location.fill")
                        .font(.headline)

                    detailRow("Place", loc.placeName ?? "Unknown")
                    detailRow("Type", loc.placeType ?? "Unknown")
                    detailRow("Latitude", String(format: "%.6f", loc.coordinate.latitude))
                    detailRow("Longitude", String(format: "%.6f", loc.coordinate.longitude))
                    detailRow("Altitude", String(format: "%.1fm", loc.altitude))
                    detailRow("Speed", String(format: "%.1f m/s", loc.speed))
                    detailRow("Accuracy", String(format: "%.1fm", loc.horizontalAccuracy))
                }
                .padding()
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "location.slash")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("Location not available")
                        .font(.headline)
                    Text("Enable location permissions to see context data")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Enable Location") {
                        contextEngine.locationManager.requestPermission()
                    }
                    .buttonStyle(.borderedProminent)
                }
                .padding(40)
            }

            // Frequently visited
            if !contextEngine.locationManager.significantLocations.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Frequent Locations", systemImage: "mappin.and.ellipse")
                        .font(.headline)
                    ForEach(
                        contextEngine.locationManager.significantLocations.sorted(by: { $0.value > $1.value }),
                        id: \.key
                    ) { name, count in
                        HStack {
                            Text(name)
                                .font(.body)
                            Spacer()
                            Text("\(count) visits")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding()
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
        .padding()
    }

    // MARK: - Devices

    private var devicesSection: some View {
        VStack(spacing: 16) {
            HStack {
                Label(
                    contextEngine.deviceDiscovery.isScanning ? "Scanning..." : "Scan paused",
                    systemImage: "antenna.radiowaves.left.and.right"
                )
                .font(.headline)
                Spacer()
                Text("\(contextEngine.deviceDiscovery.nearbyDevices.count) found")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if contextEngine.deviceDiscovery.nearbyDevices.isEmpty {
                VStack(spacing: 12) {
                    Image(systemName: "wifi.slash")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("No nearby devices found")
                        .font(.headline)
                    Text("Make sure Bluetooth is enabled")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(40)
            } else {
                ForEach(contextEngine.deviceDiscovery.nearbyDevices) { device in
                    DeviceRow(device: device) {
                        selectedDevice = device
                        showAttributionSheet = true
                    }
                }
            }
        }
        .padding()
    }

    // MARK: - Timeline

    private var timelineSection: some View {
        VStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Label("Today's Context Timeline", systemImage: "clock.arrow.circlepath")
                    .font(.headline)

                if contextEngine.locationManager.locationHistory.isEmpty {
                    Text("No location history yet. Context is being collected.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding()
                } else {
                    ForEach(Array(contextEngine.locationManager.locationHistory.suffix(20).enumerated()), id: \.offset) { _, loc in
                        HStack(alignment: .top, spacing: 12) {
                            VStack {
                                Circle()
                                    .fill(Color.blue)
                                    .frame(width: 8, height: 8)
                                Rectangle()
                                    .fill(Color.blue.opacity(0.3))
                                    .frame(width: 1)
                            }
                            .frame(width: 8)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(loc.placeName ?? "(\(String(format: "%.4f", loc.coordinate.latitude)), \(String(format: "%.4f", loc.coordinate.longitude)))")
                                    .font(.subheadline)
                                Text(timeString(loc.timestamp))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()
                        }
                    }
                }
            }
            .padding()
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding()
    }

    // MARK: - Helpers

    private func statCard(icon: String, label: String, value: String, color: Color) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: icon)
                .foregroundStyle(color)
                .font(.title3)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func detailRow(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }

    private var attributedDevices: [NearbyDevice] {
        contextEngine.deviceDiscovery.nearbyDevices.filter { $0.attributedTo != nil }
    }

    private func timeString(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

// MARK: - Device Row

struct DeviceRow: View {
    let device: NearbyDevice
    let onAttribute: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: deviceIcon)
                .foregroundStyle(deviceColor)
                .frame(width: 30)

            VStack(alignment: .leading, spacing: 2) {
                Text(device.name)
                    .font(.subheadline)
                    .fontWeight(.medium)
                HStack(spacing: 8) {
                    Text(device.type.rawValue.capitalized)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text("RSSI: \(device.signalStrength)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if device.isConnected {
                        Text("Connected")
                            .font(.caption2)
                            .foregroundStyle(.green)
                    }
                }
                if let person = device.attributedTo {
                    Label(person, systemImage: "person.fill")
                        .font(.caption2)
                        .foregroundStyle(.blue)
                }
            }

            Spacer()

            Button {
                onAttribute()
            } label: {
                Image(systemName: device.attributedTo != nil ? "person.fill.checkmark" : "person.badge.plus")
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private var deviceIcon: String {
        switch device.type {
        case .bluetooth: return "wave.3.right"
        case .wifi: return "wifi"
        case .bonjour: return "network"
        case .unknown: return "questionmark.circle"
        }
    }

    private var deviceColor: Color {
        switch device.type {
        case .bluetooth: return .blue
        case .wifi: return .green
        case .bonjour: return .orange
        case .unknown: return .gray
        }
    }
}

// MARK: - Device Attribution Sheet

struct DeviceAttributionSheet: View {
    let device: NearbyDevice
    let discovery: DeviceDiscovery
    @Environment(\.dismiss) private var dismiss
    @State private var personName = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Device") {
                    LabeledContent("Name", value: device.name)
                    LabeledContent("Type", value: device.type.rawValue.capitalized)
                    LabeledContent("Signal", value: device.signalLabel)
                }

                Section("Attribution") {
                    TextField("Person or place name", text: $personName)
                    Text("Associate this device with a person or location. Life OS will use this to understand who is nearby.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section {
                    Button("Save Attribution") {
                        if !personName.trimmingCharacters(in: .whitespaces).isEmpty {
                            discovery.attributeDevice(device.id, to: personName)
                        }
                        dismiss()
                    }
                    .disabled(personName.trimmingCharacters(in: .whitespaces).isEmpty)

                    if device.attributedTo != nil {
                        Button("Remove Attribution", role: .destructive) {
                            discovery.removeAttribution(device.id)
                            dismiss()
                        }
                    }
                }
            }
            .navigationTitle("Attribute Device")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onAppear {
                personName = device.attributedTo ?? ""
            }
        }
    }
}

extension NearbyDevice: @retroactive Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

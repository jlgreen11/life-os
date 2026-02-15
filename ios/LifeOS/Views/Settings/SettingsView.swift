import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var contextEngine: ContextEngine
    @State private var serverURL: String = ""
    @State private var contextCollectionEnabled = true
    @State private var locationTrackingEnabled = true
    @State private var deviceDiscoveryEnabled = true
    @State private var contextInterval: Double = 300
    @State private var showingResetAlert = false

    var body: some View {
        NavigationStack {
            Form {
                // Connection
                Section("Server Connection") {
                    TextField("Server URL", text: $serverURL)
                        .textContentType(.URL)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                    HStack {
                        Circle()
                            .fill(appState.isConnected ? Color.green : Color.red)
                            .frame(width: 8, height: 8)
                        Text(appState.isConnected ? "Connected" : "Disconnected")
                            .font(.caption)
                    }
                    Button("Connect") {
                        appState.serverURL = serverURL
                        appState.setupAPIClient()
                        contextEngine.configure(serverURL: serverURL)
                    }
                    Button("Test Connection") {
                        appState.checkConnection()
                    }
                }

                // Context Collection
                Section("Context Collection") {
                    Toggle("Enable Context Collection", isOn: $contextCollectionEnabled)
                        .onChange(of: contextCollectionEnabled) { _, enabled in
                            if enabled {
                                contextEngine.startCollecting()
                            } else {
                                contextEngine.stopCollecting()
                            }
                        }

                    Toggle("Location Tracking", isOn: $locationTrackingEnabled)
                        .onChange(of: locationTrackingEnabled) { _, enabled in
                            if enabled {
                                contextEngine.locationManager.startTracking()
                            } else {
                                contextEngine.locationManager.stopTracking()
                            }
                        }

                    Toggle("Device Discovery", isOn: $deviceDiscoveryEnabled)
                        .onChange(of: deviceDiscoveryEnabled) { _, enabled in
                            if enabled {
                                contextEngine.deviceDiscovery.startDiscovery()
                            } else {
                                contextEngine.deviceDiscovery.stopDiscovery()
                            }
                        }

                    VStack(alignment: .leading) {
                        Text("Context Snapshot Interval: \(Int(contextInterval))s")
                            .font(.subheadline)
                        Slider(value: $contextInterval, in: 60...900, step: 60)
                        Text("How often contextual data is captured and sent to the backend.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                // Privacy
                Section("Privacy") {
                    NavigationLink("Location Permissions") {
                        locationPermissionsView
                    }
                    NavigationLink("Data Collection Details") {
                        dataCollectionView
                    }
                }

                // Stats
                Section("Session Stats") {
                    LabeledContent("Context Events Sent", value: "\(contextEngine.contextEventCount)")
                    LabeledContent("Location Updates", value: "\(contextEngine.locationManager.locationHistory.count)")
                    LabeledContent("Devices Discovered", value: "\(contextEngine.deviceDiscovery.nearbyDevices.count)")
                    LabeledContent("Notifications", value: "\(appState.notifications.count)")
                    LabeledContent("Tasks", value: "\(appState.tasks.count)")
                }

                // About
                Section("About") {
                    LabeledContent("App Version", value: "1.0.0")
                    LabeledContent("Backend", value: appState.serverURL)
                    LabeledContent("iOS", value: UIDevice.current.systemVersion)
                }

                // Reset
                Section {
                    Button("Reset All Data", role: .destructive) {
                        showingResetAlert = true
                    }
                }
            }
            .navigationTitle("Settings")
            .onAppear {
                serverURL = appState.serverURL
            }
            .alert("Reset All Data?", isPresented: $showingResetAlert) {
                Button("Reset", role: .destructive) {
                    UserDefaults.standard.removePersistentDomain(forName: Bundle.main.bundleIdentifier ?? "")
                    serverURL = "http://localhost:8080"
                    appState.serverURL = serverURL
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("This will clear all local settings, device attributions, and cached data. Your backend data will not be affected.")
            }
        }
    }

    // MARK: - Location Permissions

    private var locationPermissionsView: some View {
        Form {
            Section {
                HStack {
                    Text("Status")
                    Spacer()
                    Text(locationStatusText)
                        .foregroundStyle(locationStatusColor)
                }
            }
            Section {
                Text("Life OS uses your location to understand where you are throughout the day. This helps the AI learn your routines, suggest actions based on where you are, and associate places with activities.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section {
                Button("Open System Settings") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
            }
        }
        .navigationTitle("Location Permissions")
    }

    private var locationStatusText: String {
        switch contextEngine.locationManager.authorizationStatus {
        case .authorizedAlways: return "Always"
        case .authorizedWhenInUse: return "When In Use"
        case .denied: return "Denied"
        case .restricted: return "Restricted"
        case .notDetermined: return "Not Set"
        @unknown default: return "Unknown"
        }
    }

    private var locationStatusColor: Color {
        switch contextEngine.locationManager.authorizationStatus {
        case .authorizedAlways: return .green
        case .authorizedWhenInUse: return .yellow
        default: return .red
        }
    }

    // MARK: - Data Collection

    private var dataCollectionView: some View {
        Form {
            Section("What We Collect") {
                dataRow("Location", "GPS coordinates, place names, visit frequency")
                dataRow("Nearby Devices", "Bluetooth/WiFi device names, signal strength")
                dataRow("Time Context", "Local time, day of week, part of day")
                dataRow("Device Info", "Device model, OS version, battery level")
            }
            Section("How It's Used") {
                Text("All data is sent to your private Life OS backend server. The AI uses this context to:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("- Learn your daily routines and patterns")
                    .font(.caption)
                Text("- Understand who is around you based on their devices")
                    .font(.caption)
                Text("- Provide location-aware suggestions")
                    .font(.caption)
                Text("- Build your temporal profile (morning person, etc.)")
                    .font(.caption)
            }
            Section("Data Storage") {
                Text("All data stays on your server. Nothing is sent to third parties. You control your data completely.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Data Collection")
    }

    private func dataRow(_ title: String, _ description: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
            Text(description)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

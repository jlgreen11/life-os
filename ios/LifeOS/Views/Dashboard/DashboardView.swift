import SwiftUI

struct DashboardView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var contextEngine: ContextEngine
    @State private var briefing: Briefing?
    @State private var isLoadingBriefing = false
    @State private var showCommandBar = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Connection Status
                    connectionBanner

                    // Context Summary Card
                    contextSummaryCard

                    // Quick Actions
                    quickActions

                    // Briefing
                    briefingSection

                    // Active Notifications
                    notificationsSection

                    // Tasks
                    tasksSection
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Life OS")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showCommandBar = true
                    } label: {
                        Image(systemName: "magnifyingglass")
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await appState.refreshData() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                }
            }
            .sheet(isPresented: $showCommandBar) {
                CommandBarSheet()
            }
            .refreshable {
                await appState.refreshData()
            }
        }
    }

    // MARK: - Connection Banner

    private var connectionBanner: some View {
        HStack {
            Circle()
                .fill(appState.isConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(appState.isConnected ? "Connected" : "Disconnected")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            if let snapshot = contextEngine.latestSnapshot {
                Label(snapshot.timeContext.partOfDay.capitalized, systemImage: timeIcon(snapshot.timeContext.partOfDay))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 4)
    }

    // MARK: - Context Summary

    private var contextSummaryCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "location.viewfinder")
                    .foregroundStyle(.blue)
                Text("Current Context")
                    .font(.headline)
                Spacer()
            }

            HStack(spacing: 16) {
                if let loc = contextEngine.locationManager.currentLocation {
                    contextChip(
                        icon: "location.fill",
                        label: loc.placeName ?? "Locating...",
                        color: .blue
                    )
                }

                contextChip(
                    icon: "antenna.radiowaves.left.and.right",
                    label: "\(contextEngine.deviceDiscovery.nearbyDevices.count) nearby",
                    color: .purple
                )

                if let snapshot = contextEngine.latestSnapshot {
                    contextChip(
                        icon: "clock",
                        label: snapshot.timeContext.dayOfWeek.prefix(3).description,
                        color: .orange
                    )
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func contextChip(icon: String, label: String, color: Color) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(color)
            Text(label)
                .font(.caption)
                .lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(color.opacity(0.1))
        .clipShape(Capsule())
    }

    // MARK: - Quick Actions

    private var quickActions: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                quickActionButton(icon: "sun.max", label: "Briefing") {
                    Task { await loadBriefing() }
                }
                quickActionButton(icon: "text.bubble", label: "Ask AI") {
                    appState.currentTab = .chat
                }
                quickActionButton(icon: "checklist", label: "Tasks") {
                    // Scroll to tasks
                }
                quickActionButton(icon: "location.viewfinder", label: "Context") {
                    appState.currentTab = .context
                }
            }
        }
    }

    private func quickActionButton(icon: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.title3)
                Text(label)
                    .font(.caption2)
            }
            .frame(width: 70, height: 60)
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Briefing

    private var briefingSection: some View {
        Group {
            if let briefing = briefing {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "sun.max.fill")
                            .foregroundStyle(.yellow)
                        Text("Today's Briefing")
                            .font(.headline)
                    }
                    Text(briefing.summary)
                        .font(.body)
                        .foregroundStyle(.secondary)

                    if let sections = briefing.sections {
                        ForEach(sections) { section in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(section.title)
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Text(section.content)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(8)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color(.tertiarySystemGroupedBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                .padding()
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
            } else if isLoadingBriefing {
                ProgressView("Loading briefing...")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color(.secondarySystemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    // MARK: - Notifications

    private var notificationsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "bell.fill")
                    .foregroundStyle(.orange)
                Text("Notifications")
                    .font(.headline)
                Spacer()
                Text("\(appState.notifications.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if appState.notifications.isEmpty {
                Text("No notifications")
                    .font(.body)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity)
                    .padding()
            } else {
                ForEach(appState.notifications.prefix(5)) { notif in
                    NotificationRow(notification: notif)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Tasks

    private var tasksSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "checklist")
                    .foregroundStyle(.green)
                Text("Tasks")
                    .font(.headline)
                Spacer()
                Text("\(appState.tasks.filter { !$0.isComplete }.count) active")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if appState.tasks.isEmpty {
                Text("No tasks")
                    .font(.body)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity)
                    .padding()
            } else {
                ForEach(appState.tasks.prefix(5)) { task in
                    TaskRow(task: task)
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Helpers

    private func loadBriefing() async {
        isLoadingBriefing = true
        defer { isLoadingBriefing = false }
        if let response = await appState.sendCommand("briefing") {
            briefing = Briefing(summary: response.content, sections: nil)
        }
    }

    private func timeIcon(_ partOfDay: String) -> String {
        switch partOfDay {
        case "morning": return "sunrise"
        case "afternoon": return "sun.max"
        case "evening": return "sunset"
        case "night": return "moon"
        default: return "clock"
        }
    }
}

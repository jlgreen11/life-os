//
//  SettingsTabView.swift
//  Life OS — Settings tab (connectors + preferences)
//
//  Wireframe (DESIGN.md §Information Architecture):
//
//      ┌ Settings ───────────────────────┐
//      ├ CONNECTORS ─────────────────────┤
//      │ • Proton Mail   Ready · 3m ago ▸│
//      │ ◐ iMessage      Syncing…      ▸ │
//      │ ○ CalDAV        Paused        ▸ │
//      │ ⨯ iOS Context   Error · detail▸ │
//      ├ PREFERENCES ────────────────────┤
//      │ Quiet hours     [ 22:00 – 07:00]│
//      │ Autonomy        [——●——]         │
//      │ Proactivity     [—●——]          │
//      └─────────────────────────────────┘
//
//  Hard rules from DESIGN.md / NEXT_TASKS.md:
//  - Raw credentials are NEVER rendered here — only status-level fields
//    come over the wire (`ConnectorOut` omits secrets) and the row's
//    copy reinforces that.
//  - Tap a connector row → push `ConnectorEditView` (detail pane).
//  - Preferences: quiet hours (two HH:MM text fields), autonomy slider,
//    proactivity slider. No raw JSON, no "developer mode".
//  - Calm tone — "Paused" / "Ready", never "❌ OFFLINE".
//
//  Pure helpers (`statusDot`, `statusLine`, `displayName`, `lastSyncLabel`)
//  are exposed statically so `SettingsTabViewTests` can pin copy without
//  rendering a SwiftUI tree.
//

import SwiftUI

struct SettingsTabView: View {

    // MARK: - Inputs

    /// Connector roster. `@State` so previews can seed different fixtures
    /// and the enable-toggle can mutate without a view model.
    @State private var connectors: [Connector]

    /// Preferences row (quiet hours + sliders). `@State` so slider /
    /// text-field edits land without a view model.
    @State private var preferences: Preferences

    /// Anchor timestamp for "synced Nm ago" labels. Tests freeze this;
    /// the default is `Date()`.
    private let anchor: Date

    init(
        connectors: [Connector] = MockData.connectors,
        preferences: Preferences = MockData.preferences,
        anchor: Date = Date()
    ) {
        _connectors = State(initialValue: connectors)
        _preferences = State(initialValue: preferences)
        self.anchor = anchor
    }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.sectionGap) {
                    connectorsSection
                    preferencesSection
                }
                .padding(.horizontal, Spacing.s4)
                .padding(.vertical, Spacing.s6)
            }
            .background(Color.bgBase.ignoresSafeArea())
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .navigationDestination(for: Connector.self) { connector in
                ConnectorEditView(connector: connector)
            }
        }
    }

    // MARK: - CONNECTORS

    @ViewBuilder
    private var connectorsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("CONNECTORS")
            if connectors.isEmpty {
                emptyState(
                    title: "No connectors configured yet.",
                    subtitle: "Connectors stream mail, messages, calendar, and device signals into the Moment engine."
                )
            } else {
                VStack(spacing: Spacing.s2) {
                    ForEach(connectors.indices, id: \.self) { idx in
                        connectorRow(idx)
                    }
                }
            }
        }
    }

    private func connectorRow(_ index: Int) -> some View {
        let connector = connectors[index]
        return HStack(alignment: .center, spacing: Spacing.s3) {
            NavigationLink(value: connector) {
                HStack(alignment: .center, spacing: Spacing.s3) {
                    Circle()
                        .fill(Self.statusDotColor(for: connector.status))
                        .frame(width: 8, height: 8)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: Spacing.s1) {
                        Text(Self.displayName(for: connector))
                            .font(.body15.weight(FontWeightToken.medium))
                            .foregroundStyle(Color.textPrimary)
                        Text(Self.statusLine(for: connector, anchor: anchor))
                            .font(.meta13)
                            .foregroundStyle(Color.textSecondary)
                    }
                    Spacer(minLength: 0)
                    Image(systemName: "chevron.right")
                        .font(.caption11)
                        .foregroundStyle(Color.textTertiary)
                        .accessibilityHidden(true)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(Self.accessibilityLabel(for: connector, anchor: anchor))

            Toggle(
                "",
                isOn: Binding(
                    get: { connectors[index].enabled },
                    set: { connectors[index] = Self.toggled(connectors[index], enabled: $0) }
                )
            )
            .labelsHidden()
            .tint(Color.primaryAction)
            .accessibilityLabel(Text("Enable \(Self.displayName(for: connector))"))
        }
        .padding(.horizontal, Spacing.s3)
        .padding(.vertical, Spacing.s3)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.md))
    }

    // MARK: - PREFERENCES

    @ViewBuilder
    private var preferencesSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("PREFERENCES")
            VStack(alignment: .leading, spacing: Spacing.s5) {
                quietHoursRow
                autonomyRow
                proactivityRow
            }
            .padding(.horizontal, Spacing.s4)
            .padding(.vertical, Spacing.s5)
            .background(Color.bgRaised)
            .clipShape(RoundedRectangle(cornerRadius: Radius.md))
        }
    }

    @ViewBuilder
    private var quietHoursRow: some View {
        VStack(alignment: .leading, spacing: Spacing.s2) {
            Text("Quiet hours")
                .font(.body15.weight(FontWeightToken.medium))
                .foregroundStyle(Color.textPrimary)
            HStack(spacing: Spacing.s2) {
                hhmmField(
                    label: "start",
                    value: Binding(
                        get: { preferences.quietHoursStart },
                        set: { preferences = Self.withQuietHoursStart(preferences, $0) }
                    )
                )
                Text("–")
                    .font(.body15)
                    .foregroundStyle(Color.textTertiary)
                hhmmField(
                    label: "end",
                    value: Binding(
                        get: { preferences.quietHoursEnd },
                        set: { preferences = Self.withQuietHoursEnd(preferences, $0) }
                    )
                )
            }
            if !Self.isValidTime(preferences.quietHoursStart) || !Self.isValidTime(preferences.quietHoursEnd) {
                Text("Use HH:MM (24-hour).")
                    .font(.meta13)
                    .foregroundStyle(Color.statusWarning)
            }
        }
    }

    private func hhmmField(label: String, value: Binding<String>) -> some View {
        TextField(label, text: value)
            .font(.mono(size: FontSize.t15))
            .foregroundStyle(Color.textPrimary)
            .textFieldStyle(.plain)
            .frame(maxWidth: 72)
            .padding(.horizontal, Spacing.s3)
            .padding(.vertical, Spacing.s2)
            .background(Color.bgSunken)
            .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
            .accessibilityLabel(Text("Quiet hours \(label)"))
    }

    @ViewBuilder
    private var autonomyRow: some View {
        preferenceSlider(
            title: "Autonomy",
            subtitle: "How much the system decides on your behalf.",
            value: Binding(
                get: { preferences.autonomyLevel },
                set: { preferences = Self.withAutonomy(preferences, $0) }
            ),
            accessibilityIdentifier: "autonomy-slider"
        )
    }

    @ViewBuilder
    private var proactivityRow: some View {
        preferenceSlider(
            title: "Proactivity",
            subtitle: "How often new Moments surface without prompt.",
            value: Binding(
                get: { preferences.proactivity },
                set: { preferences = Self.withProactivity(preferences, $0) }
            ),
            accessibilityIdentifier: "proactivity-slider"
        )
    }

    private func preferenceSlider(
        title: String,
        subtitle: String,
        value: Binding<Double>,
        accessibilityIdentifier: String
    ) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s2) {
            HStack(alignment: .firstTextBaseline) {
                Text(title)
                    .font(.body15.weight(FontWeightToken.medium))
                    .foregroundStyle(Color.textPrimary)
                Spacer(minLength: 0)
                Text(Self.percentLabel(for: value.wrappedValue))
                    .font(.mono(size: FontSize.t13))
                    .foregroundStyle(Color.textSecondary)
            }
            Slider(value: value, in: 0 ... 1)
                .tint(Color.primaryAction)
                .accessibilityIdentifier(accessibilityIdentifier)
            Text(subtitle)
                .font(.meta13)
                .foregroundStyle(Color.textTertiary)
        }
    }

    // MARK: - Chrome

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption11.weight(FontWeightToken.semibold))
            .tracking(LetterSpacing.caps)
            .foregroundStyle(Color.textTertiary)
            .accessibilityAddTraits(.isHeader)
    }

    private func emptyState(title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s1) {
            Text(title)
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
            Text(subtitle)
                .font(.meta13)
                .foregroundStyle(Color.textTertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Spacing.s4)
        .padding(.vertical, Spacing.s5)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.md))
    }

    // MARK: - Pure helpers (testable without rendering)

    /// Semantic colour for the status dot. Calm-first: `ready`→success,
    /// `syncing`→info, `paused`→tertiary, `error`/`failed`→error. Unknown
    /// values fall back to textTertiary rather than raising.
    static func statusDotColor(for status: String) -> Color {
        switch status.lowercased() {
        case "ready", "ok", "healthy": return .statusSuccess
        case "syncing", "connecting": return .statusInfo
        case "paused", "disabled":    return .textTertiary
        case "error", "failed":       return .statusError
        default:                      return .textTertiary
        }
    }

    /// Display-friendly row name — maps wire `kind` ("proton_mail",
    /// "ios_context") to human names. Falls back to a title-cased
    /// version of `id` when the kind is unknown.
    static func displayName(for connector: Connector) -> String {
        switch connector.kind {
        case "proton_mail": return "Proton Mail"
        case "imessage":    return "iMessage"
        case "caldav":      return "CalDAV"
        case "ios_context": return "iOS Context"
        default:
            return connector.id
                .replacingOccurrences(of: "_", with: " ")
                .split(separator: " ")
                .map { $0.prefix(1).uppercased() + $0.dropFirst() }
                .joined(separator: " ")
        }
    }

    /// Secondary-line copy for a connector row. Mirrors DESIGN.md
    /// "Paused" / "Ready · 3m ago" / "Error · detail" — calm, never
    /// alarmist. `error` rows surface the first line of `lastError`
    /// (truncated at 48 chars) when present.
    static func statusLine(for connector: Connector, anchor: Date) -> String {
        let label = statusWord(for: connector.status)
        switch connector.status.lowercased() {
        case "error", "failed":
            if let msg = connector.lastError?.split(separator: "\n").first {
                let trimmed = msg.trimmingCharacters(in: .whitespaces)
                let short = trimmed.count > 48 ? String(trimmed.prefix(48)) + "…" : String(trimmed)
                return "\(label) · \(short)"
            }
            return label
        case "paused", "disabled":
            return label
        default:
            if let ts = connector.lastSyncAt {
                return "\(label) · \(lastSyncLabel(ts: ts, anchor: anchor))"
            }
            return "\(label) · never synced"
        }
    }

    /// Human label for the status wire word.
    static func statusWord(for status: String) -> String {
        switch status.lowercased() {
        case "ready", "ok", "healthy": return "Ready"
        case "syncing":                return "Syncing"
        case "connecting":             return "Connecting"
        case "paused":                 return "Paused"
        case "disabled":               return "Disabled"
        case "error":                  return "Error"
        case "failed":                 return "Failed"
        default:                       return status.prefix(1).uppercased() + status.dropFirst()
        }
    }

    /// "3m ago" / "2h ago" / "3d ago" recency label from a Unix
    /// timestamp. Future timestamps clamp to "just now" — calm default
    /// rather than showing "in 5s".
    static func lastSyncLabel(ts: Int, anchor: Date) -> String {
        let delta = anchor.timeIntervalSince(Date(timeIntervalSince1970: TimeInterval(ts)))
        if delta < 30 { return "just now" }
        let seconds = Int(delta)
        if seconds < 3_600 {
            return "\(max(1, seconds / 60))m ago"
        }
        if seconds < 86_400 {
            return "\(seconds / 3_600)h ago"
        }
        return "\(seconds / 86_400)d ago"
    }

    /// VoiceOver label for a connector row. Spells out the name +
    /// status word so screen readers announce a sentence rather than
    /// two glued tokens.
    static func accessibilityLabel(for connector: Connector, anchor: Date) -> String {
        "\(displayName(for: connector)). \(statusLine(for: connector, anchor: anchor))"
    }

    /// `HH:MM` (24-hour) validator. Used by the quiet-hours text
    /// fields. Accepts a single-digit minute if zero-padded; rejects
    /// anything out of `00:00 ... 23:59`.
    static func isValidTime(_ raw: String) -> Bool {
        let parts = raw.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return false }
        guard parts[0].count == 2, parts[1].count == 2 else { return false }
        guard let h = Int(parts[0]), let m = Int(parts[1]) else { return false }
        return (0 ... 23).contains(h) && (0 ... 59).contains(m)
    }

    /// Clamp a slider value into `[0, 1]` so stray taps on the track
    /// can't push it out of range.
    static func clampUnit(_ value: Double) -> Double {
        min(1.0, max(0.0, value))
    }

    /// "42%" label for a 0…1 slider value. Rounded half-up.
    static func percentLabel(for value: Double) -> String {
        "\(Int((clampUnit(value) * 100).rounded()))%"
    }

    // MARK: - Pure state helpers (avoid `Binding.map` for testability)

    /// Return a copy of the connector with ``enabled`` replaced. Keeps
    /// the toggle's setter pure — tests drive the setter without a
    /// view model.
    static func toggled(_ connector: Connector, enabled: Bool) -> Connector {
        Connector(
            id: connector.id,
            kind: connector.kind,
            enabled: enabled,
            status: connector.status,
            lastSyncAt: connector.lastSyncAt,
            lastError: connector.lastError
        )
    }

    static func withQuietHoursStart(_ p: Preferences, _ value: String) -> Preferences {
        Preferences(
            quietHoursStart: value,
            quietHoursEnd: p.quietHoursEnd,
            autonomyLevel: p.autonomyLevel,
            proactivity: p.proactivity
        )
    }

    static func withQuietHoursEnd(_ p: Preferences, _ value: String) -> Preferences {
        Preferences(
            quietHoursStart: p.quietHoursStart,
            quietHoursEnd: value,
            autonomyLevel: p.autonomyLevel,
            proactivity: p.proactivity
        )
    }

    static func withAutonomy(_ p: Preferences, _ value: Double) -> Preferences {
        Preferences(
            quietHoursStart: p.quietHoursStart,
            quietHoursEnd: p.quietHoursEnd,
            autonomyLevel: clampUnit(value),
            proactivity: p.proactivity
        )
    }

    static func withProactivity(_ p: Preferences, _ value: Double) -> Preferences {
        Preferences(
            quietHoursStart: p.quietHoursStart,
            quietHoursEnd: p.quietHoursEnd,
            autonomyLevel: p.autonomyLevel,
            proactivity: clampUnit(value)
        )
    }
}

// MARK: - Connector + convenience

/// The decoded `Connector` type has no memberwise init (custom
/// `init(from decoder:)`). Re-expose one for the toggle helper above
/// and for tests seeding fixtures.
extension Connector {
    init(
        id: String,
        kind: String,
        enabled: Bool,
        status: String,
        lastSyncAt: Int?,
        lastError: String?
    ) {
        self.id = id
        self.kind = kind
        self.enabled = enabled
        self.status = status
        self.lastSyncAt = lastSyncAt
        self.lastError = lastError
    }
}

/// `NavigationLink(value:)` requires `Hashable` on the route. Connector
/// conforms to `Equatable` for free via synthesis; we extend it to
/// `Hashable` so the pushed route round-trips through `NavigationPath`.
extension Connector: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

// MARK: - Previews

#Preview("Populated") {
    SettingsTabView(
        connectors: MockData.connectors,
        preferences: MockData.preferences,
        anchor: MockData.anchorDate
    )
    .preferredColorScheme(.dark)
}

#Preview("Empty install") {
    SettingsTabView(
        connectors: MockData.emptyConnectors,
        preferences: MockData.defaultPreferences,
        anchor: MockData.anchorDate
    )
    .preferredColorScheme(.dark)
}

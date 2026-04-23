//
//  ConnectorEditView.swift
//  Life OS — Settings detail pane (per-connector edit form)
//
//  Pushed by `SettingsTabView` when the user taps a connector row.
//  Mirrors the web Settings detail pane (`web/templates/partials/
//  connector_edit_form.html`) — non-secret config fields shown as
//  editable text, secret fields shown as "••• saved" placeholders
//  with empty inputs for updates, a status header, and an enable
//  toggle.
//
//  Hard rules from DESIGN.md / NEXT_TASKS.md:
//  - Raw credentials are NEVER displayed. The wire `ConnectorOut` doesn't
//    carry secret values; the form surfaces only the *names* of secret
//    fields and expects the user to re-enter plaintext (which the server
//    re-encrypts). A blank submission means "keep existing".
//  - Single primary action: `[Save changes]`. Disabled until the form
//    carries at least one valid change.
//  - Calm tone — no "⚠ DANGER". An error is "Error · <detail>".
//  - Empty / nothing-to-edit states render the warm fallback copy from
//    DESIGN.md § "Empty states".
//
//  Validation (pure `Self.canSave(...)` helper, exercised by tests):
//  - `hasChanges` → enabled differs OR any config/secret draft is
//    non-whitespace.
//  - `allEditedValuesValid` → every populated draft trims to a
//    non-empty string (blank is fine — blank means "no change").
//

import SwiftUI

struct ConnectorEditView: View {

    // MARK: - Inputs

    /// Connector being edited. Read-only — edits land in the three
    /// drafts below and are committed via the save action (which a
    /// follow-up ViewModels task wires to `APIClient.updateConnector`).
    let connector: Connector

    /// Key list for non-secret config fields. Seeded from the server's
    /// `edit_view` response in production; stubbed from the connector's
    /// kind in the preview. Keys are stable wire names (snake_case).
    private let configKeys: [String]

    /// Key list for secret fields — names only, no values, per the
    /// Fernet boundary rule.
    private let secretKeys: [String]

    /// Draft config values keyed by field name. Empty string = the user
    /// hasn't typed a new value for that key yet.
    @State private var configDraft: [String: String]

    /// Draft secret values. Blank = "keep existing"; non-blank = new
    /// plaintext that the server should re-encrypt.
    @State private var secretDraft: [String: String]

    /// Draft enable flag; starts from the connector's persisted state.
    @State private var enabledDraft: Bool

    init(
        connector: Connector,
        configKeys: [String]? = nil,
        secretKeys: [String]? = nil
    ) {
        self.connector = connector
        let resolvedConfigKeys = configKeys ?? Self.defaultConfigKeys(for: connector.kind)
        let resolvedSecretKeys = secretKeys ?? Self.defaultSecretKeys(for: connector.kind)
        self.configKeys = resolvedConfigKeys
        self.secretKeys = resolvedSecretKeys
        _configDraft = State(
            initialValue: Dictionary(uniqueKeysWithValues: resolvedConfigKeys.map { ($0, "") })
        )
        _secretDraft = State(
            initialValue: Dictionary(uniqueKeysWithValues: resolvedSecretKeys.map { ($0, "") })
        )
        _enabledDraft = State(initialValue: connector.enabled)
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.sectionGap) {
                headerCard
                enableSection
                configSection
                secretsSection
                saveBar
            }
            .padding(.horizontal, Spacing.s4)
            .padding(.vertical, Spacing.s6)
        }
        .background(Color.bgBase.ignoresSafeArea())
        .navigationTitle(SettingsTabView.displayName(for: connector))
        .navigationBarTitleDisplayMode(.inline)
    }

    // MARK: - Header

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: Spacing.s2) {
            HStack(spacing: Spacing.s2) {
                Circle()
                    .fill(SettingsTabView.statusDotColor(for: connector.status))
                    .frame(width: 8, height: 8)
                    .accessibilityHidden(true)
                Text(SettingsTabView.statusWord(for: connector.status))
                    .font(.body15.weight(FontWeightToken.medium))
                    .foregroundStyle(Color.textPrimary)
                Spacer()
                Text("id: \(connector.id)")
                    .font(.mono(size: FontSize.t13))
                    .foregroundStyle(Color.textTertiary)
            }
            Text(Self.headerSubtitle(for: connector))
                .font(.meta13)
                .foregroundStyle(Color.textSecondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Spacing.s4)
        .padding(.vertical, Spacing.s4)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.md))
    }

    // MARK: - Enable toggle

    private var enableSection: some View {
        HStack {
            VStack(alignment: .leading, spacing: Spacing.s1) {
                Text("Enabled")
                    .font(.body15.weight(FontWeightToken.medium))
                    .foregroundStyle(Color.textPrimary)
                Text("Pausing stops new events without removing stored config.")
                    .font(.meta13)
                    .foregroundStyle(Color.textTertiary)
            }
            Spacer()
            Toggle("", isOn: $enabledDraft)
                .labelsHidden()
                .tint(Color.primaryAction)
                .accessibilityLabel(Text("Enable connector"))
        }
        .padding(.horizontal, Spacing.s4)
        .padding(.vertical, Spacing.s4)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.md))
    }

    // MARK: - Configuration

    @ViewBuilder
    private var configSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("CONFIGURATION")
            if configKeys.isEmpty {
                emptyState(
                    title: "No configuration required.",
                    subtitle: "This connector pulls everything it needs from the host environment."
                )
            } else {
                VStack(alignment: .leading, spacing: Spacing.s4) {
                    ForEach(configKeys, id: \.self) { key in
                        configField(key)
                    }
                }
                .padding(.horizontal, Spacing.s4)
                .padding(.vertical, Spacing.s5)
                .background(Color.bgRaised)
                .clipShape(RoundedRectangle(cornerRadius: Radius.md))
            }
        }
    }

    private func configField(_ key: String) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s1) {
            Text(Self.labelForKey(key))
                .font(.body15.weight(FontWeightToken.medium))
                .foregroundStyle(Color.textPrimary)
            TextField("", text: binding(for: key, in: .configDraft))
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
                .textFieldStyle(.plain)
                .padding(.horizontal, Spacing.s3)
                .padding(.vertical, Spacing.s2)
                .background(Color.bgSunken)
                .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
                .accessibilityLabel(Text(Self.labelForKey(key)))
            if !Self.isValidDraftValue(configDraft[key] ?? "") {
                Text("Value cannot be whitespace only.")
                    .font(.meta13)
                    .foregroundStyle(Color.statusWarning)
            }
        }
    }

    // MARK: - Secrets

    @ViewBuilder
    private var secretsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("CREDENTIALS")
            if secretKeys.isEmpty {
                emptyState(
                    title: "No credentials to manage.",
                    subtitle: "This connector stores nothing that needs re-entry."
                )
            } else {
                VStack(alignment: .leading, spacing: Spacing.s4) {
                    Text("Existing credentials stay encrypted at rest. Type a new value to rotate; leave blank to keep the saved one.")
                        .font(.meta13)
                        .foregroundStyle(Color.textTertiary)
                    ForEach(secretKeys, id: \.self) { key in
                        secretField(key)
                    }
                }
                .padding(.horizontal, Spacing.s4)
                .padding(.vertical, Spacing.s5)
                .background(Color.bgRaised)
                .clipShape(RoundedRectangle(cornerRadius: Radius.md))
            }
        }
    }

    private func secretField(_ key: String) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s1) {
            HStack(alignment: .firstTextBaseline) {
                Text(Self.labelForKey(key))
                    .font(.body15.weight(FontWeightToken.medium))
                    .foregroundStyle(Color.textPrimary)
                Spacer()
                Text("••• saved")
                    .font(.mono(size: FontSize.t13))
                    .foregroundStyle(Color.textTertiary)
                    .accessibilityHidden(true)
            }
            SecureField("Replace", text: binding(for: key, in: .secretDraft))
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
                .textFieldStyle(.plain)
                .padding(.horizontal, Spacing.s3)
                .padding(.vertical, Spacing.s2)
                .background(Color.bgSunken)
                .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
                .accessibilityLabel(Text("\(Self.labelForKey(key)) (replace)"))
        }
    }

    // MARK: - Save bar

    @ViewBuilder
    private var saveBar: some View {
        let canSave = Self.canSave(
            connector: connector,
            enabledDraft: enabledDraft,
            configDraft: configDraft,
            secretDraft: secretDraft
        )
        HStack {
            if Self.hasChanges(
                connector: connector,
                enabledDraft: enabledDraft,
                configDraft: configDraft,
                secretDraft: secretDraft
            ) && !canSave {
                Text("Fix highlighted fields to save.")
                    .font(.meta13)
                    .foregroundStyle(Color.statusWarning)
            }
            Spacer()
            Button("Save changes") {
                // Wiring to APIClient.updateConnector lands in the
                // ViewModels task further down NEXT_TASKS.md.
            }
            .font(.body15.weight(FontWeightToken.medium))
            .foregroundStyle(canSave ? Color.white : Color.textDisabled)
            .padding(.horizontal, Spacing.s3)
            .padding(.vertical, Spacing.s2)
            .background(canSave ? Color.primaryAction : Color.bgRaised)
            .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
            .disabled(!canSave)
            .accessibilityLabel(Text("Save connector changes"))
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

    // MARK: - Binding helpers

    /// Dictionary-keyed binding that preserves the dict when the key is
    /// missing (e.g., first-time edit). Used by every config / secret
    /// input.
    private func binding(
        for key: String,
        in keyPath: ReferenceKeyPath
    ) -> Binding<String> {
        switch keyPath {
        case .configDraft:
            return Binding(
                get: { configDraft[key] ?? "" },
                set: { configDraft[key] = $0 }
            )
        case .secretDraft:
            return Binding(
                get: { secretDraft[key] ?? "" },
                set: { secretDraft[key] = $0 }
            )
        }
    }

    private enum ReferenceKeyPath { case configDraft, secretDraft }

    // MARK: - Pure helpers (testable without rendering)

    /// Default non-secret config keys per connector kind. In production
    /// the server provides these via `edit_view(id)`; the stubbed list
    /// here keeps the preview / tests deterministic.
    static func defaultConfigKeys(for kind: String) -> [String] {
        switch kind {
        case "proton_mail": return ["username", "mailbox_folder"]
        case "imessage":    return ["chat_db_path"]
        case "caldav":      return ["server_url", "calendar_id"]
        case "ios_context": return []
        default:            return []
        }
    }

    /// Default secret keys per connector kind. The values are never
    /// known to the client — only the names.
    static func defaultSecretKeys(for kind: String) -> [String] {
        switch kind {
        case "proton_mail": return ["password", "bridge_token"]
        case "imessage":    return []
        case "caldav":      return ["password"]
        case "ios_context": return ["device_push_token"]
        default:            return []
        }
    }

    /// Title-cased label for a snake_case field key ("mailbox_folder" →
    /// "Mailbox folder"). Used by every form row.
    static func labelForKey(_ key: String) -> String {
        let parts = key.split(separator: "_").map(String.init)
        guard let first = parts.first else { return key }
        let head = first.prefix(1).uppercased() + first.dropFirst()
        return ([head] + parts.dropFirst().map { $0.lowercased() }).joined(separator: " ")
    }

    /// Returns true when a draft value is either empty (meaning
    /// "unchanged") or contains non-whitespace content. Pure
    /// whitespace is the only invalid case — we don't want to let
    /// the user save a config field to `"   "`.
    static func isValidDraftValue(_ value: String) -> Bool {
        value.isEmpty || !value.trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Header subtitle line — surfaces `lastError` when the status is
    /// `error`, otherwise renders a calm "Last sync …" sentence.
    static func headerSubtitle(for connector: Connector) -> String {
        switch connector.status.lowercased() {
        case "error", "failed":
            if let err = connector.lastError, !err.isEmpty {
                return err
            }
            return "Connector reported an error on the last sync."
        default:
            if let ts = connector.lastSyncAt {
                let date = Date(timeIntervalSince1970: TimeInterval(ts))
                let fmt = DateFormatter()
                fmt.dateFormat = "yyyy-MM-dd HH:mm"
                fmt.timeZone = TimeZone(identifier: "UTC")
                return "Last sync \(fmt.string(from: date)) UTC"
            }
            return "No sync recorded yet."
        }
    }

    /// True when the draft differs from the persisted connector in at
    /// least one field (enabled flipped, or any draft value is
    /// non-empty).
    static func hasChanges(
        connector: Connector,
        enabledDraft: Bool,
        configDraft: [String: String],
        secretDraft: [String: String]
    ) -> Bool {
        if enabledDraft != connector.enabled { return true }
        if configDraft.values.contains(where: { !$0.isEmpty }) { return true }
        if secretDraft.values.contains(where: { !$0.isEmpty }) { return true }
        return false
    }

    /// True when every non-empty draft value passes
    /// ``isValidDraftValue`` (i.e., no pure-whitespace fields).
    static func allEditedValuesValid(
        configDraft: [String: String],
        secretDraft: [String: String]
    ) -> Bool {
        for v in configDraft.values where !isValidDraftValue(v) { return false }
        for v in secretDraft.values where !isValidDraftValue(v) { return false }
        return true
    }

    /// Primary save-button gate — ``hasChanges`` AND
    /// ``allEditedValuesValid``. Exposed so tests can lock the
    /// transition matrix.
    static func canSave(
        connector: Connector,
        enabledDraft: Bool,
        configDraft: [String: String],
        secretDraft: [String: String]
    ) -> Bool {
        hasChanges(
            connector: connector,
            enabledDraft: enabledDraft,
            configDraft: configDraft,
            secretDraft: secretDraft
        )
            && allEditedValuesValid(configDraft: configDraft, secretDraft: secretDraft)
    }
}

// MARK: - Previews

#Preview("Proton (populated)") {
    NavigationStack {
        ConnectorEditView(connector: MockData.connectorProton)
    }
    .preferredColorScheme(.dark)
}

#Preview("iOS Context (error)") {
    NavigationStack {
        ConnectorEditView(connector: MockData.connectorIOSContext)
    }
    .preferredColorScheme(.dark)
}

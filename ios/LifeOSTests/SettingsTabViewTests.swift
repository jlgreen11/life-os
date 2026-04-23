//
//  SettingsTabViewTests.swift
//  Life OS — Settings tab + ConnectorEditView logic tests
//
//  Verifies:
//    1. `MockData.connectors` / `MockData.emptyConnectors` and
//       `MockData.preferences` / `MockData.defaultPreferences` match
//       the shape the Settings tab's sections expect.
//    2. `SettingsTabView` pure helpers (`statusDotColor`, `displayName`,
//       `statusLine`, `statusWord`, `lastSyncLabel`, `accessibilityLabel`,
//       `isValidTime`, `clampUnit`, `percentLabel`, `toggled`,
//       `withQuietHoursStart/End`, `withAutonomy`, `withProactivity`)
//       produce deterministic copy under the frozen anchor date.
//    3. `ConnectorEditView` pure helpers (`defaultConfigKeys`,
//       `defaultSecretKeys`, `labelForKey`, `isValidDraftValue`,
//       `headerSubtitle`, `hasChanges`, `allEditedValuesValid`,
//       `canSave`) lock the form-validation truth table.
//    4. `Preferences` round-trips its four wire keys + defaults match
//       `_PREFERENCE_DEFAULTS` in `api/routes/settings.py`.
//    5. `Connector: Hashable` so `NavigationLink(value:)` can
//       round-trip the row through `NavigationPath`.
//

import XCTest
@testable import LifeOS

// MARK: - MockData shape

final class MockSettingsShapeTests: XCTestCase {

    func test_connectors_exerciseEveryStatusBranch() {
        let statuses = Set(MockData.connectors.map(\.status))
        XCTAssertTrue(statuses.contains("ready"))
        XCTAssertTrue(statuses.contains("syncing"))
        XCTAssertTrue(statuses.contains("paused"))
        XCTAssertTrue(statuses.contains("error"))
    }

    func test_connectors_haveUniqueIDs() {
        let ids = MockData.connectors.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    func test_errorConnector_carriesLastError() {
        guard let c = MockData.connectors.first(where: { $0.status == "error" }) else {
            return XCTFail("MockData.connectors must include an error row")
        }
        XCTAssertNotNil(c.lastError)
        XCTAssertFalse(c.lastError?.isEmpty ?? true)
    }

    func test_emptyConnectors_isEmpty() {
        XCTAssertTrue(MockData.emptyConnectors.isEmpty)
    }

    func test_preferences_offsetFromDefaults() {
        XCTAssertNotEqual(MockData.preferences.autonomyLevel, Preferences.defaults.autonomyLevel)
        XCTAssertNotEqual(MockData.preferences.proactivity, Preferences.defaults.proactivity)
    }

    func test_defaultPreferences_matchWireDefaults() {
        // Mirrors `_PREFERENCE_DEFAULTS` in `api/routes/settings.py`.
        XCTAssertEqual(MockData.defaultPreferences.quietHoursStart, "22:00")
        XCTAssertEqual(MockData.defaultPreferences.quietHoursEnd, "07:00")
        XCTAssertEqual(MockData.defaultPreferences.autonomyLevel, 0.5, accuracy: 0.0001)
        XCTAssertEqual(MockData.defaultPreferences.proactivity, 0.5, accuracy: 0.0001)
    }
}

// MARK: - Preferences round-trip

final class PreferencesCodingTests: XCTestCase {

    func test_encodesSnakeCase() throws {
        let p = Preferences(
            quietHoursStart: "21:30",
            quietHoursEnd: "06:45",
            autonomyLevel: 0.7,
            proactivity: 0.2
        )
        let data = try JSONEncoder().encode(p)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["quiet_hours_start"] as? String, "21:30")
        XCTAssertEqual(json["quiet_hours_end"] as? String, "06:45")
        XCTAssertEqual(json["autonomy_level"] as? Double, 0.7)
        XCTAssertEqual(json["proactivity"] as? Double, 0.2)
    }

    func test_decodesMissingKeys_toDefaults() throws {
        let data = Data("{}".utf8)
        let p = try JSONDecoder().decode(Preferences.self, from: data)
        XCTAssertEqual(p, Preferences.defaults)
    }

    func test_defaultsMatchDESIGN_md_andPythonDefaults() {
        XCTAssertEqual(Preferences.defaults.quietHoursStart, "22:00")
        XCTAssertEqual(Preferences.defaults.quietHoursEnd, "07:00")
        XCTAssertEqual(Preferences.defaults.autonomyLevel, 0.5, accuracy: 0.0001)
        XCTAssertEqual(Preferences.defaults.proactivity, 0.5, accuracy: 0.0001)
    }
}

// MARK: - SettingsTabView.statusDotColor + statusWord

final class SettingsStatusHelperTests: XCTestCase {

    func test_statusDotColor_knownBranches() {
        XCTAssertEqual(SettingsTabView.statusDotColor(for: "ready"), .statusSuccess)
        XCTAssertEqual(SettingsTabView.statusDotColor(for: "syncing"), .statusInfo)
        XCTAssertEqual(SettingsTabView.statusDotColor(for: "paused"), .textTertiary)
        XCTAssertEqual(SettingsTabView.statusDotColor(for: "error"), .statusError)
        XCTAssertEqual(SettingsTabView.statusDotColor(for: "failed"), .statusError)
    }

    func test_statusDotColor_isCaseInsensitive() {
        XCTAssertEqual(SettingsTabView.statusDotColor(for: "READY"), .statusSuccess)
    }

    func test_statusDotColor_unknownFallsBackToTertiary() {
        XCTAssertEqual(SettingsTabView.statusDotColor(for: "wat"), .textTertiary)
    }

    func test_statusWord_capitalizesKnownBranches() {
        XCTAssertEqual(SettingsTabView.statusWord(for: "ready"), "Ready")
        XCTAssertEqual(SettingsTabView.statusWord(for: "syncing"), "Syncing")
        XCTAssertEqual(SettingsTabView.statusWord(for: "paused"), "Paused")
        XCTAssertEqual(SettingsTabView.statusWord(for: "error"), "Error")
    }

    func test_statusWord_fallbackCapitalizesFirstLetter() {
        XCTAssertEqual(SettingsTabView.statusWord(for: "throttled"), "Throttled")
    }
}

// MARK: - SettingsTabView.displayName

final class SettingsDisplayNameTests: XCTestCase {

    func test_knownKinds_mapToBrandedNames() {
        XCTAssertEqual(
            SettingsTabView.displayName(for: MockData.connectorProton),
            "Proton Mail"
        )
        XCTAssertEqual(
            SettingsTabView.displayName(for: MockData.connectorIMessage),
            "iMessage"
        )
        XCTAssertEqual(
            SettingsTabView.displayName(for: MockData.connectorCalDAV),
            "CalDAV"
        )
        XCTAssertEqual(
            SettingsTabView.displayName(for: MockData.connectorIOSContext),
            "iOS Context"
        )
    }

    func test_unknownKind_fallsBackToTitleCasedId() {
        let c = Connector(
            id: "plaid_v3",
            kind: "plaid_bank",
            enabled: true,
            status: "ready",
            lastSyncAt: nil,
            lastError: nil
        )
        XCTAssertEqual(SettingsTabView.displayName(for: c), "Plaid V3")
    }
}

// MARK: - SettingsTabView.lastSyncLabel + statusLine

final class SettingsRecencyTests: XCTestCase {

    private let anchor = MockData.anchorDate

    private func ts(secondsAgo: Int) -> Int {
        Int(anchor.timeIntervalSince1970) - secondsAgo
    }

    func test_lastSyncLabel_sub30Seconds_isJustNow() {
        XCTAssertEqual(SettingsTabView.lastSyncLabel(ts: ts(secondsAgo: 5), anchor: anchor), "just now")
    }

    func test_lastSyncLabel_minutes() {
        XCTAssertEqual(SettingsTabView.lastSyncLabel(ts: ts(secondsAgo: 180), anchor: anchor), "3m ago")
    }

    func test_lastSyncLabel_hours() {
        XCTAssertEqual(SettingsTabView.lastSyncLabel(ts: ts(secondsAgo: 7_200), anchor: anchor), "2h ago")
    }

    func test_lastSyncLabel_days() {
        XCTAssertEqual(
            SettingsTabView.lastSyncLabel(ts: ts(secondsAgo: 86_400 * 3), anchor: anchor),
            "3d ago"
        )
    }

    func test_lastSyncLabel_futureClampsToJustNow() {
        XCTAssertEqual(
            SettingsTabView.lastSyncLabel(ts: ts(secondsAgo: -10), anchor: anchor),
            "just now"
        )
    }

    func test_statusLine_readyConnector_readsReadyPlusRecency() {
        let line = SettingsTabView.statusLine(for: MockData.connectorProton, anchor: anchor)
        XCTAssertTrue(line.hasPrefix("Ready · "))
        XCTAssertTrue(line.contains("3m ago") || line.contains("m ago"))
    }

    func test_statusLine_pausedConnector_omitsLastSync() {
        let line = SettingsTabView.statusLine(for: MockData.connectorCalDAV, anchor: anchor)
        XCTAssertEqual(line, "Paused")
    }

    func test_statusLine_errorConnector_surfacesLastError() {
        let line = SettingsTabView.statusLine(for: MockData.connectorIOSContext, anchor: anchor)
        XCTAssertTrue(line.hasPrefix("Error · "))
        XCTAssertTrue(line.contains("Device unreachable"))
    }

    func test_statusLine_readyConnector_noLastSync_readsNeverSynced() {
        let c = Connector(
            id: "fresh",
            kind: "proton_mail",
            enabled: true,
            status: "ready",
            lastSyncAt: nil,
            lastError: nil
        )
        XCTAssertEqual(SettingsTabView.statusLine(for: c, anchor: anchor), "Ready · never synced")
    }

    func test_statusLine_errorWithoutLastError_doesNotCrash() {
        let c = Connector(
            id: "mystery",
            kind: "proton_mail",
            enabled: true,
            status: "error",
            lastSyncAt: ts(secondsAgo: 60),
            lastError: nil
        )
        XCTAssertEqual(SettingsTabView.statusLine(for: c, anchor: anchor), "Error")
    }
}

// MARK: - SettingsTabView.isValidTime

final class SettingsIsValidTimeTests: XCTestCase {

    func test_wellFormed_returnsTrue() {
        XCTAssertTrue(SettingsTabView.isValidTime("00:00"))
        XCTAssertTrue(SettingsTabView.isValidTime("09:30"))
        XCTAssertTrue(SettingsTabView.isValidTime("23:59"))
    }

    func test_singleDigitHour_isRejected() {
        XCTAssertFalse(SettingsTabView.isValidTime("9:30"))
    }

    func test_outOfRange_isRejected() {
        XCTAssertFalse(SettingsTabView.isValidTime("24:00"))
        XCTAssertFalse(SettingsTabView.isValidTime("12:60"))
    }

    func test_missingColon_isRejected() {
        XCTAssertFalse(SettingsTabView.isValidTime("0930"))
    }

    func test_nonNumeric_isRejected() {
        XCTAssertFalse(SettingsTabView.isValidTime("ab:cd"))
    }

    func test_empty_isRejected() {
        XCTAssertFalse(SettingsTabView.isValidTime(""))
    }
}

// MARK: - SettingsTabView slider helpers

final class SettingsSliderHelperTests: XCTestCase {

    func test_clampUnit_clampsBelowZero() {
        XCTAssertEqual(SettingsTabView.clampUnit(-0.3), 0.0, accuracy: 0.0001)
    }

    func test_clampUnit_clampsAboveOne() {
        XCTAssertEqual(SettingsTabView.clampUnit(1.6), 1.0, accuracy: 0.0001)
    }

    func test_clampUnit_passesThroughInRange() {
        XCTAssertEqual(SettingsTabView.clampUnit(0.42), 0.42, accuracy: 0.0001)
    }

    func test_percentLabel_roundsAndAppendsGlyph() {
        XCTAssertEqual(SettingsTabView.percentLabel(for: 0.5), "50%")
        XCTAssertEqual(SettingsTabView.percentLabel(for: 0.425), "43%")
        XCTAssertEqual(SettingsTabView.percentLabel(for: 0.0), "0%")
        XCTAssertEqual(SettingsTabView.percentLabel(for: 1.0), "100%")
    }
}

// MARK: - SettingsTabView state helpers

final class SettingsStateHelperTests: XCTestCase {

    func test_toggled_preservesEverythingElse() {
        let flipped = SettingsTabView.toggled(MockData.connectorProton, enabled: false)
        XCTAssertFalse(flipped.enabled)
        XCTAssertEqual(flipped.id, MockData.connectorProton.id)
        XCTAssertEqual(flipped.kind, MockData.connectorProton.kind)
        XCTAssertEqual(flipped.status, MockData.connectorProton.status)
        XCTAssertEqual(flipped.lastSyncAt, MockData.connectorProton.lastSyncAt)
    }

    func test_withQuietHoursStart_onlyMutatesStart() {
        let updated = SettingsTabView.withQuietHoursStart(MockData.preferences, "23:00")
        XCTAssertEqual(updated.quietHoursStart, "23:00")
        XCTAssertEqual(updated.quietHoursEnd, MockData.preferences.quietHoursEnd)
        XCTAssertEqual(updated.autonomyLevel, MockData.preferences.autonomyLevel, accuracy: 0.0001)
    }

    func test_withQuietHoursEnd_onlyMutatesEnd() {
        let updated = SettingsTabView.withQuietHoursEnd(MockData.preferences, "08:15")
        XCTAssertEqual(updated.quietHoursEnd, "08:15")
        XCTAssertEqual(updated.quietHoursStart, MockData.preferences.quietHoursStart)
    }

    func test_withAutonomy_clampsOverOne() {
        let updated = SettingsTabView.withAutonomy(MockData.preferences, 1.8)
        XCTAssertEqual(updated.autonomyLevel, 1.0, accuracy: 0.0001)
    }

    func test_withProactivity_clampsBelowZero() {
        let updated = SettingsTabView.withProactivity(MockData.preferences, -0.2)
        XCTAssertEqual(updated.proactivity, 0.0, accuracy: 0.0001)
    }
}

// MARK: - SettingsTabView.accessibilityLabel

final class SettingsAccessibilityLabelTests: XCTestCase {

    func test_readyConnector_labelMentionsNameAndStatus() {
        let label = SettingsTabView.accessibilityLabel(
            for: MockData.connectorProton,
            anchor: MockData.anchorDate
        )
        XCTAssertTrue(label.contains("Proton Mail"))
        XCTAssertTrue(label.contains("Ready"))
    }

    func test_errorConnector_labelIncludesErrorDetail() {
        let label = SettingsTabView.accessibilityLabel(
            for: MockData.connectorIOSContext,
            anchor: MockData.anchorDate
        )
        XCTAssertTrue(label.contains("iOS Context"))
        XCTAssertTrue(label.contains("Error"))
    }
}

// MARK: - Connector: Hashable

final class ConnectorHashableTests: XCTestCase {

    func test_equalConnectors_hashSame() {
        let a = Connector(id: "c:a", kind: "proton_mail", enabled: true, status: "ready", lastSyncAt: 1, lastError: nil)
        let b = Connector(id: "c:a", kind: "proton_mail", enabled: true, status: "ready", lastSyncAt: 1, lastError: nil)
        XCTAssertEqual(a.hashValue, b.hashValue)
    }

    func test_differentIDs_areNotEqual() {
        let a = Connector(id: "c:a", kind: "proton_mail", enabled: true, status: "ready", lastSyncAt: nil, lastError: nil)
        let b = Connector(id: "c:b", kind: "proton_mail", enabled: true, status: "ready", lastSyncAt: nil, lastError: nil)
        XCTAssertNotEqual(a, b)
    }

    func test_setRoundTrip() {
        let rows = MockData.connectors
        XCTAssertEqual(Set(rows).count, rows.count)
    }
}

// MARK: - ConnectorEditView.defaultConfigKeys / defaultSecretKeys

final class ConnectorEditDefaultKeysTests: XCTestCase {

    func test_proton_config() {
        XCTAssertEqual(
            ConnectorEditView.defaultConfigKeys(for: "proton_mail"),
            ["username", "mailbox_folder"]
        )
    }

    func test_proton_hasSecrets() {
        XCTAssertTrue(ConnectorEditView.defaultSecretKeys(for: "proton_mail").contains("password"))
    }

    func test_imessage_noSecrets() {
        XCTAssertTrue(ConnectorEditView.defaultSecretKeys(for: "imessage").isEmpty)
    }

    func test_iosContext_noConfig_butHasPushToken() {
        XCTAssertTrue(ConnectorEditView.defaultConfigKeys(for: "ios_context").isEmpty)
        XCTAssertEqual(ConnectorEditView.defaultSecretKeys(for: "ios_context"), ["device_push_token"])
    }

    func test_unknownKind_returnsEmptyLists() {
        XCTAssertTrue(ConnectorEditView.defaultConfigKeys(for: "plaid").isEmpty)
        XCTAssertTrue(ConnectorEditView.defaultSecretKeys(for: "plaid").isEmpty)
    }
}

// MARK: - ConnectorEditView.labelForKey

final class ConnectorEditLabelForKeyTests: XCTestCase {

    func test_snakeCase_titleCasesFirst_lowercasesRest() {
        XCTAssertEqual(ConnectorEditView.labelForKey("mailbox_folder"), "Mailbox folder")
        XCTAssertEqual(ConnectorEditView.labelForKey("device_push_token"), "Device push token")
    }

    func test_singleWord_isCapitalized() {
        XCTAssertEqual(ConnectorEditView.labelForKey("password"), "Password")
    }

    func test_empty_isEmpty() {
        XCTAssertEqual(ConnectorEditView.labelForKey(""), "")
    }
}

// MARK: - ConnectorEditView.isValidDraftValue

final class ConnectorEditIsValidDraftValueTests: XCTestCase {

    func test_empty_isValid_meansNoChange() {
        XCTAssertTrue(ConnectorEditView.isValidDraftValue(""))
    }

    func test_nonEmpty_validWhenNonWhitespace() {
        XCTAssertTrue(ConnectorEditView.isValidDraftValue("secret"))
    }

    func test_whitespaceOnly_invalid() {
        XCTAssertFalse(ConnectorEditView.isValidDraftValue("   "))
        XCTAssertFalse(ConnectorEditView.isValidDraftValue("\n\t"))
    }
}

// MARK: - ConnectorEditView.headerSubtitle

final class ConnectorEditHeaderSubtitleTests: XCTestCase {

    func test_errorConnector_returnsLastError() {
        let line = ConnectorEditView.headerSubtitle(for: MockData.connectorIOSContext)
        XCTAssertEqual(line, MockData.connectorIOSContext.lastError)
    }

    func test_errorConnector_fallsBack_whenLastErrorMissing() {
        let c = Connector(id: "x", kind: "proton_mail", enabled: true, status: "error", lastSyncAt: nil, lastError: nil)
        XCTAssertEqual(
            ConnectorEditView.headerSubtitle(for: c),
            "Connector reported an error on the last sync."
        )
    }

    func test_readyConnector_readsLastSync_inUTC() {
        let line = ConnectorEditView.headerSubtitle(for: MockData.connectorProton)
        XCTAssertTrue(line.hasPrefix("Last sync "))
        XCTAssertTrue(line.hasSuffix(" UTC"))
    }

    func test_readyConnector_noLastSync_readsNoSyncRecorded() {
        let c = Connector(id: "x", kind: "proton_mail", enabled: true, status: "ready", lastSyncAt: nil, lastError: nil)
        XCTAssertEqual(ConnectorEditView.headerSubtitle(for: c), "No sync recorded yet.")
    }
}

// MARK: - ConnectorEditView.hasChanges / allEditedValuesValid / canSave

final class ConnectorEditFormValidationTests: XCTestCase {

    private let baseline = MockData.connectorProton

    // hasChanges matrix

    func test_hasChanges_false_whenAllDraftsEmpty_andEnabledUnchanged() {
        XCTAssertFalse(
            ConnectorEditView.hasChanges(
                connector: baseline,
                enabledDraft: baseline.enabled,
                configDraft: ["username": ""],
                secretDraft: ["password": ""]
            )
        )
    }

    func test_hasChanges_true_whenEnabledFlipped() {
        XCTAssertTrue(
            ConnectorEditView.hasChanges(
                connector: baseline,
                enabledDraft: !baseline.enabled,
                configDraft: ["username": ""],
                secretDraft: ["password": ""]
            )
        )
    }

    func test_hasChanges_true_whenConfigPopulated() {
        XCTAssertTrue(
            ConnectorEditView.hasChanges(
                connector: baseline,
                enabledDraft: baseline.enabled,
                configDraft: ["username": "alice"],
                secretDraft: [:]
            )
        )
    }

    func test_hasChanges_true_whenSecretPopulated() {
        XCTAssertTrue(
            ConnectorEditView.hasChanges(
                connector: baseline,
                enabledDraft: baseline.enabled,
                configDraft: [:],
                secretDraft: ["password": "hunter2"]
            )
        )
    }

    // allEditedValuesValid matrix

    func test_allEditedValuesValid_true_whenAllEmptyOrSubstantive() {
        XCTAssertTrue(
            ConnectorEditView.allEditedValuesValid(
                configDraft: ["username": "alice", "mailbox_folder": ""],
                secretDraft: ["password": "hunter2"]
            )
        )
    }

    func test_allEditedValuesValid_false_whenWhitespaceOnlyConfig() {
        XCTAssertFalse(
            ConnectorEditView.allEditedValuesValid(
                configDraft: ["username": "   "],
                secretDraft: [:]
            )
        )
    }

    func test_allEditedValuesValid_false_whenWhitespaceOnlySecret() {
        XCTAssertFalse(
            ConnectorEditView.allEditedValuesValid(
                configDraft: [:],
                secretDraft: ["password": "\t"]
            )
        )
    }

    // canSave — top-level gate

    func test_canSave_false_whenNoChanges() {
        XCTAssertFalse(
            ConnectorEditView.canSave(
                connector: baseline,
                enabledDraft: baseline.enabled,
                configDraft: [:],
                secretDraft: [:]
            )
        )
    }

    func test_canSave_true_whenEnabledFlipped_andAllFieldsValid() {
        XCTAssertTrue(
            ConnectorEditView.canSave(
                connector: baseline,
                enabledDraft: !baseline.enabled,
                configDraft: [:],
                secretDraft: [:]
            )
        )
    }

    func test_canSave_false_whenChangesButWhitespaceField() {
        XCTAssertFalse(
            ConnectorEditView.canSave(
                connector: baseline,
                enabledDraft: baseline.enabled,
                configDraft: ["username": "   "],
                secretDraft: [:]
            )
        )
    }

    func test_canSave_true_whenSecretRotation() {
        XCTAssertTrue(
            ConnectorEditView.canSave(
                connector: baseline,
                enabledDraft: baseline.enabled,
                configDraft: [:],
                secretDraft: ["password": "hunter2"]
            )
        )
    }
}

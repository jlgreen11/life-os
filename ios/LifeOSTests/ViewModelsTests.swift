//
//  ViewModelsTests.swift
//  Life OS — Per-tab view model unit tests
//
//  One file, one mock, four `@MainActor` test classes — keeps the suite
//  flat to match the existing per-tab test files in this directory.
//
//  The mock (`MockAPIClient`) is a plain class so tests can poke its
//  `*Result` slots between calls. Every protocol method records its
//  call name into `callLog` so order-of-operations is testable too.
//
//  Coverage:
//
//    NowViewModelTests
//      • load() happy + error paths — feed populates / error string set
//      • accept / dismiss / snooze / undo — re-bucket the returned
//        Moment based on its post-transition state
//      • reconcile() truth table for every MomentState
//
//    YouViewModelTests
//      • load() happy + error paths
//
//    PeopleViewModelTests
//      • load() happy + error paths
//      • query="" sends nil; non-empty query is forwarded
//      • loadContact happy / error
//
//    SettingsViewModelTests
//      • loadConnectors() happy + error
//      • updateConnector upserts in place / appends new
//      • updatePreference patches the preferences slot for known keys
//        and leaves unknown keys alone
//

import XCTest
@testable import LifeOS

// MARK: - Mock APIClient

/// In-memory implementation of `APIClientProtocol`. Each method has a
/// `Result`-typed slot that tests pre-stage; on call the slot's
/// `.get()` either returns the value or throws the staged error.
///
/// Defaults are wired to `MockData` fixtures so any unstaged method
/// returns the same canned shape — keeps tests focused on the
/// behavior they care about and avoids "had to stage every method"
/// boilerplate.
final class MockAPIClient: APIClientProtocol {

    // Staged responses — overwrite per test.
    var nowResult: Result<MomentFeed, Error> = .success(MockData.feed)
    var youResult: Result<SelfPortrait, Error> = .success(MockData.selfPortrait)
    var peopleResult: Result<PeopleList, Error> = .success(MockData.peopleList)
    var contactResult: Result<ContactDossier, Error> = .success(MockData.contactDossierDad)
    var connectorsResult: Result<[Connector], Error> = .success(MockData.connectors)
    var updateConnectorResult: Result<Connector, Error> = .success(MockData.connectorProton)
    var actionResult: Result<Moment, Error> = .success(MockData.momentDone)
    var preferenceUpdateError: Error? = nil

    // Recorded calls.
    var callLog: [String] = []
    var lastPeopleQuery: String? = nil
    var lastPeoplePage: Int? = nil
    var lastUpdateConnectorId: String? = nil
    var lastUpdateConnectorPayload: ConnectorConfigUpdate? = nil
    var lastPreferenceKey: String? = nil
    var lastPreferenceValue: AnyCodable? = nil
    var lastSnoozeUntil: Int? = nil
    var lastEditParams: [String: AnyCodable]? = nil

    // MARK: APIClientProtocol — Now

    func getNow() async throws -> MomentFeed {
        callLog.append("getNow")
        return try nowResult.get()
    }

    func acceptMoment(id: String, annotation: String?) async throws -> Moment {
        callLog.append("accept:\(id)")
        return try actionResult.get()
    }

    func dismissMoment(id: String, annotation: String?) async throws -> Moment {
        callLog.append("dismiss:\(id)")
        return try actionResult.get()
    }

    func snoozeMoment(id: String, snoozeUntil: Int, annotation: String?) async throws -> Moment {
        callLog.append("snooze:\(id)")
        lastSnoozeUntil = snoozeUntil
        return try actionResult.get()
    }

    func undoMoment(id: String) async throws -> Moment {
        callLog.append("undo:\(id)")
        return try actionResult.get()
    }

    func editMoment(id: String, actionParams: [String: AnyCodable]) async throws -> Moment {
        callLog.append("edit:\(id)")
        lastEditParams = actionParams
        return try actionResult.get()
    }

    // MARK: APIClientProtocol — You

    func getYou() async throws -> SelfPortrait {
        callLog.append("getYou")
        return try youResult.get()
    }

    // MARK: APIClientProtocol — People

    func getPeople(query: String?, page: Int, pageSize: Int?) async throws -> PeopleList {
        callLog.append("getPeople")
        lastPeopleQuery = query
        lastPeoplePage = page
        return try peopleResult.get()
    }

    func getContact(id: String) async throws -> ContactDossier {
        callLog.append("getContact:\(id)")
        return try contactResult.get()
    }

    // MARK: APIClientProtocol — Settings

    func getConnectors() async throws -> [Connector] {
        callLog.append("getConnectors")
        return try connectorsResult.get()
    }

    func updateConnector(id: String, update: ConnectorConfigUpdate) async throws -> Connector {
        callLog.append("updateConnector:\(id)")
        lastUpdateConnectorId = id
        lastUpdateConnectorPayload = update
        return try updateConnectorResult.get()
    }

    func updatePreference(key: String, value: AnyCodable) async throws {
        callLog.append("updatePreference:\(key)")
        lastPreferenceKey = key
        lastPreferenceValue = value
        if let err = preferenceUpdateError { throw err }
    }
}

// MARK: - Test error

/// Sentinel error used to drive every "load failed" assertion. Keeps
/// the assertion language consistent across the four test classes.
private struct StubError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

// MARK: - Helpers for synthesizing transitioned Moments

/// Make a near-clone of a `Moment` with the only change being its
/// terminal `state`. Used by the action-dispatch tests to verify
/// `reconcile` re-buckets correctly.
private func momentWith(_ state: MomentState, basedOn template: Moment) -> Moment {
    Moment(
        id: template.id,
        createdAt: template.createdAt,
        expiresAt: template.expiresAt,
        insight: template.insight,
        evidence: template.evidence,
        evidenceHash: template.evidenceHash,
        proposedAction: template.proposedAction,
        sourceInsightType: template.sourceInsightType,
        state: state,
        scheduledFor: template.scheduledFor,
        contextTrigger: template.contextTrigger,
        snoozeUntil: template.snoozeUntil,
        confidence: template.confidence,
        feedbackWeight: template.feedbackWeight,
        stateHistory: template.stateHistory
    )
}

// MARK: - NowViewModel tests

@MainActor
final class NowViewModelTests: XCTestCase {

    func test_load_happyPath_populatesFeed_andClearsError() async {
        let mock = MockAPIClient()
        mock.nowResult = .success(MockData.feed)
        let vm = NowViewModel(client: mock)

        await vm.load()

        XCTAssertEqual(vm.feed.pending.count, 3)
        XCTAssertEqual(vm.feed.scheduled.count, 2)
        XCTAssertEqual(vm.feed.done.count, 1)
        XCTAssertNil(vm.errorMessage)
        XCTAssertFalse(vm.isLoading)
        XCTAssertEqual(mock.callLog, ["getNow"])
    }

    func test_load_errorPath_setsErrorMessage_andLeavesFeedUntouched() async {
        let mock = MockAPIClient()
        mock.nowResult = .failure(StubError(message: "now-fetch-failed"))
        let vm = NowViewModel(client: mock)

        // Seed a known feed so we can assert it didn't change.
        let seeded = MockData.feed
        vm.feed = seeded

        await vm.load()

        XCTAssertEqual(vm.errorMessage, "now-fetch-failed")
        XCTAssertEqual(vm.feed, seeded)
        XCTAssertFalse(vm.isLoading)
    }

    func test_accept_movesPendingMomentToDoneBucket() async {
        let mock = MockAPIClient()
        let vm = NowViewModel(client: mock)
        vm.feed = MockData.feed

        // Server returns the same moment with state=.done.
        mock.actionResult = .success(momentWith(.done, basedOn: MockData.momentCadence))

        await vm.accept(momentId: MockData.momentCadence.id)

        XCTAssertFalse(vm.feed.pending.contains(where: { $0.id == MockData.momentCadence.id }))
        XCTAssertTrue(vm.feed.done.contains(where: { $0.id == MockData.momentCadence.id }))
        XCTAssertNil(vm.errorMessage)
        XCTAssertEqual(mock.callLog, ["accept:\(MockData.momentCadence.id)"])
    }

    func test_dismiss_removesMomentFromEveryBucket() async {
        let mock = MockAPIClient()
        let vm = NowViewModel(client: mock)
        vm.feed = MockData.feed

        mock.actionResult = .success(momentWith(.dismissed, basedOn: MockData.momentRelationship))

        await vm.dismiss(momentId: MockData.momentRelationship.id)

        XCTAssertFalse(vm.feed.pending.contains(where: { $0.id == MockData.momentRelationship.id }))
        XCTAssertFalse(vm.feed.scheduled.contains(where: { $0.id == MockData.momentRelationship.id }))
        XCTAssertFalse(vm.feed.done.contains(where: { $0.id == MockData.momentRelationship.id }))
    }

    func test_snooze_movesPendingMomentToScheduledBucket_andForwardsTimestamp() async {
        let mock = MockAPIClient()
        let vm = NowViewModel(client: mock)
        vm.feed = MockData.feed

        mock.actionResult = .success(momentWith(.snoozed, basedOn: MockData.momentTemporal))

        let until = 1_777_300_000
        await vm.snooze(momentId: MockData.momentTemporal.id, snoozeUntil: until)

        XCTAssertFalse(vm.feed.pending.contains(where: { $0.id == MockData.momentTemporal.id }))
        XCTAssertTrue(vm.feed.scheduled.contains(where: { $0.id == MockData.momentTemporal.id }))
        XCTAssertEqual(mock.lastSnoozeUntil, until)
    }

    func test_undo_re_addsMomentToPending_whenServerRestoresSuggested() async {
        let mock = MockAPIClient()
        let vm = NowViewModel(client: mock)
        // Start with a feed where momentCadence has been dismissed (not
        // present anywhere). Undo restores it as `.suggested`.
        let stripped = NowViewModel.reconcile(
            feed: MockData.feed,
            updated: momentWith(.dismissed, basedOn: MockData.momentCadence)
        )
        vm.feed = stripped

        mock.actionResult = .success(momentWith(.suggested, basedOn: MockData.momentCadence))

        await vm.undo(momentId: MockData.momentCadence.id)

        XCTAssertTrue(vm.feed.pending.contains(where: { $0.id == MockData.momentCadence.id }))
    }

    func test_action_errorPath_setsErrorMessage_andLeavesFeedUntouched() async {
        let mock = MockAPIClient()
        let vm = NowViewModel(client: mock)
        vm.feed = MockData.feed
        let snapshot = vm.feed
        mock.actionResult = .failure(StubError(message: "accept-failed"))

        await vm.accept(momentId: MockData.momentCadence.id)

        XCTAssertEqual(vm.errorMessage, "accept-failed")
        XCTAssertEqual(vm.feed, snapshot)
    }

    func test_isLoading_togglesAroundLoad() async {
        let mock = MockAPIClient()
        let vm = NowViewModel(client: mock)
        XCTAssertFalse(vm.isLoading)
        await vm.load()
        XCTAssertFalse(vm.isLoading, "isLoading must reset to false on success")

        mock.nowResult = .failure(StubError(message: "boom"))
        await vm.load()
        XCTAssertFalse(vm.isLoading, "isLoading must reset to false on failure too")
    }
}

// MARK: - Reconcile truth table

@MainActor
final class NowReconcileTests: XCTestCase {

    func test_reconcile_suggested_landsInPending() {
        let updated = momentWith(.suggested, basedOn: MockData.momentCadence)
        let result = NowViewModel.reconcile(feed: MomentFeed(pending: [], scheduled: [], done: []), updated: updated)
        XCTAssertEqual(result.pending.map(\.id), [updated.id])
        XCTAssertTrue(result.scheduled.isEmpty)
        XCTAssertTrue(result.done.isEmpty)
    }

    func test_reconcile_accepted_landsInPending() {
        let updated = momentWith(.accepted, basedOn: MockData.momentCadence)
        let result = NowViewModel.reconcile(feed: MomentFeed(pending: [], scheduled: [], done: []), updated: updated)
        XCTAssertEqual(result.pending.map(\.id), [updated.id])
    }

    func test_reconcile_snoozed_landsInScheduled() {
        let updated = momentWith(.snoozed, basedOn: MockData.momentCadence)
        let result = NowViewModel.reconcile(feed: MomentFeed(pending: [], scheduled: [], done: []), updated: updated)
        XCTAssertEqual(result.scheduled.map(\.id), [updated.id])
        XCTAssertTrue(result.pending.isEmpty)
    }

    func test_reconcile_done_landsInDone() {
        let updated = momentWith(.done, basedOn: MockData.momentCadence)
        let result = NowViewModel.reconcile(feed: MomentFeed(pending: [], scheduled: [], done: []), updated: updated)
        XCTAssertEqual(result.done.map(\.id), [updated.id])
    }

    func test_reconcile_dismissed_disappearsFromEveryBucket() {
        let dismissed = momentWith(.dismissed, basedOn: MockData.momentCadence)
        let seeded = MomentFeed(
            pending: [MockData.momentCadence],
            scheduled: [],
            done: []
        )
        let result = NowViewModel.reconcile(feed: seeded, updated: dismissed)
        XCTAssertTrue(result.pending.isEmpty)
        XCTAssertTrue(result.scheduled.isEmpty)
        XCTAssertTrue(result.done.isEmpty)
    }

    func test_reconcile_expired_disappearsFromEveryBucket() {
        let expired = momentWith(.expired, basedOn: MockData.momentRoutine)
        let seeded = MomentFeed(
            pending: [],
            scheduled: [MockData.momentRoutine],
            done: []
        )
        let result = NowViewModel.reconcile(feed: seeded, updated: expired)
        XCTAssertTrue(result.scheduled.isEmpty)
    }

    func test_reconcile_movesAcrossBuckets_inOneStep() {
        // Snoozed moment sitting in `scheduled` is undone back to
        // `.suggested`, which should land it in `pending`.
        let snoozed = momentWith(.snoozed, basedOn: MockData.momentCadence)
        let seeded = MomentFeed(
            pending: [],
            scheduled: [snoozed],
            done: []
        )
        let restored = momentWith(.suggested, basedOn: MockData.momentCadence)
        let result = NowViewModel.reconcile(feed: seeded, updated: restored)
        XCTAssertTrue(result.scheduled.isEmpty)
        XCTAssertEqual(result.pending.map(\.id), [restored.id])
    }
}

// MARK: - YouViewModel tests

@MainActor
final class YouViewModelTests: XCTestCase {

    func test_load_happyPath_populatesPortrait() async {
        let mock = MockAPIClient()
        mock.youResult = .success(MockData.selfPortrait)
        let vm = YouViewModel(client: mock)

        await vm.load()

        XCTAssertEqual(vm.portrait, MockData.selfPortrait)
        XCTAssertNil(vm.errorMessage)
        XCTAssertFalse(vm.isLoading)
        XCTAssertEqual(mock.callLog, ["getYou"])
    }

    func test_load_errorPath_setsErrorMessage_andLeavesPortraitUntouched() async {
        let mock = MockAPIClient()
        mock.youResult = .failure(StubError(message: "you-fetch-failed"))
        let vm = YouViewModel(client: mock)

        let seeded = MockData.selfPortrait
        vm.portrait = seeded

        await vm.load()

        XCTAssertEqual(vm.errorMessage, "you-fetch-failed")
        XCTAssertEqual(vm.portrait, seeded)
        XCTAssertFalse(vm.isLoading)
    }

    func test_defaultPortrait_isFreshInstall_beforeLoad() {
        let vm = YouViewModel(client: MockAPIClient())
        XCTAssertEqual(vm.portrait, SelfPortrait())
    }
}

// MARK: - PeopleViewModel tests

@MainActor
final class PeopleViewModelTests: XCTestCase {

    func test_load_happyPath_populatesPeople_andClearsError() async {
        let mock = MockAPIClient()
        mock.peopleResult = .success(MockData.peopleList)
        let vm = PeopleViewModel(client: mock)

        await vm.load()

        XCTAssertEqual(vm.people, MockData.peopleList)
        XCTAssertNil(vm.errorMessage)
        XCTAssertFalse(vm.isLoading)
    }

    func test_load_errorPath_setsErrorMessage() async {
        let mock = MockAPIClient()
        mock.peopleResult = .failure(StubError(message: "people-fetch-failed"))
        let vm = PeopleViewModel(client: mock)

        await vm.load()

        XCTAssertEqual(vm.errorMessage, "people-fetch-failed")
    }

    func test_load_emptyQuery_sendsNilOnTheWire() async {
        let mock = MockAPIClient()
        let vm = PeopleViewModel(client: mock)
        vm.query = ""

        await vm.load()

        XCTAssertNil(mock.lastPeopleQuery)
        XCTAssertEqual(mock.lastPeoplePage, 1)
    }

    func test_load_whitespaceOnlyQuery_sendsNilOnTheWire() async {
        let mock = MockAPIClient()
        let vm = PeopleViewModel(client: mock)
        vm.query = "   "

        await vm.load()

        XCTAssertNil(mock.lastPeopleQuery)
    }

    func test_load_nonEmptyQuery_isForwardedTrimmed() async {
        let mock = MockAPIClient()
        let vm = PeopleViewModel(client: mock)
        vm.query = "sam"

        await vm.load()

        XCTAssertEqual(mock.lastPeopleQuery, "sam")
    }

    func test_loadContact_happyPath_setsDossier() async {
        let mock = MockAPIClient()
        mock.contactResult = .success(MockData.contactDossierDad)
        let vm = PeopleViewModel(client: mock)

        await vm.loadContact(id: "contact:dad")

        XCTAssertEqual(vm.dossier?.contactId, "contact:dad")
        XCTAssertNil(vm.errorMessage)
        XCTAssertEqual(mock.callLog, ["getContact:contact:dad"])
    }

    func test_loadContact_errorPath_setsError_andLeavesDossierUntouched() async {
        let mock = MockAPIClient()
        let vm = PeopleViewModel(client: mock)
        vm.dossier = MockData.contactDossierSparse

        mock.contactResult = .failure(StubError(message: "contact-not-found"))

        await vm.loadContact(id: "contact:nobody")

        XCTAssertEqual(vm.errorMessage, "contact-not-found")
        XCTAssertEqual(vm.dossier?.contactId, MockData.contactDossierSparse.contactId)
    }

    func test_clearDossier_resetsSlot() {
        let vm = PeopleViewModel(client: MockAPIClient())
        vm.dossier = MockData.contactDossierDad
        vm.clearDossier()
        XCTAssertNil(vm.dossier)
    }
}

// MARK: - SettingsViewModel tests

@MainActor
final class SettingsViewModelTests: XCTestCase {

    func test_loadConnectors_happyPath_populatesList() async {
        let mock = MockAPIClient()
        mock.connectorsResult = .success(MockData.connectors)
        let vm = SettingsViewModel(client: mock)

        await vm.loadConnectors()

        XCTAssertEqual(vm.connectors.count, MockData.connectors.count)
        XCTAssertNil(vm.errorMessage)
        XCTAssertFalse(vm.isLoading)
    }

    func test_loadConnectors_errorPath_setsErrorMessage() async {
        let mock = MockAPIClient()
        mock.connectorsResult = .failure(StubError(message: "connectors-fetch-failed"))
        let vm = SettingsViewModel(client: mock)

        await vm.loadConnectors()

        XCTAssertEqual(vm.errorMessage, "connectors-fetch-failed")
    }

    func test_updateConnector_upsertsExistingRow_inPlace() async {
        let mock = MockAPIClient()
        let vm = SettingsViewModel(client: mock)
        vm.connectors = MockData.connectors

        // Server flips proton's enabled flag to false.
        let flipped = Connector.makeForTesting(
            id: "proton",
            kind: "proton_mail",
            enabled: false,
            status: "paused",
            lastSyncAt: MockData.connectorProton.lastSyncAt,
            lastError: nil
        )
        mock.updateConnectorResult = .success(flipped)

        await vm.updateConnector(id: "proton", update: ConnectorConfigUpdate(enabled: false))

        XCTAssertEqual(vm.connectors.count, MockData.connectors.count)
        XCTAssertEqual(vm.connectors.first(where: { $0.id == "proton" })?.enabled, false)
        XCTAssertEqual(mock.lastUpdateConnectorId, "proton")
        XCTAssertEqual(mock.lastUpdateConnectorPayload?.enabled, false)
    }

    func test_updateConnector_appendsNewRow_whenIdNotPresent() async {
        let mock = MockAPIClient()
        let vm = SettingsViewModel(client: mock)
        vm.connectors = []

        mock.updateConnectorResult = .success(MockData.connectorProton)

        await vm.updateConnector(id: "proton", update: ConnectorConfigUpdate(enabled: true))

        XCTAssertEqual(vm.connectors.count, 1)
        XCTAssertEqual(vm.connectors.first?.id, "proton")
    }

    func test_updateConnector_errorPath_leavesListUntouched() async {
        let mock = MockAPIClient()
        let vm = SettingsViewModel(client: mock)
        vm.connectors = MockData.connectors
        let snapshot = vm.connectors

        mock.updateConnectorResult = .failure(StubError(message: "patch-failed"))

        await vm.updateConnector(id: "proton", update: ConnectorConfigUpdate(enabled: false))

        XCTAssertEqual(vm.connectors, snapshot)
        XCTAssertEqual(vm.errorMessage, "patch-failed")
    }

    func test_updatePreference_patchesQuietHoursStart_locally() async {
        let mock = MockAPIClient()
        let vm = SettingsViewModel(client: mock)

        await vm.updatePreference(key: "quiet_hours_start", value: AnyCodable("21:00"))

        XCTAssertEqual(vm.preferences.quietHoursStart, "21:00")
        XCTAssertEqual(mock.lastPreferenceKey, "quiet_hours_start")
    }

    func test_updatePreference_patchesAutonomy_andProactivity() async {
        let mock = MockAPIClient()
        let vm = SettingsViewModel(client: mock)

        await vm.updatePreference(key: "autonomy_level", value: AnyCodable(0.8))
        await vm.updatePreference(key: "proactivity", value: AnyCodable(0.2))

        XCTAssertEqual(vm.preferences.autonomyLevel, 0.8, accuracy: 0.0001)
        XCTAssertEqual(vm.preferences.proactivity, 0.2, accuracy: 0.0001)
    }

    func test_updatePreference_unknownKey_isNoOp() async {
        let mock = MockAPIClient()
        let vm = SettingsViewModel(client: mock)
        let before = vm.preferences

        await vm.updatePreference(key: "unrecognized_key", value: AnyCodable("42"))

        XCTAssertEqual(vm.preferences, before)
    }

    func test_updatePreference_errorPath_doesNotPatchLocally() async {
        let mock = MockAPIClient()
        let vm = SettingsViewModel(client: mock)
        let before = vm.preferences
        mock.preferenceUpdateError = StubError(message: "pref-failed")

        await vm.updatePreference(key: "quiet_hours_start", value: AnyCodable("23:00"))

        XCTAssertEqual(vm.preferences, before)
        XCTAssertEqual(vm.errorMessage, "pref-failed")
    }

    func test_applying_truthTable_locks_perKey_routing() {
        let base = Preferences.defaults

        let q1 = SettingsViewModel.applying(key: "quiet_hours_start", value: AnyCodable("21:30"), to: base)
        XCTAssertEqual(q1.quietHoursStart, "21:30")
        XCTAssertEqual(q1.quietHoursEnd, base.quietHoursEnd)

        let q2 = SettingsViewModel.applying(key: "quiet_hours_end", value: AnyCodable("06:30"), to: base)
        XCTAssertEqual(q2.quietHoursEnd, "06:30")

        let a = SettingsViewModel.applying(key: "autonomy_level", value: AnyCodable(0.9), to: base)
        XCTAssertEqual(a.autonomyLevel, 0.9, accuracy: 0.0001)

        // Int payload should coerce to Double.
        let p = SettingsViewModel.applying(key: "proactivity", value: AnyCodable(1), to: base)
        XCTAssertEqual(p.proactivity, 1.0, accuracy: 0.0001)

        // Wrong-type payload is a no-op.
        let bad = SettingsViewModel.applying(key: "autonomy_level", value: AnyCodable("nope"), to: base)
        XCTAssertEqual(bad, base)

        // Unknown key is a no-op.
        let unknown = SettingsViewModel.applying(key: "foo", value: AnyCodable(1), to: base)
        XCTAssertEqual(unknown, base)
    }
}

// MARK: - Connector test factory

/// `Connector` ships a custom `init(from decoder:)` which suppresses the
/// compiler-synthesized memberwise init. The other test files lean on
/// fixtures from `MockData`; this helper mints arbitrary instances for
/// tests that need to exercise the upsert path with a row that isn't
/// already in the fixture set.
private extension Connector {
    static func makeForTesting(
        id: String,
        kind: String,
        enabled: Bool,
        status: String,
        lastSyncAt: Int?,
        lastError: String?
    ) -> Connector {
        let payload: [String: Any?] = [
            "id": id,
            "kind": kind,
            "enabled": enabled,
            "status": status,
            "last_sync_at": lastSyncAt as Any?,
            "last_error": lastError as Any?,
        ]
        let json = try! JSONSerialization.data(withJSONObject: payload.compactMapValues { $0 })
        return try! JSONDecoder().decode(Connector.self, from: json)
    }
}

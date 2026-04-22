//
//  PeopleTabViewTests.swift
//  Life OS — People tab + ContactDossier logic tests
//
//  Verifies:
//    1. `MockData.peopleList` / `MockData.emptyPeopleList` match the
//       shape the People tab's sections expect (YOU pinned, two
//       directory sections populated vs empty).
//    2. `MockData.contactDossierDad` / `contactDossierSparse` drive
//       the populated and fallback paths in `ContactDossierView`.
//    3. `PeopleTabView` pure helpers (`filter`, `lastSeenLabel`,
//       `cadenceLabel`, `youSubtitle`, `accessibilityLabel`,
//       `dossier(for:)`) produce deterministic copy under a frozen
//       anchor date — the scaffold renders the same thing every run.
//    4. `ContactDossierView` pure helpers (`lastSeenLine`,
//       `commStyleOrFallback`, `predictedNextLine`, `cadenceFooter`,
//       `sparklinePoints`, `sparklineAccessibilityLabel`) lock copy
//       to DESIGN.md and assert the sparkline-point invariants.
//    5. `ContactSummary: Hashable` so `NavigationLink(value:)` can
//       round-trip the row through `NavigationPath`.
//

import XCTest
@testable import LifeOS

// MARK: - MockData shape

final class MockPeopleListShapeTests: XCTestCase {

    func test_peopleList_youIsPinned_andBothSectionsPopulated() {
        let p = MockData.peopleList
        XCTAssertFalse(p.needsAttention.isEmpty, "NEEDS ATTENTION must have placeholder rows")
        XCTAssertFalse(p.activeThisWeek.isEmpty, "ACTIVE THIS WEEK must have placeholder rows")
        XCTAssertEqual(p.you, MockData.selfPortrait, "YOU row must reuse the self-portrait fixture")
        XCTAssertEqual(p.total, 4)
    }

    func test_emptyPeopleList_isAllEmpty_butYOUstillPresent() {
        let p = MockData.emptyPeopleList
        XCTAssertTrue(p.needsAttention.isEmpty)
        XCTAssertTrue(p.activeThisWeek.isEmpty)
        XCTAssertEqual(p.total, 0)
        XCTAssertEqual(p.you, MockData.emptySelfPortrait)
    }

    func test_peopleList_everyContactHasUniqueID() {
        let ids = (MockData.peopleList.needsAttention + MockData.peopleList.activeThisWeek).map(\.contactId)
        XCTAssertEqual(Set(ids).count, ids.count, "ContactSummary rows must have unique IDs")
    }

    func test_needsAttentionRows_areFlaggedAndCarryCadenceData() {
        for c in MockData.peopleList.needsAttention {
            XCTAssertTrue(c.needsAttention, "rows under NEEDS ATTENTION must carry needsAttention=true")
            XCTAssertNotNil(c.lastContactTs)
            XCTAssertNotNil(c.cadenceDeviationDays)
        }
    }
}

final class MockContactDossierShapeTests: XCTestCase {

    func test_contactDossierDad_hasPopulatedFields() {
        let d = MockData.contactDossierDad
        XCTAssertFalse(d.name.isEmpty)
        XCTAssertNotNil(d.lastContactTs)
        XCTAssertNotNil(d.usualCadenceDays)
        XCTAssertNotNil(d.commTemplate)
        XCTAssertFalse(d.cadenceSparkline.isEmpty)
        XCTAssertGreaterThan(d.recentTopics.count, 0)
        XCTAssertNotNil(d.predictedNext)
    }

    func test_contactDossierSparse_exercisesFallbackPath() {
        let d = MockData.contactDossierSparse
        XCTAssertNil(d.lastContactTs)
        XCTAssertNil(d.usualCadenceDays)
        XCTAssertNil(d.commTemplate)
        XCTAssertTrue(d.cadenceSparkline.isEmpty)
        XCTAssertTrue(d.recentTopics.isEmpty)
        XCTAssertNil(d.predictedNext)
    }

    func test_dadSparkline_includesAtLeastOneNonZeroDay() {
        XCTAssertTrue(
            MockData.contactDossierDad.cadenceSparkline.contains(where: { $0 > 0 }),
            "Fixture must include a visible peak so sparklinePoints exercises min/max normalization"
        )
    }
}

// MARK: - PeopleTabView.filter

final class PeopleFilterTests: XCTestCase {

    private let roster: [ContactSummary] = [
        ContactSummary(contactId: "c:a", name: "Alice Park"),
        ContactSummary(contactId: "c:b", name: "Bob Chen"),
        ContactSummary(contactId: "c:c", name: "Carla Rossi"),
    ]

    func test_emptyQuery_returnsAll() {
        XCTAssertEqual(PeopleTabView.filter(roster, query: "").map(\.contactId),
                       roster.map(\.contactId))
    }

    func test_whitespaceQuery_isTreatedAsEmpty() {
        XCTAssertEqual(PeopleTabView.filter(roster, query: "   \n").map(\.contactId),
                       roster.map(\.contactId))
    }

    func test_caseInsensitiveMatch() {
        let hits = PeopleTabView.filter(roster, query: "CARLA")
        XCTAssertEqual(hits.map(\.contactId), ["c:c"])
    }

    func test_partialMatch() {
        let hits = PeopleTabView.filter(roster, query: "che")
        XCTAssertEqual(hits.map(\.contactId), ["c:b"])
    }

    func test_noMatch_returnsEmpty() {
        XCTAssertTrue(PeopleTabView.filter(roster, query: "zzz").isEmpty)
    }
}

// MARK: - PeopleTabView.lastSeenLabel

final class PeopleLastSeenLabelTests: XCTestCase {

    private let anchor = MockData.anchorDate

    private func contact(daysAgo: Int?) -> ContactSummary {
        ContactSummary(
            contactId: "c:x",
            name: "X",
            lastContactTs: daysAgo.map { Int(anchor.timeIntervalSince1970) - $0 * 86_400 },
            cadenceDeviationDays: nil,
            needsAttention: false
        )
    }

    func test_nilLastContact_rendersEmDash() {
        XCTAssertEqual(PeopleTabView.lastSeenLabel(for: contact(daysAgo: nil), anchor: anchor), "—")
    }

    func test_sameDay_readsToday() {
        XCTAssertEqual(PeopleTabView.lastSeenLabel(for: contact(daysAgo: 0), anchor: anchor), "today")
    }

    func test_oneDayAgo_isSingular() {
        XCTAssertEqual(PeopleTabView.lastSeenLabel(for: contact(daysAgo: 1), anchor: anchor), "1d ago")
    }

    func test_nineDaysAgo_rendersDigitForm() {
        XCTAssertEqual(PeopleTabView.lastSeenLabel(for: contact(daysAgo: 9), anchor: anchor), "9d ago")
    }

    func test_futureTimestamp_clampsToToday() {
        let future = ContactSummary(
            contactId: "c:future",
            name: "Future",
            lastContactTs: Int(anchor.timeIntervalSince1970) + 3_600,
            cadenceDeviationDays: nil,
            needsAttention: false
        )
        XCTAssertEqual(PeopleTabView.lastSeenLabel(for: future, anchor: anchor), "today")
    }
}

// MARK: - PeopleTabView.cadenceLabel

final class PeopleCadenceLabelTests: XCTestCase {

    func test_nilDeviation_isNil() {
        let c = ContactSummary(contactId: "c:x", name: "X", cadenceDeviationDays: nil)
        XCTAssertNil(PeopleTabView.cadenceLabel(for: c))
    }

    func test_zeroDeviation_isNil_calmDefault() {
        let c = ContactSummary(contactId: "c:x", name: "X", cadenceDeviationDays: 0)
        XCTAssertNil(PeopleTabView.cadenceLabel(for: c))
    }

    func test_positiveDeviation_usesPlusSign() {
        let c = ContactSummary(contactId: "c:x", name: "X", cadenceDeviationDays: 4)
        XCTAssertEqual(PeopleTabView.cadenceLabel(for: c), "+4d")
    }

    func test_negativeDeviation_usesMinusGlyph() {
        // Uses the Unicode minus sign (U+2212) — tighter typographic match
        // to the "+" glyph in the paired label than ASCII "-".
        let c = ContactSummary(contactId: "c:x", name: "X", cadenceDeviationDays: -2)
        XCTAssertEqual(PeopleTabView.cadenceLabel(for: c), "−2d")
    }
}

// MARK: - PeopleTabView.youSubtitle

final class PeopleYouSubtitleTests: XCTestCase {

    func test_populatedSubtitle_includesObservedMonthsAndInteractions() {
        let line = PeopleTabView.youSubtitle(for: MockData.selfPortrait)
        XCTAssertTrue(line.hasPrefix("Observed 9 months"))
        XCTAssertTrue(line.contains("interactions"))
        XCTAssertTrue(line.contains("·"))
    }

    func test_singleMonth_singleInteraction_readSingular() {
        let p = SelfPortrait(observedMonths: 1, interactionsCount: 1)
        let line = PeopleTabView.youSubtitle(for: p)
        XCTAssertTrue(line.contains("1 month "))
        XCTAssertFalse(line.contains("1 months"))
        XCTAssertTrue(line.contains("1 interaction"))
        XCTAssertFalse(line.contains("1 interactions"))
    }

    func test_emptySubtitle_usesCalmFallback() {
        XCTAssertEqual(
            PeopleTabView.youSubtitle(for: MockData.emptySelfPortrait),
            "Just getting started — no observations yet."
        )
    }
}

// MARK: - PeopleTabView.accessibilityLabel + dossier(for:)

final class PeopleRowA11yAndDossierTests: XCTestCase {

    func test_accessibilityLabel_includesNameAndLastSeen() {
        let label = PeopleTabView.accessibilityLabel(
            for: MockData.contactDad,
            anchor: MockData.anchorDate
        )
        XCTAssertTrue(label.contains("Dad"))
        XCTAssertTrue(label.contains("9d ago"))
        XCTAssertTrue(label.contains("+4d"))
    }

    func test_accessibilityLabel_omitsCadence_whenNone() {
        let label = PeopleTabView.accessibilityLabel(
            for: MockData.contactSam,
            anchor: MockData.anchorDate
        )
        XCTAssertTrue(label.contains("Sam"))
        XCTAssertFalse(label.contains("Cadence"))
    }

    func test_dossierStub_copiesNameAndContactId() {
        let d = PeopleTabView.dossier(for: MockData.contactDad)
        XCTAssertEqual(d.contactId, MockData.contactDad.contactId)
        XCTAssertEqual(d.name, MockData.contactDad.name)
    }
}

// MARK: - ContactSummary: Hashable

final class ContactSummaryHashableTests: XCTestCase {

    func test_hashingIsStableAcrossEqualRows() {
        let a = ContactSummary(contactId: "c:x", name: "X", lastContactTs: 123)
        let b = ContactSummary(contactId: "c:x", name: "X", lastContactTs: 123)
        XCTAssertEqual(a.hashValue, b.hashValue)
    }

    func test_differentIDsHashDifferently() {
        let a = ContactSummary(contactId: "c:a", name: "A")
        let b = ContactSummary(contactId: "c:b", name: "A")
        XCTAssertNotEqual(a, b)
    }

    func test_setRoundTrip() {
        let rows = [MockData.contactDad, MockData.contactMaya, MockData.contactSam]
        let set = Set(rows)
        XCTAssertEqual(set.count, rows.count)
    }
}

// MARK: - ContactDossierView.lastSeenLine

final class DossierLastSeenLineTests: XCTestCase {

    private let anchor = MockData.anchorDate

    private func dossier(daysAgo: Int?) -> ContactDossier {
        ContactDossier(
            contactId: "c:x",
            name: "X",
            lastContactTs: daysAgo.map { Int(anchor.timeIntervalSince1970) - $0 * 86_400 }
        )
    }

    func test_nilLastContact_readsNeverRecorded() {
        XCTAssertEqual(
            ContactDossierView.lastSeenLine(for: dossier(daysAgo: nil), anchor: anchor),
            "No contact recorded yet."
        )
    }

    func test_sameDay_readsToday() {
        XCTAssertEqual(
            ContactDossierView.lastSeenLine(for: dossier(daysAgo: 0), anchor: anchor),
            "Last contact today."
        )
    }

    func test_oneDay_isSingular() {
        XCTAssertEqual(
            ContactDossierView.lastSeenLine(for: dossier(daysAgo: 1), anchor: anchor),
            "Last contact 1 day ago."
        )
    }

    func test_nineDays_isPlural() {
        XCTAssertEqual(
            ContactDossierView.lastSeenLine(for: dossier(daysAgo: 9), anchor: anchor),
            "Last contact 9 days ago."
        )
    }
}

// MARK: - ContactDossierView.commStyleOrFallback

final class DossierCommStyleTests: XCTestCase {

    func test_populatedTemplate_returnsVerbatim() {
        XCTAssertEqual(
            ContactDossierView.commStyleOrFallback(for: MockData.contactDossierDad),
            MockData.contactDossierDad.commTemplate
        )
    }

    func test_nilTemplate_usesFallback() {
        let line = ContactDossierView.commStyleOrFallback(for: MockData.contactDossierSparse)
        XCTAssertTrue(line.lowercased().contains("voice") || line.lowercased().contains("style"))
        XCTAssertFalse(line.contains("!"))
    }

    func test_emptyTemplate_usesFallback() {
        let d = ContactDossier(contactId: "c:e", name: "E", commTemplate: "")
        XCTAssertNotEqual(ContactDossierView.commStyleOrFallback(for: d), "")
    }
}

// MARK: - ContactDossierView.predictedNextLine

final class DossierPredictedNextTests: XCTestCase {

    func test_populated_returnsVerbatim() {
        XCTAssertEqual(
            ContactDossierView.predictedNextLine(for: MockData.contactDossierDad),
            MockData.contactDossierDad.predictedNext
        )
    }

    func test_nil_usesCalmFallback() {
        let line = ContactDossierView.predictedNextLine(for: MockData.contactDossierSparse)
        XCTAssertTrue(line.lowercased().contains("no prediction"))
        XCTAssertFalse(line.contains("!"))
    }
}

// MARK: - ContactDossierView.cadenceFooter

final class DossierCadenceFooterTests: XCTestCase {

    func test_populatedSparklineAndUsual_joinsWithMiddot() {
        let line = ContactDossierView.cadenceFooter(for: MockData.contactDossierDad)
        XCTAssertTrue(line.contains("14 days"))
        XCTAssertTrue(line.contains("·"))
        XCTAssertTrue(line.contains("usual every 5 days"))
    }

    func test_sparklineOnly_noUsual() {
        let d = ContactDossier(
            contactId: "c:s", name: "S",
            usualCadenceDays: nil,
            cadenceSparkline: [0, 1, 2]
        )
        XCTAssertEqual(ContactDossierView.cadenceFooter(for: d), "3 days")
    }

    func test_usualOnly_noSparkline() {
        let d = ContactDossier(
            contactId: "c:u", name: "U",
            usualCadenceDays: 7,
            cadenceSparkline: []
        )
        let line = ContactDossierView.cadenceFooter(for: d)
        XCTAssertTrue(line.hasPrefix("No recent contact data."))
        XCTAssertTrue(line.contains("7 days"))
    }

    func test_noneAtAll_justFallback() {
        let d = ContactDossier(
            contactId: "c:n", name: "N",
            usualCadenceDays: nil,
            cadenceSparkline: []
        )
        XCTAssertEqual(ContactDossierView.cadenceFooter(for: d), "No recent contact data.")
    }

    func test_singleDayCadence_isSingular() {
        let d = ContactDossier(
            contactId: "c:1", name: "1",
            usualCadenceDays: 1,
            cadenceSparkline: [0, 1, 1]
        )
        let line = ContactDossierView.cadenceFooter(for: d)
        XCTAssertTrue(line.contains("usual every 1 day"))
        XCTAssertFalse(line.contains("1 days"))
    }
}

// MARK: - ContactDossierView.sparklinePoints

final class DossierSparklinePointsTests: XCTestCase {

    private let size = CGSize(width: 100, height: 40)

    func test_emptyValues_returnsEmpty() {
        XCTAssertEqual(ContactDossierView.sparklinePoints(for: [], in: size), [])
    }

    func test_singleValue_returnsOneMidPoint() {
        let pts = ContactDossierView.sparklinePoints(for: [5], in: size)
        XCTAssertEqual(pts.count, 1)
        XCTAssertEqual(pts[0], CGPoint(x: 0, y: 20))
    }

    func test_multipleValues_xEvenlySpannedEnd_toEnd() {
        let pts = ContactDossierView.sparklinePoints(for: [0, 1, 2, 3], in: size)
        XCTAssertEqual(pts.count, 4)
        XCTAssertEqual(pts.first?.x, 0)
        XCTAssertEqual(pts.last?.x, 100)
        // Evenly spaced: 0, 33.33, 66.66, 100
        XCTAssertEqual(pts[1].x, 100.0 / 3.0, accuracy: 0.001)
        XCTAssertEqual(pts[2].x, 200.0 / 3.0, accuracy: 0.001)
    }

    func test_yScaling_minAtBottom_maxAtTop() {
        let pts = ContactDossierView.sparklinePoints(for: [0, 5, 10], in: size)
        XCTAssertEqual(pts.count, 3)
        // min (0) → y = height (bottom of the rect)
        XCTAssertEqual(pts[0].y, size.height, accuracy: 0.001)
        // max (10) → y = 0 (top)
        XCTAssertEqual(pts[2].y, 0, accuracy: 0.001)
        // mid (5) → y = height / 2
        XCTAssertEqual(pts[1].y, size.height / 2, accuracy: 0.001)
    }

    func test_flatSeries_rendersAtMidY() {
        let pts = ContactDossierView.sparklinePoints(for: [7, 7, 7, 7], in: size)
        for p in pts {
            XCTAssertEqual(p.y, size.height / 2, accuracy: 0.001)
        }
    }

    func test_fixturePoints_countMatchesInput() {
        let values = MockData.contactDossierDad.cadenceSparkline
        let pts = ContactDossierView.sparklinePoints(for: values, in: size)
        XCTAssertEqual(pts.count, values.count)
    }
}

// MARK: - ContactDossierView.sparklineAccessibilityLabel

final class DossierSparklineA11yTests: XCTestCase {

    func test_emptySeries_readsNoData() {
        XCTAssertEqual(
            ContactDossierView.sparklineAccessibilityLabel(for: MockData.contactDossierSparse),
            "No recent cadence data."
        )
    }

    func test_populatedSeries_summarizesDaysAndContacts() {
        let line = ContactDossierView.sparklineAccessibilityLabel(for: MockData.contactDossierDad)
        XCTAssertTrue(line.contains("14 days"))
        // Sparkline [0, 1, 0, 0, 2, 1, 0, 0, 3, 1, 0, 0, 0, 0] → 8 total
        XCTAssertTrue(line.contains("8 contacts"))
    }

    func test_singleContact_isSingular() {
        let d = ContactDossier(
            contactId: "c:1", name: "1",
            cadenceSparkline: [0, 0, 1, 0]
        )
        let line = ContactDossierView.sparklineAccessibilityLabel(for: d)
        XCTAssertTrue(line.contains("1 contact."))
        XCTAssertFalse(line.contains("1 contacts"))
    }
}

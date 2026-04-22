//
//  YouTabViewTests.swift
//  Life OS — You tab logic tests
//
//  Verifies:
//    1. `MockData.selfPortrait` matches the wireframe shape (4 sections,
//       populated rows) so previews/tests stay grounded in DESIGN.md.
//    2. `MockData.emptySelfPortrait` round-trips empty (every list = []).
//    3. `YouSection.allCases` is locked to the wireframe order from
//       DESIGN.md §"You: self-portrait. Sections: When you're at your
//       best · How you write · Your routines · Drifting".
//    4. The pure helpers exposed on `YouTabView` (`headerLine`,
//       `routineSubtitle`, `driftingDetail`) produce the exact copy
//       called for by DESIGN.md (calm tone, no alarm icons).
//    5. Empty-state copy on every `YouSection` case is filled in.
//

import XCTest
@testable import LifeOS

// MARK: - MockData shape

final class MockSelfPortraitShapeTests: XCTestCase {

    func test_selfPortrait_headerCountersArePopulated() {
        let p = MockData.selfPortrait
        XCTAssertGreaterThan(p.observedMonths, 0)
        XCTAssertGreaterThan(p.interactionsCount, 0)
        XCTAssertGreaterThan(p.confidencePct, 0)
    }

    func test_selfPortrait_everySectionHasRows() {
        let p = MockData.selfPortrait
        XCTAssertFalse(p.whenAtBest.isEmpty,   "WHEN YOU'RE AT YOUR BEST must have placeholder rows")
        XCTAssertFalse(p.howYouWrite.isEmpty,  "HOW YOU WRITE must have placeholder rows")
        XCTAssertFalse(p.yourRoutines.isEmpty, "YOUR ROUTINES must have placeholder rows")
        XCTAssertFalse(p.drifting.isEmpty,     "DRIFTING must have placeholder rows")
    }

    func test_selfPortrait_includesAtLeastOneUndetectedRoutine() {
        // The undetected routine row exercises the "tertiary text +
        // 'Not detected yet.'" path that DESIGN.md calls for instead
        // of hiding the row entirely.
        let undetected = MockData.selfPortrait.yourRoutines.contains { !$0.detected }
        XCTAssertTrue(undetected, "Routines mock should include a detected=false row")
    }

    func test_selfPortrait_driftingRowsMatchDESIGNmdShape() {
        // DESIGN.md example: "Dad · 9 days (usual 5)". Each row therefore
        // needs a name, a positive days-since-last, and a usual cadence.
        for contact in MockData.selfPortrait.drifting {
            XCTAssertFalse(contact.name.isEmpty)
            XCTAssertGreaterThan(contact.daysSinceLast, 0)
            XCTAssertGreaterThan(contact.usualCadenceDays, 0)
        }
    }

    func test_emptySelfPortrait_isAllZeroAndAllEmpty() {
        let p = MockData.emptySelfPortrait
        XCTAssertEqual(p.observedMonths, 0)
        XCTAssertEqual(p.interactionsCount, 0)
        XCTAssertEqual(p.confidencePct, 0)
        XCTAssertTrue(p.whenAtBest.isEmpty)
        XCTAssertTrue(p.howYouWrite.isEmpty)
        XCTAssertTrue(p.yourRoutines.isEmpty)
        XCTAssertTrue(p.drifting.isEmpty)
    }
}

// MARK: - Section order (locked to wireframe)

final class YouSectionOrderTests: XCTestCase {

    func test_sectionOrder_matchesDESIGNmd() {
        // DESIGN.md §"You: self-portrait. Sections: When you're at your
        // best · How you write · Your routines · Drifting."
        XCTAssertEqual(
            YouSection.allCases,
            [.whenAtBest, .howYouWrite, .yourRoutines, .drifting]
        )
    }

    func test_sectionTitles_matchUppercaseLabels() {
        // The view renders `section.rawValue` as the section header.
        // Lock the strings so a typo in the enum invalidates tests.
        XCTAssertEqual(YouSection.whenAtBest.rawValue,   "WHEN YOU'RE AT YOUR BEST")
        XCTAssertEqual(YouSection.howYouWrite.rawValue,  "HOW YOU WRITE")
        XCTAssertEqual(YouSection.yourRoutines.rawValue, "YOUR ROUTINES")
        XCTAssertEqual(YouSection.drifting.rawValue,     "DRIFTING")
    }

    func test_sectionIDs_areUnique() {
        let ids = YouSection.allCases.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count)
    }

    func test_everySection_hasNonEmptyEmptyStateCopy() {
        for section in YouSection.allCases {
            XCTAssertFalse(section.emptyTitle.isEmpty,    "\(section) is missing an empty-state title")
            XCTAssertFalse(section.emptySubtitle.isEmpty, "\(section) is missing an empty-state subtitle")
        }
    }

    func test_emptyStateCopy_doesNotShout() {
        // DESIGN.md §"Calm over urgent" + §"Empty states" — no caps,
        // no exclamation. Spot-check that the copy doesn't shout or
        // beg the user to onboard with a "Get started!" CTA.
        for section in YouSection.allCases {
            XCTAssertFalse(section.emptyTitle.contains("!"))
            XCTAssertFalse(section.emptyTitle.uppercased() == section.emptyTitle,
                           "\(section).emptyTitle reads as SHOUTING: \(section.emptyTitle)")
            XCTAssertFalse(section.emptyTitle.lowercased().contains("get started"))
        }
    }
}

// MARK: - YouTabView.headerLine

final class YouTabHeaderLineTests: XCTestCase {

    func test_populatedHeader_includesObservedMonths_andInteractionCount() {
        let line = YouTabView.headerLine(for: MockData.selfPortrait)
        XCTAssertTrue(line.hasPrefix("Observed 9 months"),
                      "header should lead with months count, was: \(line)")
        XCTAssertTrue(line.contains("interactions"),
                      "header should include 'interactions', was: \(line)")
        XCTAssertTrue(line.contains("·"),
                      "header should use middot separator per DESIGN.md, was: \(line)")
    }

    func test_singleMonth_isSingular() {
        let p = SelfPortrait(observedMonths: 1, interactionsCount: 12)
        let line = YouTabView.headerLine(for: p)
        XCTAssertTrue(line.contains("1 month "), "expected singular 'month', got: \(line)")
        XCTAssertFalse(line.contains("1 months"))
    }

    func test_singleInteraction_isSingular() {
        let p = SelfPortrait(observedMonths: 2, interactionsCount: 1)
        let line = YouTabView.headerLine(for: p)
        XCTAssertTrue(line.contains("1 interaction"))
        XCTAssertFalse(line.contains("1 interactions"))
    }

    func test_emptyHeader_usesCalmFallback() {
        let line = YouTabView.headerLine(for: MockData.emptySelfPortrait)
        // DESIGN.md §"Calm over urgent" — no "no data" alarm copy.
        XCTAssertEqual(line, "Just getting started — no observations yet.")
    }
}

// MARK: - YouTabView.routineSubtitle

final class YouTabRoutineSubtitleTests: XCTestCase {

    func test_undetectedRoutine_saysNotDetectedYet() {
        let r = Routine(name: "Walking", detected: false)
        XCTAssertEqual(YouTabView.routineSubtitle(for: r), "Not detected yet.")
    }

    func test_detectedRoutine_returnsDescription() {
        let r = Routine(name: "Run", detected: true, description: "9:15 – 9:45 most weeks")
        XCTAssertEqual(YouTabView.routineSubtitle(for: r), "9:15 – 9:45 most weeks")
    }

    func test_detectedRoutine_withNilDescription_fallsBackToDetected() {
        let r = Routine(name: "Reading", detected: true, description: nil)
        XCTAssertEqual(YouTabView.routineSubtitle(for: r), "Detected.")
    }
}

// MARK: - YouTabView.driftingDetail

final class YouTabDriftingDetailTests: XCTestCase {

    func test_driftingDetail_matchesDESIGNmdExample() {
        // DESIGN.md: "Dad · 9 days (usual 5) not '⚠ OVERDUE'".
        let dad = DriftingContact(contactId: "c:dad", name: "Dad", daysSinceLast: 9, usualCadenceDays: 5)
        XCTAssertEqual(YouTabView.driftingDetail(for: dad), "9 days (usual 5)")
    }

    func test_driftingDetail_singularDay() {
        let c = DriftingContact(contactId: "c:x", name: "X", daysSinceLast: 1, usualCadenceDays: 3)
        XCTAssertEqual(YouTabView.driftingDetail(for: c), "1 day (usual 3)")
    }

    func test_driftingDetail_doesNotUseAlarmCopy() {
        // Calm-over-urgent sanity check.
        let c = DriftingContact(contactId: "c:y", name: "Y", daysSinceLast: 30, usualCadenceDays: 5)
        let detail = YouTabView.driftingDetail(for: c)
        XCTAssertFalse(detail.contains("OVERDUE"))
        XCTAssertFalse(detail.contains("!"))
        XCTAssertFalse(detail.contains("⚠"))
    }
}

//
//  NowTabViewTests.swift
//  Life OS — Now tab logic tests
//
//  Verifies the pure helpers exposed on `NowTabView` and `MomentCardView`
//  that drive the IA described in DESIGN.md. SwiftUI view bodies aren't
//  introspected here (no third-party deps); the contract we care about
//  lives in the static helpers and in the `MockData` fixture shape.
//

import XCTest
@testable import LifeOS

// MARK: - MockData sanity

final class MockDataShapeTests: XCTestCase {

    func test_feed_hasPopulatedBuckets() {
        let feed = MockData.feed
        XCTAssertEqual(feed.pending.count, 3)
        XCTAssertEqual(feed.scheduled.count, 2)
        XCTAssertEqual(feed.done.count, 1)
    }

    func test_pendingMoments_areSuggested() {
        for m in MockData.feed.pending {
            XCTAssertEqual(m.state, .suggested, "pending bucket must hold .suggested moments")
        }
    }

    func test_doneBucket_holdsDoneMoments() {
        for m in MockData.feed.done {
            XCTAssertEqual(m.state, .done)
        }
    }

    func test_emptyFeed_isAllEmpty() {
        let f = MockData.emptyFeed
        XCTAssertTrue(f.pending.isEmpty)
        XCTAssertTrue(f.scheduled.isEmpty)
        XCTAssertTrue(f.done.isEmpty)
    }

    func test_mockMoments_haveUniqueIDs() {
        let ids = [
            MockData.momentCadence.id,
            MockData.momentRelationship.id,
            MockData.momentTemporal.id,
            MockData.momentRoutine.id,
            MockData.momentSpatial.id,
            MockData.momentDone.id,
        ]
        XCTAssertEqual(Set(ids).count, ids.count, "Mock moments must have distinct IDs")
    }

    func test_scheduledMoments_carryTriggerOrScheduledFor() {
        for m in MockData.feed.scheduled {
            XCTAssertTrue(
                m.contextTrigger != nil || m.scheduledFor != nil,
                "scheduled bucket must always know when/why to fire"
            )
        }
    }
}

// MARK: - NowTabView section splitting

final class NowTabViewSectionTests: XCTestCase {

    func test_nowSection_clampsToThree() {
        let feed = MomentFeed(
            pending: [
                MockData.momentCadence,
                MockData.momentRelationship,
                MockData.momentTemporal,
                // Extra one should be trimmed.
                MockData.momentCadence,
            ],
            scheduled: [],
            done: []
        )
        XCTAssertEqual(NowTabView.nowSectionMoments(from: feed).count, 3)
    }

    func test_nowSection_honorsCustomLimit() {
        let feed = MockData.feed
        XCTAssertEqual(NowTabView.nowSectionMoments(from: feed, limit: 2).count, 2)
    }

    func test_nowSection_empty_whenNoPending() {
        XCTAssertEqual(NowTabView.nowSectionMoments(from: MockData.emptyFeed), [])
    }

    func test_upNextSection_returnsScheduledBucketInOrder() {
        let ordered = NowTabView.upNextSectionMoments(from: MockData.feed)
        XCTAssertEqual(ordered.map(\.id), MockData.feed.scheduled.map(\.id))
    }

    func test_doneTodaySection_returnsDoneBucket() {
        let done = NowTabView.doneTodaySectionMoments(from: MockData.feed)
        XCTAssertEqual(done.map(\.id), MockData.feed.done.map(\.id))
    }

    func test_nowSectionLimitConstant_isThree() {
        // DESIGN.md calls for "2-3 cards" in NOW; lock the upper bound.
        XCTAssertEqual(NowTabView.nowSectionLimit, 3)
    }
}

// MARK: - NowTabView.upNextPrefix

final class NowTabUpNextPrefixTests: XCTestCase {

    func test_prefersContextTrigger_overScheduledFor() {
        // momentSpatial has both; trigger should win.
        let m = MockData.momentSpatial
        XCTAssertEqual(NowTabView.upNextPrefix(for: m), "arrive:home")
    }

    func test_fallsBackToScheduledFor_whenNoTrigger() {
        // Build a moment with scheduledFor but no contextTrigger.
        let m = Moment(
            id: "t:sched",
            createdAt: Date(timeIntervalSince1970: 1_777_200_000),
            expiresAt: Date(timeIntervalSince1970: 1_777_459_200),
            insight: "Test",
            evidence: [],
            evidenceHash: "sha256:t",
            proposedAction: Action(kind: .nudge),
            sourceInsightType: .temporal,
            state: .suggested,
            scheduledFor: Date(timeIntervalSince1970: 1_777_230_000),
            contextTrigger: nil
        )
        let prefix = NowTabView.upNextPrefix(for: m)
        // 1_777_230_000 → 2026-04-26 13:00 UTC. Format is `HH:mm` in local
        // time; we assert shape rather than exact string to stay timezone-agnostic.
        XCTAssertEqual(prefix.count, 5)
        XCTAssertEqual(prefix[prefix.index(prefix.startIndex, offsetBy: 2)], ":")
    }

    func test_fallbackWhenNeitherSet() {
        let m = Moment(
            id: "t:neither",
            createdAt: Date(timeIntervalSince1970: 1_777_200_000),
            expiresAt: Date(timeIntervalSince1970: 1_777_459_200),
            insight: "Test",
            evidence: [],
            evidenceHash: "sha256:t",
            proposedAction: Action(kind: .nudge),
            sourceInsightType: .temporal,
            state: .suggested,
            scheduledFor: nil,
            contextTrigger: nil
        )
        XCTAssertEqual(NowTabView.upNextPrefix(for: m), "later")
    }
}

// MARK: - MomentCardView.primaryActionLabel

final class MomentCardPrimaryLabelTests: XCTestCase {

    func test_everyActionKind_hasNonEmptyLabel() {
        // Guards against adding an ActionKind case without updating the
        // primary-button mapping.
        for kind in ActionKind.allCases {
            let label = MomentCardView.primaryActionLabel(for: kind)
            XCTAssertFalse(label.isEmpty, "ActionKind.\(kind) has no primary label")
        }
    }

    func test_draftMessage_label_matchesDESIGNmd() {
        // DESIGN.md §Action button hierarchy explicitly lists "[Start a message]".
        XCTAssertEqual(
            MomentCardView.primaryActionLabel(for: .draftMessage),
            "Start a message"
        )
    }

    func test_nudge_usesStartTimer_perDESIGNmd() {
        // DESIGN.md example lists "[Start timer]".
        XCTAssertEqual(MomentCardView.primaryActionLabel(for: .nudge), "Start timer")
    }

    func test_sendMessage_usesSend_perDESIGNmd() {
        // DESIGN.md example lists "[Send]".
        XCTAssertEqual(MomentCardView.primaryActionLabel(for: .sendMessage), "Send")
    }

    func test_labelsAreUnique_acrossKinds() {
        let labels = ActionKind.allCases.map { MomentCardView.primaryActionLabel(for: $0) }
        XCTAssertEqual(Set(labels).count, labels.count, "Two ActionKinds share a primary label")
    }
}

// MARK: - MomentCardView.draftBody / .hasDraft

final class MomentCardDraftBlockTests: XCTestCase {

    func test_draftBody_returnsBody_whenDraftMessageWithBody() {
        let body = MomentCardView.draftBody(from: MockData.momentCadence)
        XCTAssertNotNil(body)
        XCTAssertTrue(body?.contains("Sam") == true)
    }

    func test_draftBody_nil_whenNotDraftMessage() {
        // Temporal nudge has no draft.
        XCTAssertNil(MomentCardView.draftBody(from: MockData.momentTemporal))
    }

    func test_draftBody_nil_whenBodyMissing() {
        let m = Moment(
            id: "t:nobody",
            createdAt: Date(),
            expiresAt: Date(timeIntervalSinceNow: 3600),
            insight: "Test",
            evidence: [],
            evidenceHash: "sha256:t",
            proposedAction: Action(kind: .draftMessage, params: ["channel": AnyCodable("imessage")]),
            sourceInsightType: .cadence
        )
        XCTAssertNil(MomentCardView.draftBody(from: m))
    }

    func test_draftBody_nil_whenBodyWhitespaceOnly() {
        let m = Moment(
            id: "t:whitespace",
            createdAt: Date(),
            expiresAt: Date(timeIntervalSinceNow: 3600),
            insight: "Test",
            evidence: [],
            evidenceHash: "sha256:t",
            proposedAction: Action(kind: .draftMessage, params: ["body": AnyCodable("   \n  ")]),
            sourceInsightType: .cadence
        )
        XCTAssertNil(MomentCardView.draftBody(from: m))
    }

    func test_hasDraft_matchesDraftBodyPresence() {
        XCTAssertTrue(MomentCardView.hasDraft(MockData.momentCadence))
        XCTAssertFalse(MomentCardView.hasDraft(MockData.momentTemporal))
    }
}

// MARK: - MomentCardView.evidenceLinkLabel + accessibility

final class MomentCardCopyTests: XCTestCase {

    func test_evidenceLabel_pluralForMany() {
        XCTAssertEqual(
            MomentCardView.evidenceLinkLabel(for: MockData.momentCadence),
            "From 3 sources"
        )
    }

    func test_evidenceLabel_singularForOne() {
        XCTAssertEqual(
            MomentCardView.evidenceLinkLabel(for: MockData.momentTemporal),
            "From 1 source"
        )
    }

    func test_evidenceLabel_handlesZero() {
        let m = Moment(
            id: "t:noevidence",
            createdAt: Date(),
            expiresAt: Date(timeIntervalSinceNow: 3600),
            insight: "Test",
            evidence: [],
            evidenceHash: "sha256:t",
            proposedAction: Action(kind: .nudge),
            sourceInsightType: .temporal
        )
        XCTAssertEqual(MomentCardView.evidenceLinkLabel(for: m), "From 0 sources")
    }

    func test_accessibilityLabel_containsInsightAndAction() {
        let label = MomentCardView.accessibilityLabel(for: MockData.momentCadence)
        XCTAssertTrue(label.contains(MockData.momentCadence.insight))
        XCTAssertTrue(label.contains("Start a message"))
    }
}

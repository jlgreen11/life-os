//
//  MockData.swift
//  Life OS — Preview / test fixtures
//
//  Shared placeholder `Moment` instances used by SwiftUI previews and by
//  XCTest layout / helper tests. Everything here is deliberately static
//  and Swift-synthesized (no bundle resource loading) so it works both
//  inside an Xcode preview canvas and a headless test host.
//
//  The shape of each `Moment` mirrors a realistic API payload from the
//  v2 backend (`api/schemas.py::MomentOut`) — the Previews directory
//  is the right home because these fixtures exist to make the UI
//  tangible, not to drive production code.
//

import Foundation

enum MockData {

    // MARK: - Sample Moments (pending / NOW bucket)

    /// Cadence insight with a drafted reach-out message — exercises the
    /// "insight + evidence + draft + primary action" card layout.
    static let momentCadence = Moment(
        id: "mock:cadence-01",
        createdAt: Date(timeIntervalSince1970: 1_777_204_800),
        expiresAt: Date(timeIntervalSince1970: 1_777_464_000),
        insight: "You usually text Sam every 4 days — it's been 7.",
        evidence: [
            "Last 8 conversations averaged 4.1 days apart.",
            "Most recent inbound: 2026-04-15T14:22:00Z.",
            "Sam's last three replies came within 2 hours.",
        ],
        evidenceHash: "sha256:mock-cadence-01",
        proposedAction: Action(
            kind: .draftMessage,
            params: [
                "body": AnyCodable(
                    "Hey Sam — saw the link you sent earlier this week and it made me think of you. How's your week going?"
                ),
                "channel": AnyCodable("imessage"),
                "contact_id": AnyCodable("contact:sam-okonkwo"),
            ]
        ),
        sourceInsightType: .cadence,
        state: .suggested,
        confidence: 0.82,
        feedbackWeight: 1.04
    )

    /// Relationship drift insight — exercises the "dismiss-heavy" card with
    /// a short draft and two pieces of evidence.
    static let momentRelationship = Moment(
        id: "mock:relationship-01",
        createdAt: Date(timeIntervalSince1970: 1_777_208_400),
        expiresAt: Date(timeIntervalSince1970: 1_777_467_600),
        insight: "Drifting: Maya — usual cadence 5 days, now 11.",
        evidence: [
            "Last contact 2026-04-11.",
            "Cadence window shifted 120% beyond norm.",
        ],
        evidenceHash: "sha256:mock-relationship-01",
        proposedAction: Action(
            kind: .draftMessage,
            params: [
                "body": AnyCodable("Hey Maya — been a minute. How are you?"),
                "channel": AnyCodable("imessage"),
                "contact_id": AnyCodable("contact:maya"),
            ]
        ),
        sourceInsightType: .relationship,
        state: .suggested,
        confidence: 0.71,
        feedbackWeight: 1.00
    )

    /// Temporal nudge — exercises a Moment with no draft block (non-message action).
    static let momentTemporal = Moment(
        id: "mock:temporal-01",
        createdAt: Date(timeIntervalSince1970: 1_777_212_000),
        expiresAt: Date(timeIntervalSince1970: 1_777_471_200),
        insight: "You've been heads-down for 90 minutes — time for a break?",
        evidence: [
            "No keyboard idle window in the last 90 minutes.",
        ],
        evidenceHash: "sha256:mock-temporal-01",
        proposedAction: Action(kind: .nudge),
        sourceInsightType: .temporal,
        state: .suggested,
        confidence: 0.65,
        feedbackWeight: 1.00
    )

    // MARK: - Sample Moments (scheduled / UP NEXT bucket)

    static let momentRoutine = Moment(
        id: "mock:routine-01",
        createdAt: Date(timeIntervalSince1970: 1_777_200_000),
        expiresAt: Date(timeIntervalSince1970: 1_777_459_200),
        insight: "Tuesday morning run — your usual routine starts at 9:30.",
        evidence: [
            "4 of last 5 Tuesdays recorded a run between 9:15 and 9:45.",
        ],
        evidenceHash: "sha256:mock-routine-01",
        proposedAction: Action(kind: .setReminder),
        sourceInsightType: .routine,
        state: .suggested,
        scheduledFor: Date(timeIntervalSince1970: 1_777_230_000),
        contextTrigger: ContextTrigger(expression: "time:09:30"),
        confidence: 0.74,
        feedbackWeight: 1.00
    )

    static let momentSpatial = Moment(
        id: "mock:spatial-01",
        createdAt: Date(timeIntervalSince1970: 1_777_201_000),
        expiresAt: Date(timeIntervalSince1970: 1_777_460_200),
        insight: "When you're home, you usually read for 20 minutes.",
        evidence: [
            "Observed in 9 of the last 14 evenings.",
        ],
        evidenceHash: "sha256:mock-spatial-01",
        proposedAction: Action(kind: .nudge),
        sourceInsightType: .spatial,
        state: .suggested,
        scheduledFor: Date(timeIntervalSince1970: 1_777_280_000),
        contextTrigger: ContextTrigger(expression: "arrive:home"),
        confidence: 0.68,
        feedbackWeight: 1.00
    )

    // MARK: - Sample Moments (done / DONE TODAY bucket)

    static let momentDone = Moment(
        id: "mock:done-01",
        createdAt: Date(timeIntervalSince1970: 1_777_180_000),
        expiresAt: Date(timeIntervalSince1970: 1_777_439_200),
        insight: "Reached out to Devon.",
        evidence: [
            "Accepted at 10:42 AM.",
        ],
        evidenceHash: "sha256:mock-done-01",
        proposedAction: Action(
            kind: .sendMessage,
            params: ["contact_id": AnyCodable("contact:devon")]
        ),
        sourceInsightType: .cadence,
        state: .done,
        confidence: 0.88,
        feedbackWeight: 1.06,
        stateHistory: [
            StateHistoryEntry(
                fromState: nil,
                toState: .suggested,
                ts: Date(timeIntervalSince1970: 1_777_180_000),
                annotation: nil
            ),
            StateHistoryEntry(
                fromState: .suggested,
                toState: .accepted,
                ts: Date(timeIntervalSince1970: 1_777_183_600),
                annotation: "user accepted"
            ),
            StateHistoryEntry(
                fromState: .accepted,
                toState: .done,
                ts: Date(timeIntervalSince1970: 1_777_190_000),
                annotation: "outbox dispatched"
            ),
        ]
    )

    // MARK: - Full feed

    /// `MomentFeed` matching the `/api/now` response shape. Three pending,
    /// two scheduled, one done — enough surface area to exercise every
    /// layout branch in `NowTabView` and `MomentCardView`.
    static let feed = MomentFeed(
        pending: [momentCadence, momentRelationship, momentTemporal],
        scheduled: [momentRoutine, momentSpatial],
        done: [momentDone]
    )

    /// Empty feed — used by preview/tests to exercise the empty-state path.
    static let emptyFeed = MomentFeed(pending: [], scheduled: [], done: [])
}

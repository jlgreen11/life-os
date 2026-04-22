//
//  MomentTests.swift
//  Life OS — Swift Moment decode tests
//
//  Verifies the Codable Moment mirror in `ios/LifeOS/Models/Moment.swift`
//  decodes the same JSON shape the v2 API emits (api/schemas.py::MomentOut).
//
//  The fixtures live both at `tests/fixtures/v2_moment_*.json` (shared with
//  Python tests) AND inline as raw string literals below — the inline copy
//  keeps these tests self-contained so they don't depend on Xcode bundle
//  resource setup. Any edit to the JSON should land in BOTH places.
//

import XCTest
@testable import LifeOS

// MARK: - Inline fixtures (kept in sync with tests/fixtures/v2_moment_*.json)

private let sampleMomentJSON = #"""
{
  "id": "01HXYZ1234567890ABCDEFGHIJ",
  "created_at": 1777204800,
  "expires_at": 1777464000,
  "insight": "You usually text Sam every 4 days — it's been 7.",
  "evidence": [
    "Last 8 conversations averaged 4.1 days apart.",
    "Most recent inbound: 2026-04-15T14:22:00Z."
  ],
  "evidence_hash": "sha256:f1d2e3c4b5a697887766554433221100ffeeddccbbaa9988",
  "proposed_action": {
    "kind": "draft_message",
    "params": {
      "channel": "imessage",
      "contact_id": "contact:sam-okonkwo",
      "body": "Hey Sam — saw the link you sent and it made me think of you. How's your week?",
      "draft_confidence": 0.78,
      "tags": ["reach-out", "warm-tone"],
      "metadata": {"tokens_in": 142, "tokens_out": 31}
    }
  },
  "source_insight_type": "cadence",
  "state": "suggested",
  "scheduled_for": 1777208400,
  "context_trigger": "calendar:gap>30m",
  "snooze_until": null,
  "confidence": 0.82,
  "feedback_weight": 1.04,
  "state_history": [
    {
      "from_state": null,
      "to_state": "suggested",
      "ts": 1777204800,
      "annotation": "created by cadence producer"
    }
  ]
}
"""#

private let minimalMomentJSON = #"""
{
  "id": "01HXYZMINIMAL00000000000000",
  "created_at": 1777204800,
  "expires_at": 1777464000,
  "insight": "Routine deviation: you usually run on Tuesday mornings.",
  "evidence_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "proposed_action": {"kind": "nudge"},
  "source_insight_type": "routine"
}
"""#

private let feedJSON = #"""
{
  "pending": [
    {
      "id": "01HXYZPENDING00000000000001",
      "created_at": 1777204800,
      "expires_at": 1777464000,
      "insight": "Drifting: Maya — usual cadence 5 days, now 11.",
      "evidence": ["Last contact 2026-04-11."],
      "evidence_hash": "sha256:11111111111111111111111111111111",
      "proposed_action": {
        "kind": "draft_message",
        "params": {"channel": "imessage", "contact_id": "contact:maya"}
      },
      "source_insight_type": "relationship",
      "state": "suggested",
      "confidence": 0.71,
      "feedback_weight": 1.00,
      "state_history": [
        {"from_state": null, "to_state": "suggested", "ts": 1777204800, "annotation": null}
      ]
    }
  ],
  "scheduled": [
    {
      "id": "01HXYZSCHEDULED0000000000002",
      "created_at": 1777200000,
      "expires_at": 1777459200,
      "scheduled_for": 1777230000,
      "context_trigger": "time:09:30",
      "insight": "Tuesday morning run reminder.",
      "evidence_hash": "sha256:22222222222222222222222222222222",
      "proposed_action": {"kind": "nudge", "params": {}},
      "source_insight_type": "temporal",
      "state": "suggested",
      "confidence": 0.65,
      "feedback_weight": 1.00,
      "state_history": [
        {"from_state": null, "to_state": "suggested", "ts": 1777200000, "annotation": null}
      ]
    }
  ],
  "done": [
    {
      "id": "01HXYZDONE00000000000000003",
      "created_at": 1777180000,
      "expires_at": 1777439200,
      "insight": "Reach out to Devon.",
      "evidence_hash": "sha256:33333333333333333333333333333333",
      "proposed_action": {"kind": "send_message", "params": {"contact_id": "contact:devon"}},
      "source_insight_type": "cadence",
      "state": "done",
      "confidence": 0.88,
      "feedback_weight": 1.06,
      "state_history": [
        {"from_state": null, "to_state": "suggested", "ts": 1777180000, "annotation": null},
        {"from_state": "suggested", "to_state": "accepted", "ts": 1777183600, "annotation": "user accepted"},
        {"from_state": "accepted", "to_state": "done", "ts": 1777190000, "annotation": "outbox dispatched"}
      ]
    }
  ]
}
"""#

// MARK: - Enum raw-value tests (lock the Python contract)

final class MomentEnumWireFormatTests: XCTestCase {

    func test_momentState_rawValues_matchPython() {
        XCTAssertEqual(MomentState.suggested.rawValue, "suggested")
        XCTAssertEqual(MomentState.accepted.rawValue, "accepted")
        XCTAssertEqual(MomentState.dismissed.rawValue, "dismissed")
        XCTAssertEqual(MomentState.snoozed.rawValue, "snoozed")
        XCTAssertEqual(MomentState.done.rawValue, "done")
        XCTAssertEqual(MomentState.expired.rawValue, "expired")
    }

    func test_momentState_hasExactlySixCases() {
        XCTAssertEqual(MomentState.allCases.count, 6)
    }

    func test_insightType_rawValues_matchPython() {
        XCTAssertEqual(InsightType.cadence.rawValue, "cadence")
        XCTAssertEqual(InsightType.relationship.rawValue, "relationship")
        XCTAssertEqual(InsightType.temporal.rawValue, "temporal")
        XCTAssertEqual(InsightType.spatial.rawValue, "spatial")
        XCTAssertEqual(InsightType.commTemplate.rawValue, "comm_template")
        XCTAssertEqual(InsightType.routine.rawValue, "routine")
    }

    func test_insightType_hasExactlySixCases() {
        XCTAssertEqual(InsightType.allCases.count, 6)
    }

    func test_actionKind_rawValues_matchPython() {
        XCTAssertEqual(ActionKind.draftMessage.rawValue, "draft_message")
        XCTAssertEqual(ActionKind.sendMessage.rawValue, "send_message")
        XCTAssertEqual(ActionKind.scheduleBlock.rawValue, "schedule_block")
        XCTAssertEqual(ActionKind.archiveEvent.rawValue, "archive_event")
        XCTAssertEqual(ActionKind.nudge.rawValue, "nudge")
        XCTAssertEqual(ActionKind.setReminder.rawValue, "set_reminder")
        XCTAssertEqual(ActionKind.createCalendarEntry.rawValue, "create_calendar_entry")
        XCTAssertEqual(ActionKind.noteObservation.rawValue, "note_observation")
    }

    func test_actionKind_hasExactlyEightCases() {
        XCTAssertEqual(ActionKind.allCases.count, 8)
    }
}

// MARK: - Sample (full) Moment decode

final class MomentSampleDecodeTests: XCTestCase {

    private func decodeSample() throws -> Moment {
        let data = sampleMomentJSON.data(using: .utf8)!
        return try Moment.decoder().decode(Moment.self, from: data)
    }

    func test_decodes_topLevelScalars() throws {
        let m = try decodeSample()
        XCTAssertEqual(m.id, "01HXYZ1234567890ABCDEFGHIJ")
        XCTAssertEqual(m.insight, "You usually text Sam every 4 days — it's been 7.")
        XCTAssertEqual(m.evidenceHash, "sha256:f1d2e3c4b5a697887766554433221100ffeeddccbbaa9988")
        XCTAssertEqual(m.confidence, 0.82, accuracy: 0.0001)
        XCTAssertEqual(m.feedbackWeight, 1.04, accuracy: 0.0001)
    }

    func test_decodes_unixEpochAsDate() throws {
        let m = try decodeSample()
        XCTAssertEqual(m.createdAt.timeIntervalSince1970, 1_777_204_800, accuracy: 0.0001)
        XCTAssertEqual(m.expiresAt.timeIntervalSince1970, 1_777_464_000, accuracy: 0.0001)
        XCTAssertEqual(m.scheduledFor?.timeIntervalSince1970, 1_777_208_400)
        XCTAssertNil(m.snoozeUntil)
    }

    func test_decodes_evidenceList() throws {
        let m = try decodeSample()
        XCTAssertEqual(m.evidence.count, 2)
        XCTAssertEqual(m.evidence[0], "Last 8 conversations averaged 4.1 days apart.")
    }

    func test_decodes_state_and_sourceInsightType() throws {
        let m = try decodeSample()
        XCTAssertEqual(m.state, .suggested)
        XCTAssertEqual(m.sourceInsightType, .cadence)
    }

    func test_decodes_proposedAction_kind_andStringParams() throws {
        let m = try decodeSample()
        XCTAssertEqual(m.proposedAction.kind, .draftMessage)
        XCTAssertEqual(m.proposedAction.params["channel"]?.value as? String, "imessage")
        XCTAssertEqual(m.proposedAction.params["contact_id"]?.value as? String, "contact:sam-okonkwo")
    }

    func test_decodes_proposedAction_numericAndArrayAndDictParams() throws {
        let m = try decodeSample()
        XCTAssertEqual(m.proposedAction.params["draft_confidence"]?.value as? Double, 0.78)
        let tags = m.proposedAction.params["tags"]?.value as? [Any?]
        XCTAssertEqual(tags?.count, 2)
        XCTAssertEqual(tags?[0] as? String, "reach-out")
        let meta = m.proposedAction.params["metadata"]?.value as? [String: Any?]
        XCTAssertEqual(meta?["tokens_in"] as? Int, 142)
    }

    func test_decodes_contextTrigger_fromString() throws {
        let m = try decodeSample()
        XCTAssertNotNil(m.contextTrigger)
        XCTAssertEqual(m.contextTrigger?.expression, "calendar:gap>30m")
    }

    func test_decodes_stateHistory() throws {
        let m = try decodeSample()
        XCTAssertEqual(m.stateHistory.count, 1)
        let entry = m.stateHistory[0]
        XCTAssertNil(entry.fromState)
        XCTAssertEqual(entry.toState, .suggested)
        XCTAssertEqual(entry.ts.timeIntervalSince1970, 1_777_204_800, accuracy: 0.0001)
        XCTAssertEqual(entry.annotation, "created by cadence producer")
    }
}

// MARK: - Minimal Moment decode (defaults)

final class MomentMinimalDecodeTests: XCTestCase {

    private func decodeMinimal() throws -> Moment {
        let data = minimalMomentJSON.data(using: .utf8)!
        return try Moment.decoder().decode(Moment.self, from: data)
    }

    func test_decodes_minimalRequiredFields() throws {
        let m = try decodeMinimal()
        XCTAssertEqual(m.id, "01HXYZMINIMAL00000000000000")
        XCTAssertEqual(m.proposedAction.kind, .nudge)
        XCTAssertEqual(m.sourceInsightType, .routine)
    }

    func test_appliesDefaults_whenOptionalsOmitted() throws {
        let m = try decodeMinimal()
        XCTAssertEqual(m.evidence, [])
        XCTAssertEqual(m.state, .suggested)
        XCTAssertNil(m.scheduledFor)
        XCTAssertNil(m.contextTrigger)
        XCTAssertNil(m.snoozeUntil)
        XCTAssertEqual(m.confidence, 0.0)
        XCTAssertEqual(m.feedbackWeight, 1.0)
        XCTAssertEqual(m.stateHistory, [])
        XCTAssertEqual(m.proposedAction.params, [:])
    }
}

// MARK: - Round-trip encode/decode

final class MomentRoundTripTests: XCTestCase {

    func test_roundTrip_preservesAllScalars() throws {
        let original = try Moment.decoder().decode(Moment.self, from: sampleMomentJSON.data(using: .utf8)!)
        let encoded = try Moment.encoder().encode(original)
        let decoded = try Moment.decoder().decode(Moment.self, from: encoded)

        XCTAssertEqual(decoded.id, original.id)
        XCTAssertEqual(decoded.insight, original.insight)
        XCTAssertEqual(decoded.evidenceHash, original.evidenceHash)
        XCTAssertEqual(decoded.state, original.state)
        XCTAssertEqual(decoded.sourceInsightType, original.sourceInsightType)
        XCTAssertEqual(decoded.proposedAction.kind, original.proposedAction.kind)
        XCTAssertEqual(decoded.contextTrigger?.expression, original.contextTrigger?.expression)
        XCTAssertEqual(decoded.confidence, original.confidence, accuracy: 0.0001)
        XCTAssertEqual(decoded.feedbackWeight, original.feedbackWeight, accuracy: 0.0001)
        XCTAssertEqual(decoded.evidence, original.evidence)
        XCTAssertEqual(decoded.stateHistory.count, original.stateHistory.count)
    }

    func test_contextTrigger_encodesAsString() throws {
        let trigger = ContextTrigger(expression: "arrive:home")
        let data = try JSONEncoder().encode(trigger)
        let s = String(data: data, encoding: .utf8)
        XCTAssertEqual(s, "\"arrive:home\"")
    }
}

// MARK: - MomentFeed decode

final class MomentFeedDecodeTests: XCTestCase {

    private func decodeFeed() throws -> MomentFeed {
        let data = feedJSON.data(using: .utf8)!
        return try Moment.decoder().decode(MomentFeed.self, from: data)
    }

    func test_decodes_threeBuckets() throws {
        let feed = try decodeFeed()
        XCTAssertEqual(feed.pending.count, 1)
        XCTAssertEqual(feed.scheduled.count, 1)
        XCTAssertEqual(feed.done.count, 1)
    }

    func test_pendingBucket_hasSuggestedRelationshipMoment() throws {
        let feed = try decodeFeed()
        let m = feed.pending[0]
        XCTAssertEqual(m.state, .suggested)
        XCTAssertEqual(m.sourceInsightType, .relationship)
        XCTAssertEqual(m.proposedAction.kind, .draftMessage)
    }

    func test_scheduledBucket_hasContextTrigger() throws {
        let feed = try decodeFeed()
        let m = feed.scheduled[0]
        XCTAssertEqual(m.contextTrigger?.expression, "time:09:30")
        XCTAssertEqual(m.scheduledFor?.timeIntervalSince1970, 1_777_230_000)
    }

    func test_doneBucket_hasFullStateTrail() throws {
        let feed = try decodeFeed()
        let m = feed.done[0]
        XCTAssertEqual(m.state, .done)
        XCTAssertEqual(m.stateHistory.count, 3)
        XCTAssertEqual(m.stateHistory[0].toState, .suggested)
        XCTAssertEqual(m.stateHistory[1].fromState, .suggested)
        XCTAssertEqual(m.stateHistory[1].toState, .accepted)
        XCTAssertEqual(m.stateHistory[2].toState, .done)
    }
}

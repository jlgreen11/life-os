//
//  Moment.swift
//  Life OS — Swift Moment primitive
//
//  Codable mirror of the Python `core/moment/types.py` Moment dataclass and
//  its three string-valued enums (MomentState, InsightType, ActionKind).
//
//  Wire-format reference: api/schemas.py → MomentOut. Notable details that
//  surprised first-time readers and are worth preserving here:
//
//  • Timestamps (`created_at`, `expires_at`, `scheduled_for`, `snooze_until`,
//    state-history `ts`) are Unix epoch SECONDS as JSON integers — not ISO
//    strings. Decode with JSONDecoder.dateDecodingStrategy = .secondsSince1970,
//    or use the convenience `Moment.decoder` factory below.
//
//  • `context_trigger` arrives as a single JSON string (the expression),
//    not an object. ContextTrigger therefore has a custom Decoder that
//    unwraps the string into the struct — see `init(from:)` below.
//
//  • `params` on a proposed Action is open-shape per Python (per-`kind`
//    schema lives in the outbox dispatcher, not here). Modeled as
//    `[String: AnyCodable]` so any JSON-compatible value round-trips.
//
//  Enum cases use Swift camelCase with explicit String raw values so the
//  wire format keeps the snake_case Python contract while call sites read
//  idiomatically (`.draftMessage`, `.commTemplate`).
//

import Foundation

// MARK: - Enums

/// Lifecycle state for a Moment. Mirrors `MomentState` in Python.
/// Terminal states: `.dismissed`, `.done`, `.expired`.
enum MomentState: String, Codable, CaseIterable {
    case suggested
    case accepted
    case dismissed
    case snoozed
    case done
    case expired
}

/// Producer identity. Mirrors `InsightType` in Python (six Phase 1 producers).
/// The DB schema also accepts `"legacy_task"` for migrated v1 rows; we deliberately
/// do not enumerate it here — the producer layer never emits it.
enum InsightType: String, Codable, CaseIterable {
    case cadence
    case relationship
    case temporal
    case spatial
    case commTemplate = "comm_template"
    case routine
}

/// What a Moment proposes to do when accepted. Mirrors `ActionKind` in Python.
enum ActionKind: String, Codable, CaseIterable {
    case draftMessage = "draft_message"
    case sendMessage = "send_message"
    case scheduleBlock = "schedule_block"
    case archiveEvent = "archive_event"
    case nudge
    case setReminder = "set_reminder"
    case createCalendarEntry = "create_calendar_entry"
    case noteObservation = "note_observation"
}

// MARK: - AnyCodable

/// Minimal Codable wrapper for arbitrary JSON values used inside `Action.params`.
///
/// Swift's `Decodable` cannot directly decode `Any`, so we walk the standard
/// JSON value set: null, Bool, Int, Double, String, array, dictionary. This is
/// enough for every action-param shape the Python side emits today; the outbox
/// dispatcher is the source of truth for per-`kind` schemas.
struct AnyCodable: Codable, Equatable {
    let value: Any?

    init(_ value: Any?) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self.value = nil
        } else if let b = try? container.decode(Bool.self) {
            self.value = b
        } else if let i = try? container.decode(Int.self) {
            self.value = i
        } else if let d = try? container.decode(Double.self) {
            self.value = d
        } else if let s = try? container.decode(String.self) {
            self.value = s
        } else if let arr = try? container.decode([AnyCodable].self) {
            self.value = arr.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            self.value = dict.mapValues { $0.value }
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "AnyCodable: unsupported JSON value"
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case nil:
            try container.encodeNil()
        case let b as Bool:
            try container.encode(b)
        case let i as Int:
            try container.encode(i)
        case let d as Double:
            try container.encode(d)
        case let s as String:
            try container.encode(s)
        case let arr as [Any?]:
            try container.encode(arr.map { AnyCodable($0) })
        case let dict as [String: Any?]:
            try container.encode(dict.mapValues { AnyCodable($0) })
        default:
            throw EncodingError.invalidValue(
                value as Any,
                EncodingError.Context(
                    codingPath: encoder.codingPath,
                    debugDescription: "AnyCodable: cannot encode value of type \(type(of: value))"
                )
            )
        }
    }

    static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        switch (lhs.value, rhs.value) {
        case (nil, nil): return true
        case let (l as Bool, r as Bool): return l == r
        case let (l as Int, r as Int): return l == r
        case let (l as Double, r as Double): return l == r
        case let (l as String, r as String): return l == r
        case let (l as [Any?], r as [Any?]):
            return l.map { AnyCodable($0) } == r.map { AnyCodable($0) }
        case let (l as [String: Any?], r as [String: Any?]):
            return l.mapValues { AnyCodable($0) } == r.mapValues { AnyCodable($0) }
        default:
            return false
        }
    }
}

// MARK: - Action / ContextTrigger / StateHistoryEntry

struct Action: Codable, Equatable {
    let kind: ActionKind
    let params: [String: AnyCodable]

    init(kind: ActionKind, params: [String: AnyCodable] = [:]) {
        self.kind = kind
        self.params = params
    }

    enum CodingKeys: String, CodingKey {
        case kind
        case params
    }

    // `params` is a Pydantic default-empty dict on the wire, but a tolerant
    // client decoder accepts it being absent OR JSON null — matches the
    // shape of every other optional field on Moment.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.kind = try c.decode(ActionKind.self, forKey: .kind)
        self.params = try c.decodeIfPresent([String: AnyCodable].self, forKey: .params) ?? [:]
    }
}

/// Context-trigger expression (CEO plan grammar). The wire format is a single
/// JSON string (e.g. `"calendar:gap>30m"`); we unwrap it into a struct so call
/// sites can later parse / pattern-match without re-modeling the field.
struct ContextTrigger: Codable, Equatable {
    let expression: String

    init(expression: String) {
        self.expression = expression
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        self.expression = try container.decode(String.self)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(expression)
    }
}

struct StateHistoryEntry: Codable, Equatable {
    let fromState: MomentState?
    let toState: MomentState
    let ts: Date
    let annotation: String?

    enum CodingKeys: String, CodingKey {
        case fromState = "from_state"
        case toState = "to_state"
        case ts
        case annotation
    }
}

// MARK: - Moment

/// One evidence-backed, state-machine-governed user-facing unit.
///
/// JSON wire format mirrors `api/schemas.py::MomentOut`. Use
/// `Moment.decoder()` to get a JSONDecoder pre-configured with the
/// `.secondsSince1970` date strategy.
struct Moment: Codable, Identifiable, Equatable {
    let id: String
    let createdAt: Date
    let expiresAt: Date
    let insight: String
    let evidence: [String]
    let evidenceHash: String
    let proposedAction: Action
    let sourceInsightType: InsightType
    let state: MomentState
    let scheduledFor: Date?
    let contextTrigger: ContextTrigger?
    let snoozeUntil: Date?
    let confidence: Double
    let feedbackWeight: Double
    let stateHistory: [StateHistoryEntry]

    enum CodingKeys: String, CodingKey {
        case id
        case createdAt = "created_at"
        case expiresAt = "expires_at"
        case insight
        case evidence
        case evidenceHash = "evidence_hash"
        case proposedAction = "proposed_action"
        case sourceInsightType = "source_insight_type"
        case state
        case scheduledFor = "scheduled_for"
        case contextTrigger = "context_trigger"
        case snoozeUntil = "snooze_until"
        case confidence
        case feedbackWeight = "feedback_weight"
        case stateHistory = "state_history"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.createdAt = try c.decode(Date.self, forKey: .createdAt)
        self.expiresAt = try c.decode(Date.self, forKey: .expiresAt)
        self.insight = try c.decode(String.self, forKey: .insight)
        self.evidence = try c.decodeIfPresent([String].self, forKey: .evidence) ?? []
        self.evidenceHash = try c.decode(String.self, forKey: .evidenceHash)
        self.proposedAction = try c.decode(Action.self, forKey: .proposedAction)
        self.sourceInsightType = try c.decode(InsightType.self, forKey: .sourceInsightType)
        self.state = try c.decodeIfPresent(MomentState.self, forKey: .state) ?? .suggested
        self.scheduledFor = try c.decodeIfPresent(Date.self, forKey: .scheduledFor)
        self.contextTrigger = try c.decodeIfPresent(ContextTrigger.self, forKey: .contextTrigger)
        self.snoozeUntil = try c.decodeIfPresent(Date.self, forKey: .snoozeUntil)
        self.confidence = try c.decodeIfPresent(Double.self, forKey: .confidence) ?? 0.0
        self.feedbackWeight = try c.decodeIfPresent(Double.self, forKey: .feedbackWeight) ?? 1.0
        self.stateHistory = try c.decodeIfPresent([StateHistoryEntry].self, forKey: .stateHistory) ?? []
    }

    init(
        id: String,
        createdAt: Date,
        expiresAt: Date,
        insight: String,
        evidence: [String],
        evidenceHash: String,
        proposedAction: Action,
        sourceInsightType: InsightType,
        state: MomentState = .suggested,
        scheduledFor: Date? = nil,
        contextTrigger: ContextTrigger? = nil,
        snoozeUntil: Date? = nil,
        confidence: Double = 0.0,
        feedbackWeight: Double = 1.0,
        stateHistory: [StateHistoryEntry] = []
    ) {
        self.id = id
        self.createdAt = createdAt
        self.expiresAt = expiresAt
        self.insight = insight
        self.evidence = evidence
        self.evidenceHash = evidenceHash
        self.proposedAction = proposedAction
        self.sourceInsightType = sourceInsightType
        self.state = state
        self.scheduledFor = scheduledFor
        self.contextTrigger = contextTrigger
        self.snoozeUntil = snoozeUntil
        self.confidence = confidence
        self.feedbackWeight = feedbackWeight
        self.stateHistory = stateHistory
    }

    /// JSONDecoder pre-configured for the v2 Moment wire format.
    static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .secondsSince1970
        return d
    }

    /// JSONEncoder pre-configured for the v2 Moment wire format.
    static func encoder() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .secondsSince1970
        return e
    }
}

// MARK: - Feed

/// `GET /api/now` payload. Mirrors `api/schemas.py::MomentListOut`.
struct MomentFeed: Codable, Equatable {
    let pending: [Moment]
    let scheduled: [Moment]
    let done: [Moment]
}

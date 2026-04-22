//
//  APITypes.swift
//  Life OS — Swift mirrors of the v2 API surface
//
//  Codable counterparts to every response/request schema in
//  `api/schemas.py` plus the two legacy iOS-compat shapes
//  (briefing proxy, smoke status) served out of `api/routes/context.py`.
//  The v2 Moment/MomentFeed types live in `Moment.swift`; everything
//  else is here.
//
//  Wire contract: Unix epoch SECONDS as JSON integers (matches
//  `api/schemas.py` — timestamps never move to ISO on the API surface).
//  Every field maps to snake_case on the wire via explicit CodingKeys
//  so call sites stay idiomatic Swift.
//

import Foundation

// MARK: - Health + smoke status

/// Deep-health payload returned by `GET /api/health`. Mirrors
/// `api/schemas.py::HealthOut`. Multi-key by design — the flat ``ok``
/// summary is the AND of the component checks; callers surface
/// individual failures via the sub-dictionaries.
struct HealthStatus: Codable, Equatable {
    let ok: Bool
    let ts: Int
    let connectors: [String: String]
    let dbLastWriteTs: Int?
    let schedulerHeartbeatTs: Int?
    let producerActivity: [String: Int]
    let pendingMoments: Int
    let notes: [String]

    enum CodingKeys: String, CodingKey {
        case ok
        case ts
        case connectors
        case dbLastWriteTs = "db_last_write_ts"
        case schedulerHeartbeatTs = "scheduler_heartbeat_ts"
        case producerActivity = "producer_activity"
        case pendingMoments = "pending_moments"
        case notes
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.ok = try c.decode(Bool.self, forKey: .ok)
        self.ts = try c.decode(Int.self, forKey: .ts)
        self.connectors = try c.decodeIfPresent([String: String].self, forKey: .connectors) ?? [:]
        self.dbLastWriteTs = try c.decodeIfPresent(Int.self, forKey: .dbLastWriteTs)
        self.schedulerHeartbeatTs = try c.decodeIfPresent(Int.self, forKey: .schedulerHeartbeatTs)
        self.producerActivity = try c.decodeIfPresent([String: Int].self, forKey: .producerActivity) ?? [:]
        self.pendingMoments = try c.decodeIfPresent(Int.self, forKey: .pendingMoments) ?? 0
        self.notes = try c.decodeIfPresent([String].self, forKey: .notes) ?? []
    }

    init(
        ok: Bool,
        ts: Int,
        connectors: [String: String] = [:],
        dbLastWriteTs: Int? = nil,
        schedulerHeartbeatTs: Int? = nil,
        producerActivity: [String: Int] = [:],
        pendingMoments: Int = 0,
        notes: [String] = []
    ) {
        self.ok = ok
        self.ts = ts
        self.connectors = connectors
        self.dbLastWriteTs = dbLastWriteTs
        self.schedulerHeartbeatTs = schedulerHeartbeatTs
        self.producerActivity = producerActivity
        self.pendingMoments = pendingMoments
        self.notes = notes
    }
}

/// Smoke-check payload returned by `GET /api/status` (iOS-compat shim).
/// Flat by design — the iOS app's launch ping must round-trip in a
/// single GET without dereferencing sub-structures.
struct StatusSmoke: Codable, Equatable {
    let ok: Bool
    let ts: Int
    let eventCount: Int
    let momentCount: Int

    enum CodingKeys: String, CodingKey {
        case ok
        case ts
        case eventCount = "event_count"
        case momentCount = "moment_count"
    }
}

// MARK: - Briefing (legacy iOS proxy)

/// Daily-briefing payload served by the iOS-compat shim at
/// `GET /api/briefing`. The `error` key carries a diagnostic when the
/// LLM is down; `briefing` is nullable so the iOS dashboard can render
/// the error inline instead of going offline on 5xx.
struct BriefingResponse: Codable, Equatable {
    let briefing: String?
    let generatedAt: String
    let error: String?

    enum CodingKeys: String, CodingKey {
        case briefing
        case generatedAt = "generated_at"
        case error
    }
}

// MARK: - You tab

/// Per-audience writing summary row for the You tab §"HOW YOU WRITE".
/// Mirrors `api/schemas.py::PersonaStyleOut`.
struct PersonaStyle: Codable, Equatable {
    let audience: String
    let tone: String
    let formality: Double
    let sampleSize: Int

    enum CodingKeys: String, CodingKey {
        case audience
        case tone
        case formality
        case sampleSize = "sample_size"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.audience = try c.decode(String.self, forKey: .audience)
        self.tone = try c.decode(String.self, forKey: .tone)
        self.formality = try c.decodeIfPresent(Double.self, forKey: .formality) ?? 0.0
        self.sampleSize = try c.decodeIfPresent(Int.self, forKey: .sampleSize) ?? 0
    }

    init(audience: String, tone: String, formality: Double = 0.0, sampleSize: Int = 0) {
        self.audience = audience
        self.tone = tone
        self.formality = formality
        self.sampleSize = sampleSize
    }
}

/// Detected routine row for the You tab §"YOUR ROUTINES".
/// Mirrors `api/schemas.py::RoutineOut`. Empty-state is carried by an
/// explicit ``detected=false`` row so the UI renders "No routine
/// detected yet" without null-checking.
struct Routine: Codable, Equatable {
    let name: String
    let detected: Bool
    let description: String?
    let confidence: Double
    let sampleSize: Int

    enum CodingKeys: String, CodingKey {
        case name
        case detected
        case description
        case confidence
        case sampleSize = "sample_size"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try c.decode(String.self, forKey: .name)
        self.detected = try c.decodeIfPresent(Bool.self, forKey: .detected) ?? true
        self.description = try c.decodeIfPresent(String.self, forKey: .description)
        self.confidence = try c.decodeIfPresent(Double.self, forKey: .confidence) ?? 0.0
        self.sampleSize = try c.decodeIfPresent(Int.self, forKey: .sampleSize) ?? 0
    }

    init(name: String, detected: Bool = true, description: String? = nil, confidence: Double = 0.0, sampleSize: Int = 0) {
        self.name = name
        self.detected = detected
        self.description = description
        self.confidence = confidence
        self.sampleSize = sampleSize
    }
}

/// Drifted-contact row shared by the You tab §"DRIFTING" and the People
/// tab §"NEEDS ATTENTION". Mirrors `api/schemas.py::DriftingContactOut`.
struct DriftingContact: Codable, Equatable, Identifiable {
    let contactId: String
    let name: String
    let daysSinceLast: Int
    let usualCadenceDays: Int

    var id: String { contactId }

    enum CodingKeys: String, CodingKey {
        case contactId = "contact_id"
        case name
        case daysSinceLast = "days_since_last"
        case usualCadenceDays = "usual_cadence_days"
    }
}

/// You-tab payload returned by `GET /api/you`. Mirrors
/// `api/schemas.py::YouOut`. Every list section defaults to `[]` so
/// fresh installs round-trip the same shape as populated ones.
struct SelfPortrait: Codable, Equatable {
    let observedMonths: Int
    let interactionsCount: Int
    let confidencePct: Int
    let whenAtBest: [String]
    let howYouWrite: [PersonaStyle]
    let yourRoutines: [Routine]
    let drifting: [DriftingContact]

    enum CodingKeys: String, CodingKey {
        case observedMonths = "observed_months"
        case interactionsCount = "interactions_count"
        case confidencePct = "confidence_pct"
        case whenAtBest = "when_at_best"
        case howYouWrite = "how_you_write"
        case yourRoutines = "your_routines"
        case drifting
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.observedMonths = try c.decodeIfPresent(Int.self, forKey: .observedMonths) ?? 0
        self.interactionsCount = try c.decodeIfPresent(Int.self, forKey: .interactionsCount) ?? 0
        self.confidencePct = try c.decodeIfPresent(Int.self, forKey: .confidencePct) ?? 0
        self.whenAtBest = try c.decodeIfPresent([String].self, forKey: .whenAtBest) ?? []
        self.howYouWrite = try c.decodeIfPresent([PersonaStyle].self, forKey: .howYouWrite) ?? []
        self.yourRoutines = try c.decodeIfPresent([Routine].self, forKey: .yourRoutines) ?? []
        self.drifting = try c.decodeIfPresent([DriftingContact].self, forKey: .drifting) ?? []
    }

    init(
        observedMonths: Int = 0,
        interactionsCount: Int = 0,
        confidencePct: Int = 0,
        whenAtBest: [String] = [],
        howYouWrite: [PersonaStyle] = [],
        yourRoutines: [Routine] = [],
        drifting: [DriftingContact] = []
    ) {
        self.observedMonths = observedMonths
        self.interactionsCount = interactionsCount
        self.confidencePct = confidencePct
        self.whenAtBest = whenAtBest
        self.howYouWrite = howYouWrite
        self.yourRoutines = yourRoutines
        self.drifting = drifting
    }
}

// MARK: - People tab

/// Contact-row summary for the People tab list. Mirrors
/// `api/schemas.py::ContactSummaryOut`. The monospace right-aligned
/// stats in the UI correspond to ``lastContactTs`` +
/// ``cadenceDeviationDays``.
struct ContactSummary: Codable, Equatable, Identifiable {
    let contactId: String
    let name: String
    let lastContactTs: Int?
    let cadenceDeviationDays: Int?
    let needsAttention: Bool

    var id: String { contactId }

    enum CodingKeys: String, CodingKey {
        case contactId = "contact_id"
        case name
        case lastContactTs = "last_contact_ts"
        case cadenceDeviationDays = "cadence_deviation_days"
        case needsAttention = "needs_attention"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.contactId = try c.decode(String.self, forKey: .contactId)
        self.name = try c.decode(String.self, forKey: .name)
        self.lastContactTs = try c.decodeIfPresent(Int.self, forKey: .lastContactTs)
        self.cadenceDeviationDays = try c.decodeIfPresent(Int.self, forKey: .cadenceDeviationDays)
        self.needsAttention = try c.decodeIfPresent(Bool.self, forKey: .needsAttention) ?? false
    }

    init(
        contactId: String,
        name: String,
        lastContactTs: Int? = nil,
        cadenceDeviationDays: Int? = nil,
        needsAttention: Bool = false
    ) {
        self.contactId = contactId
        self.name = name
        self.lastContactTs = lastContactTs
        self.cadenceDeviationDays = cadenceDeviationDays
        self.needsAttention = needsAttention
    }
}

/// People-tab payload returned by `GET /api/people`. Mirrors
/// `api/schemas.py::PeopleListOut`. ``you`` is pinned to the top of the
/// list (DESIGN.md § "Always starts with YOU"); the two sub-lists are
/// rendered below.
struct PeopleList: Codable, Equatable {
    let you: SelfPortrait
    let needsAttention: [ContactSummary]
    let activeThisWeek: [ContactSummary]
    let total: Int
    let query: String?

    enum CodingKeys: String, CodingKey {
        case you
        case needsAttention = "needs_attention"
        case activeThisWeek = "active_this_week"
        case total
        case query
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.you = try c.decode(SelfPortrait.self, forKey: .you)
        self.needsAttention = try c.decodeIfPresent([ContactSummary].self, forKey: .needsAttention) ?? []
        self.activeThisWeek = try c.decodeIfPresent([ContactSummary].self, forKey: .activeThisWeek) ?? []
        self.total = try c.decodeIfPresent(Int.self, forKey: .total) ?? 0
        self.query = try c.decodeIfPresent(String.self, forKey: .query)
    }

    init(
        you: SelfPortrait,
        needsAttention: [ContactSummary] = [],
        activeThisWeek: [ContactSummary] = [],
        total: Int = 0,
        query: String? = nil
    ) {
        self.you = you
        self.needsAttention = needsAttention
        self.activeThisWeek = activeThisWeek
        self.total = total
        self.query = query
    }
}

/// Per-contact dossier returned by `GET /api/people/{id}`. Mirrors
/// `api/schemas.py::ContactDossierOut`. ``cadenceSparkline`` is a
/// per-day contact-count list for the UI sparkline.
struct ContactDossier: Codable, Equatable, Identifiable {
    let contactId: String
    let name: String
    let lastContactTs: Int?
    let usualCadenceDays: Int?
    let commTemplate: String?
    let cadenceSparkline: [Int]
    let recentTopics: [String]
    let predictedNext: String?

    var id: String { contactId }

    enum CodingKeys: String, CodingKey {
        case contactId = "contact_id"
        case name
        case lastContactTs = "last_contact_ts"
        case usualCadenceDays = "usual_cadence_days"
        case commTemplate = "comm_template"
        case cadenceSparkline = "cadence_sparkline"
        case recentTopics = "recent_topics"
        case predictedNext = "predicted_next"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.contactId = try c.decode(String.self, forKey: .contactId)
        self.name = try c.decode(String.self, forKey: .name)
        self.lastContactTs = try c.decodeIfPresent(Int.self, forKey: .lastContactTs)
        self.usualCadenceDays = try c.decodeIfPresent(Int.self, forKey: .usualCadenceDays)
        self.commTemplate = try c.decodeIfPresent(String.self, forKey: .commTemplate)
        self.cadenceSparkline = try c.decodeIfPresent([Int].self, forKey: .cadenceSparkline) ?? []
        self.recentTopics = try c.decodeIfPresent([String].self, forKey: .recentTopics) ?? []
        self.predictedNext = try c.decodeIfPresent(String.self, forKey: .predictedNext)
    }
}

// MARK: - Settings tab

/// Connector status row returned by `GET /api/connectors`. Mirrors
/// `api/schemas.py::ConnectorOut`. Fernet-encrypted credentials are
/// NEVER returned on this shape — only status-level fields.
struct Connector: Codable, Equatable, Identifiable {
    let id: String
    let kind: String
    let enabled: Bool
    let status: String
    let lastSyncAt: Int?
    let lastError: String?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case enabled
        case status
        case lastSyncAt = "last_sync_at"
        case lastError = "last_error"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.kind = try c.decode(String.self, forKey: .kind)
        self.enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
        self.status = try c.decodeIfPresent(String.self, forKey: .status) ?? "unknown"
        self.lastSyncAt = try c.decodeIfPresent(Int.self, forKey: .lastSyncAt)
        self.lastError = try c.decodeIfPresent(String.self, forKey: .lastError)
    }
}

/// PATCH-body for `PATCH /api/connectors/{id}`. Mirrors
/// `api/schemas.py::ConnectorConfigIn`. All three fields are optional
/// so a PATCH can flip ``enabled`` without re-sending creds.
struct ConnectorConfigUpdate: Codable, Equatable {
    let enabled: Bool?
    let config: [String: AnyCodable]?
    let secrets: [String: AnyCodable]?

    init(
        enabled: Bool? = nil,
        config: [String: AnyCodable]? = nil,
        secrets: [String: AnyCodable]? = nil
    ) {
        self.enabled = enabled
        self.config = config
        self.secrets = secrets
    }
}

// MARK: - Moment action body

/// POST-body shared by `/api/moments/{id}/accept|dismiss|snooze|edit`.
/// Mirrors `api/schemas.py::MomentActionIn`. Every field is optional;
/// per-endpoint validation (snooze requires ``snoozeUntil``, edit
/// requires ``actionParams``) lives server-side and surfaces as 422.
struct MomentActionBody: Codable, Equatable {
    let snoozeUntil: Int?
    let actionParams: [String: AnyCodable]?
    let annotation: String?

    enum CodingKeys: String, CodingKey {
        case snoozeUntil = "snooze_until"
        case actionParams = "action_params"
        case annotation
    }

    init(
        snoozeUntil: Int? = nil,
        actionParams: [String: AnyCodable]? = nil,
        annotation: String? = nil
    ) {
        self.snoozeUntil = snoozeUntil
        self.actionParams = actionParams
        self.annotation = annotation
    }
}

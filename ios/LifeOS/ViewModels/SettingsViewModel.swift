//
//  SettingsViewModel.swift
//  Life OS — Settings tab view model
//
//  Owns the connector roster + preferences row rendered by
//  `SettingsTabView`. Two writers:
//
//    • `updateConnector(id:update:)` — `PATCH /api/connectors/{id}`
//      (toggles enabled, swaps config, swaps secrets). On success the
//      returned row is upserted in-place inside `connectors` so the
//      list re-renders without a full reload.
//    • `updatePreference(key:value:)` — single-row upsert into the
//      `preferences` table. Pure write — there's no per-key GET
//      response to fold back, so the local `preferences` slot is
//      patched optimistically here and the next `loadConnectors()`
//      run picks up authoritative state.
//

import Foundation
import SwiftUI

@MainActor
final class SettingsViewModel: ObservableObject {

    // MARK: - Published state

    /// Connector roster driving the CONNECTORS section. Defaults empty
    /// so the empty-state copy renders before the first load returns.
    @Published var connectors: [Connector] = []

    /// Preferences row backing the PREFERENCES section. Defaults match
    /// `_PREFERENCE_DEFAULTS` in `api/routes/settings.py`.
    @Published var preferences: Preferences = .defaults

    @Published var isLoading: Bool = false

    @Published var errorMessage: String? = nil

    // MARK: - Dependencies

    private let client: APIClientProtocol

    init(client: APIClientProtocol) {
        self.client = client
    }

    // MARK: - Loaders

    /// Fetch `/api/connectors` and replace `connectors`. On failure
    /// leaves the existing list untouched and surfaces a string in
    /// `errorMessage`.
    func loadConnectors() async {
        isLoading = true
        errorMessage = nil
        do {
            connectors = try await client.getConnectors()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    // MARK: - Writers

    /// `PATCH /api/connectors/{id}` — upsert the returned row in
    /// `connectors` in place if `id` already exists, else append.
    func updateConnector(id: String, update: ConnectorConfigUpdate) async {
        errorMessage = nil
        do {
            let updated = try await client.updateConnector(id: id, update: update)
            if let idx = connectors.firstIndex(where: { $0.id == id }) {
                connectors[idx] = updated
            } else {
                connectors.append(updated)
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// `POST /api/preferences` — upsert a single `(key, value)` row.
    /// On success the local `preferences` slot is patched too so the
    /// view re-renders without a refetch; on failure nothing is mutated
    /// and the error surfaces in `errorMessage`.
    func updatePreference(key: String, value: AnyCodable) async {
        errorMessage = nil
        do {
            try await client.updatePreference(key: key, value: value)
            preferences = Self.applying(key: key, value: value, to: preferences)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Pure helpers

    /// Apply a single `(key, value)` upsert to a `Preferences` row,
    /// returning a new copy. Static so tests can lock the truth table
    /// without instantiating the view model. Unknown keys are no-ops
    /// (the backend stores them, but the four-key UI struct can't
    /// represent them).
    static func applying(key: String, value: AnyCodable, to base: Preferences) -> Preferences {
        switch key {
        case "quiet_hours_start":
            guard let s = value.value as? String else { return base }
            return Preferences(
                quietHoursStart: s,
                quietHoursEnd: base.quietHoursEnd,
                autonomyLevel: base.autonomyLevel,
                proactivity: base.proactivity
            )
        case "quiet_hours_end":
            guard let s = value.value as? String else { return base }
            return Preferences(
                quietHoursStart: base.quietHoursStart,
                quietHoursEnd: s,
                autonomyLevel: base.autonomyLevel,
                proactivity: base.proactivity
            )
        case "autonomy_level":
            guard let d = Self.coerceDouble(value) else { return base }
            return Preferences(
                quietHoursStart: base.quietHoursStart,
                quietHoursEnd: base.quietHoursEnd,
                autonomyLevel: d,
                proactivity: base.proactivity
            )
        case "proactivity":
            guard let d = Self.coerceDouble(value) else { return base }
            return Preferences(
                quietHoursStart: base.quietHoursStart,
                quietHoursEnd: base.quietHoursEnd,
                autonomyLevel: base.autonomyLevel,
                proactivity: d
            )
        default:
            return base
        }
    }

    /// Coerce an `AnyCodable` payload to `Double` — accepts both `Int`
    /// and `Double` because JSON numerics round-trip through whichever
    /// concrete type matches first inside `AnyCodable.init(from:)`.
    private static func coerceDouble(_ value: AnyCodable) -> Double? {
        if let d = value.value as? Double { return d }
        if let i = value.value as? Int { return Double(i) }
        return nil
    }
}

//
//  NowViewModel.swift
//  Life OS — Now tab view model
//
//  Owns the `MomentFeed` rendered by `NowTabView` plus the four action
//  dispatchers (accept / dismiss / snooze / undo) wired to the v2
//  API. The view model itself is `@MainActor` so `@Published` writes
//  drive SwiftUI updates without `Task { @MainActor in … }` ceremony
//  at every call site.
//
//  Reconciliation rules (locked by `reconcile(feed:updated:)` so tests
//  can assert without spinning the view model):
//
//    .suggested / .accepted → moment lives in `feed.pending`
//    .snoozed               → moment lives in `feed.scheduled`
//    .done                  → moment lives in `feed.done`
//    .dismissed / .expired  → moment dropped from every bucket
//
//  Every dispatcher updates `errorMessage` instead of throwing — the
//  view binds to the published string and renders inline.
//

import Foundation
import SwiftUI

@MainActor
final class NowViewModel: ObservableObject {

    // MARK: - Published state

    /// Current feed driving the Now tab. Defaults to an empty feed so a
    /// view bound before `load()` returns renders the empty state, not
    /// stale fixture data.
    @Published var feed: MomentFeed = MomentFeed(pending: [], scheduled: [], done: [])

    /// True while a `load()` round-trip is in flight. Toggles back to
    /// false on both success and failure.
    @Published var isLoading: Bool = false

    /// Last error surfaced by any API call. Cleared at the start of
    /// every new call. `nil` means "no error to render".
    @Published var errorMessage: String? = nil

    // MARK: - Dependencies

    private let client: APIClientProtocol

    init(client: APIClientProtocol) {
        self.client = client
    }

    // MARK: - Loaders

    /// Fetch `/api/now` and replace `feed`. On failure leaves `feed`
    /// untouched and writes a string into `errorMessage`.
    func load() async {
        isLoading = true
        errorMessage = nil
        do {
            feed = try await client.getNow()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    // MARK: - Action dispatchers

    /// `POST /api/moments/{id}/accept` then re-bucket the returned
    /// Moment based on its post-transition state.
    func accept(momentId: String, annotation: String? = nil) async {
        await dispatchAction {
            try await self.client.acceptMoment(id: momentId, annotation: annotation)
        }
    }

    /// `POST /api/moments/{id}/dismiss` — terminal transition. The
    /// returned Moment carries `state = .dismissed`, which causes
    /// `reconcile` to drop it from every visible bucket.
    func dismiss(momentId: String, annotation: String? = nil) async {
        await dispatchAction {
            try await self.client.dismissMoment(id: momentId, annotation: annotation)
        }
    }

    /// `POST /api/moments/{id}/snooze` — the returned Moment lands in
    /// `feed.scheduled` (UP NEXT bucket) when the server resolves the
    /// transition to `.snoozed`. If the server coerces past-expiry to
    /// `.expired`, `reconcile` drops the moment instead.
    func snooze(momentId: String, snoozeUntil: Int, annotation: String? = nil) async {
        await dispatchAction {
            try await self.client.snoozeMoment(
                id: momentId,
                snoozeUntil: snoozeUntil,
                annotation: annotation
            )
        }
    }

    /// `POST /api/moments/{id}/undo` — reverses the last terminal
    /// transition. The returned Moment is re-applied via `reconcile`,
    /// which puts it back into the bucket implied by its restored state.
    func undo(momentId: String) async {
        await dispatchAction {
            try await self.client.undoMoment(id: momentId)
        }
    }

    // MARK: - Reconciliation (pure helper)

    /// Re-bucket `updated` inside `feed`. Static so tests can lock the
    /// truth table (state → bucket) without instantiating the view
    /// model. The function strips `updated.id` from every bucket first
    /// so a moment can move freely between buckets in one step.
    static func reconcile(feed: MomentFeed, updated: Moment) -> MomentFeed {
        let pending = feed.pending.filter { $0.id != updated.id }
        let scheduled = feed.scheduled.filter { $0.id != updated.id }
        let done = feed.done.filter { $0.id != updated.id }

        switch updated.state {
        case .suggested, .accepted:
            return MomentFeed(pending: pending + [updated], scheduled: scheduled, done: done)
        case .snoozed:
            return MomentFeed(pending: pending, scheduled: scheduled + [updated], done: done)
        case .done:
            return MomentFeed(pending: pending, scheduled: scheduled, done: done + [updated])
        case .dismissed, .expired:
            return MomentFeed(pending: pending, scheduled: scheduled, done: done)
        }
    }

    // MARK: - Private

    /// Shared boilerplate for the four action dispatchers — clear the
    /// last error, run the API call, reconcile the returned Moment, and
    /// surface failures via `errorMessage`.
    private func dispatchAction(_ call: () async throws -> Moment) async {
        errorMessage = nil
        do {
            let updated = try await call()
            feed = Self.reconcile(feed: feed, updated: updated)
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

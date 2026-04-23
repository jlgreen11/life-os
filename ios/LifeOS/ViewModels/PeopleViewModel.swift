//
//  PeopleViewModel.swift
//  Life OS — People tab view model
//
//  Owns the `PeopleList` rendered by `PeopleTabView` plus the per-row
//  `ContactDossier` push. `query` is `@Published` so SwiftUI's
//  `.searchable` binding can read/write it directly; the view triggers
//  `load()` on changes (or with a debounce in a future iteration).
//
//  The dossier slot is a simple optional — the People tab uses
//  `NavigationLink(value:)` to push a detail pane, and that pane
//  reads `dossier` after calling `loadContact(id:)`.
//

import Foundation
import SwiftUI

@MainActor
final class PeopleViewModel: ObservableObject {

    // MARK: - Published state

    /// Roster driving the People tab. Defaults to an empty list with
    /// an empty `SelfPortrait` so the YOU pin renders the empty-state
    /// copy before `load()` returns.
    @Published var people: PeopleList = PeopleList(you: SelfPortrait())

    /// Last-loaded dossier (per-contact detail pane). Cleared when a
    /// new contact is requested.
    @Published var dossier: ContactDossier? = nil

    /// Search box bound to `.searchable`. Empty means "no filter" —
    /// `load()` translates `""` → `nil` on the wire.
    @Published var query: String = ""

    @Published var isLoading: Bool = false

    @Published var errorMessage: String? = nil

    // MARK: - Dependencies

    private let client: APIClientProtocol

    init(client: APIClientProtocol) {
        self.client = client
    }

    // MARK: - Loaders

    /// Fetch `/api/people` filtered by the current `query`. Empty
    /// query → no `q` query item.
    func load() async {
        isLoading = true
        errorMessage = nil
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        do {
            people = try await client.getPeople(
                query: q.isEmpty ? nil : q,
                page: 1,
                pageSize: nil
            )
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    /// Fetch `/api/people/{id}` and replace `dossier`. Used by the
    /// per-contact detail pane on push. Errors land in `errorMessage`
    /// and leave the previous `dossier` untouched.
    func loadContact(id: String) async {
        errorMessage = nil
        do {
            dossier = try await client.getContact(id: id)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Reset the dossier slot — called when the detail pane is popped
    /// so the next push starts from a clean state.
    func clearDossier() {
        dossier = nil
    }
}

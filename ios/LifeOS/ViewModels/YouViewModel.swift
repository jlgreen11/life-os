//
//  YouViewModel.swift
//  Life OS — You tab view model
//
//  Owns the `SelfPortrait` rendered by `YouTabView`. The view model is
//  read-mostly — the You tab has no per-row actions today (acceptance,
//  dismissal, and snooze all live on the Now tab) — so the API surface
//  is one method: `load()` against `/api/you`.
//

import Foundation
import SwiftUI

@MainActor
final class YouViewModel: ObservableObject {

    // MARK: - Published state

    /// Self-portrait driving the You tab. Defaults to a fresh-install
    /// `SelfPortrait()` so a view bound before `load()` returns sees
    /// the empty-state copy already wired in `YouTabView`.
    @Published var portrait: SelfPortrait = SelfPortrait()

    @Published var isLoading: Bool = false

    @Published var errorMessage: String? = nil

    // MARK: - Dependencies

    private let client: APIClientProtocol

    init(client: APIClientProtocol) {
        self.client = client
    }

    // MARK: - Loaders

    /// Fetch `/api/you` and replace `portrait`. On failure leaves
    /// `portrait` untouched and writes a string into `errorMessage`.
    func load() async {
        isLoading = true
        errorMessage = nil
        do {
            portrait = try await client.getYou()
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}

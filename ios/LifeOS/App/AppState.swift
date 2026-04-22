//
//  AppState.swift
//  Life OS — top-level app state
//
//  Holds the small slice of state that needs to live above the tab
//  shell:
//
//    • `serverURL`     — base URL of the v2 backend (Tailscale).
//    • `isConnected`   — last health-check result.
//    • `currentTab`    — selected tab in `RootTabView`.
//
//  Per-tab data (Now feed, You self-portrait, People list, Settings
//  connectors) is owned by per-tab view models that talk to
//  `APIClient` directly. This object stays lean so deep links and the
//  context pipeline can read/write it without pulling in tab-specific
//  state.
//

import SwiftUI

@MainActor
final class AppState: ObservableObject {
    @Published var serverURL: String {
        didSet { UserDefaults.standard.set(serverURL, forKey: "serverURL") }
    }

    @Published var isConnected: Bool = false
    @Published var currentTab: RootTab = .now

    init() {
        self.serverURL = UserDefaults.standard.string(forKey: "serverURL") ?? "http://localhost:8080"
    }
}

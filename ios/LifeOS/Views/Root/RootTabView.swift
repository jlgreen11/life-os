//
//  RootTabView.swift
//  Life OS — root navigation shell
//
//  4-tab IA per DESIGN.md §"Information Architecture":
//
//      Now · You · People · Settings (gear)
//
//  The `RootTab` enum is the single source of truth for tab identity,
//  display title, and SF Symbol icon. Tests assert against the enum
//  directly so we don't need a SwiftUI introspection harness to verify
//  the tab bar matches the wireframe.
//
//  Selection is held on `AppState.currentTab` so deep-link / URL
//  routing can mutate it from anywhere in the app.
//

import SwiftUI

/// One of the four root tabs. Order is the wireframe order from
/// DESIGN.md and is enforced by `RootTabViewTests`.
enum RootTab: String, CaseIterable, Identifiable {
    case now = "Now"
    case you = "You"
    case people = "People"
    case settings = "Settings"

    var id: String { rawValue }

    /// Human-visible tab label. Matches DESIGN.md §IA verbatim.
    var title: String { rawValue }

    /// SF Symbol name shown in the tab bar. DESIGN.md pins Settings to
    /// the gear glyph; the others are picked to read at 11pt without
    /// caption help.
    var systemImage: String {
        switch self {
        case .now: return "tray.full"
        case .you: return "person.crop.circle"
        case .people: return "person.2"
        case .settings: return "gear"
        }
    }
}

struct RootTabView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        TabView(selection: $appState.currentTab) {
            NowTabView()
                .tabItem {
                    Label(RootTab.now.title, systemImage: RootTab.now.systemImage)
                }
                .tag(RootTab.now)

            YouTabView()
                .tabItem {
                    Label(RootTab.you.title, systemImage: RootTab.you.systemImage)
                }
                .tag(RootTab.you)

            PeopleTabView()
                .tabItem {
                    Label(RootTab.people.title, systemImage: RootTab.people.systemImage)
                }
                .tag(RootTab.people)

            SettingsTabView()
                .tabItem {
                    Label(RootTab.settings.title, systemImage: RootTab.settings.systemImage)
                }
                .tag(RootTab.settings)
        }
        .tint(Color.primaryAction)
    }
}

#Preview {
    RootTabView()
        .environmentObject(AppState())
        .preferredColorScheme(.dark)
}

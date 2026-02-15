import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        TabView(selection: $appState.currentTab) {
            DashboardView()
                .tabItem {
                    Label("Dashboard", systemImage: "square.grid.2x2")
                }
                .tag(AppState.Tab.dashboard)

            ChatView()
                .tabItem {
                    Label("Assistant", systemImage: "bubble.left.and.bubble.right")
                }
                .tag(AppState.Tab.chat)

            ContextView()
                .tabItem {
                    Label("Context", systemImage: "location.viewfinder")
                }
                .tag(AppState.Tab.context)

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(AppState.Tab.settings)
        }
        .tint(.white)
    }
}

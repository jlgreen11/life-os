import SwiftUI

@main
struct LifeOSApp: App {
    @StateObject private var appState = AppState()
    @StateObject private var contextEngine = ContextEngine()

    var body: some Scene {
        WindowGroup {
            RootTabView()
                .environmentObject(appState)
                .environmentObject(contextEngine)
                .onAppear {
                    contextEngine.configure(serverURL: appState.serverURL)
                    contextEngine.startCollecting()
                }
                .preferredColorScheme(.dark)
        }
    }
}

import SwiftUI

@main
struct LifeOSApp: App {
    @StateObject private var appState = AppState()
    @StateObject private var contextEngine = ContextEngine()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState)
                .environmentObject(contextEngine)
                .onAppear {
                    contextEngine.startCollecting()
                }
                .preferredColorScheme(.dark)
        }
    }
}

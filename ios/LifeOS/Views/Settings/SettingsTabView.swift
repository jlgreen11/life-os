//
//  SettingsTabView.swift
//  Life OS — Settings tab (connectors + preferences)
//
//  Stub scaffold. Real connector list, `ConnectorEditView`, and
//  preferences (quiet hours, autonomy, proactivity) land in a follow-up
//  NEXT_TASKS.md task. Empty placeholder so the 4-tab shell compiles.
//

import SwiftUI

struct SettingsTabView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "Settings",
                systemImage: "gear",
                description: Text("Connectors + preferences scaffold — sections land in a follow-up task.")
            )
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .background(Color.bgBase.ignoresSafeArea())
        }
    }
}

#Preview {
    SettingsTabView()
        .preferredColorScheme(.dark)
}

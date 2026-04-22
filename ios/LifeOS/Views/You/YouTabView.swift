//
//  YouTabView.swift
//  Life OS — You tab (self-portrait)
//
//  Stub scaffold. Real sections (When you're at your best · How you
//  write · Your routines · Drifting) land in a follow-up NEXT_TASKS.md
//  task. Empty placeholder is wired to `RootTabView` so the 4-tab
//  shell compiles cleanly.
//

import SwiftUI

struct YouTabView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "You",
                systemImage: "person.crop.circle",
                description: Text("Self-portrait scaffold — sections land in a follow-up task.")
            )
            .navigationTitle("You")
            .navigationBarTitleDisplayMode(.large)
            .background(Color.bgBase.ignoresSafeArea())
        }
    }
}

#Preview {
    YouTabView()
        .preferredColorScheme(.dark)
}

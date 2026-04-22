//
//  PeopleTabView.swift
//  Life OS — People tab (contact list)
//
//  Stub scaffold. Real list (YOU pinned at top · NEEDS ATTENTION ·
//  ACTIVE THIS WEEK) plus the `ContactDossierView` pushed on tap land
//  in a follow-up NEXT_TASKS.md task. Empty placeholder so the 4-tab
//  shell compiles.
//

import SwiftUI

struct PeopleTabView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "People",
                systemImage: "person.2",
                description: Text("Contact list scaffold — roster lands in a follow-up task.")
            )
            .navigationTitle("People")
            .navigationBarTitleDisplayMode(.large)
            .background(Color.bgBase.ignoresSafeArea())
        }
    }
}

#Preview {
    PeopleTabView()
        .preferredColorScheme(.dark)
}

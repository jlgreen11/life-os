//
//  NowTabView.swift
//  Life OS — Now tab (action queue)
//
//  Stub scaffold. The real implementation (NOW · UP NEXT · DONE TODAY
//  sections plus `MomentCardView`) lands in the next NEXT_TASKS.md
//  task. This file exists so the 4-tab `RootTabView` compiles and the
//  rest of the iOS scaffold stack can be wired up around it.
//

import SwiftUI

struct NowTabView: View {
    var body: some View {
        NavigationStack {
            ContentUnavailableView(
                "Now",
                systemImage: "tray",
                description: Text("Action queue scaffold — content lands in the next task.")
            )
            .navigationTitle("Now")
            .navigationBarTitleDisplayMode(.large)
            .background(Color.bgBase.ignoresSafeArea())
        }
    }
}

#Preview {
    NowTabView()
        .preferredColorScheme(.dark)
}

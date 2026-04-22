//
//  NowTabView.swift
//  Life OS — Now tab (action queue)
//
//  Wireframe (DESIGN.md §Information Architecture + §Moment card):
//
//      ┌ NOW ────────────────────────────┐
//      │ MomentCardView (insight #1)     │
//      │ MomentCardView (insight #2)     │
//      │ MomentCardView (insight #3)     │
//      ├ UP NEXT ────────────────────────┤
//      │ 09:30 · Tuesday morning run…    │
//      │ arrive:home · Read for 20m…     │
//      ├ DONE TODAY (collapsed) ─────────┤
//      │   ⌄ expand to see 1 done        │
//      └─────────────────────────────────┘
//
//  "NOW" shows up to 3 pending Moments as full MomentCardView rows.
//  "UP NEXT" is a compact single-line summary per scheduled Moment.
//  "DONE TODAY" is collapsed by default (§IA: "DONE TODAY (collapsed)").
//
//  A lightweight ViewModel lands in the ViewModels task further down
//  NEXT_TASKS.md; for now the view owns its `MomentFeed` as `@State`
//  seeded from `MockData.feed`, and callers injecting a real feed go
//  through `init(feed:)`. Action callbacks are unwired stubs — this
//  task is scaffolding the IA, not the action pipeline.
//

import SwiftUI

struct NowTabView: View {

    /// Maximum number of cards shown in the NOW section before the UI
    /// asks the user to archive or dismiss to make room. DESIGN.md calls
    /// for "2-3 cards"; we take the upper bound.
    static let nowSectionLimit = 3

    // MARK: - Inputs

    /// Feed driving the view. `@State` so previews and tests can seed
    /// different fixtures without a view model.
    @State private var feed: MomentFeed

    /// Whether DONE TODAY is expanded. Collapsed by default per DESIGN.md.
    @State private var isDoneTodayExpanded: Bool = false

    init(feed: MomentFeed = MockData.feed) {
        _feed = State(initialValue: feed)
    }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.sectionGap) {
                    nowSection
                    upNextSection
                    doneTodaySection
                }
                .padding(.horizontal, Spacing.s4)
                .padding(.vertical, Spacing.s6)
            }
            .background(Color.bgBase.ignoresSafeArea())
            .navigationTitle("Now")
            .navigationBarTitleDisplayMode(.large)
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var nowSection: some View {
        let moments = Self.nowSectionMoments(from: feed)
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("NOW")
            if moments.isEmpty {
                emptyState(
                    title: "Nothing urgent.",
                    subtitle: "New Moments will appear here as evidence accumulates."
                )
            } else {
                VStack(spacing: Spacing.cardGap) {
                    ForEach(moments) { moment in
                        MomentCardView(moment: moment)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var upNextSection: some View {
        let moments = Self.upNextSectionMoments(from: feed)
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("UP NEXT")
            if moments.isEmpty {
                emptyState(
                    title: "Nothing scheduled.",
                    subtitle: "Time- and context-triggered Moments will show up here."
                )
            } else {
                VStack(spacing: Spacing.s2) {
                    ForEach(moments) { moment in
                        upNextRow(moment: moment)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var doneTodaySection: some View {
        let moments = Self.doneTodaySectionMoments(from: feed)
        VStack(alignment: .leading, spacing: Spacing.s3) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) {
                    isDoneTodayExpanded.toggle()
                }
            } label: {
                HStack(spacing: Spacing.s2) {
                    Text("DONE TODAY")
                        .font(.caption11.weight(FontWeightToken.semibold))
                        .tracking(LetterSpacing.caps)
                        .foregroundStyle(Color.textTertiary)
                    if !moments.isEmpty {
                        Text("\(moments.count)")
                            .font(.caption11.weight(FontWeightToken.semibold))
                            .foregroundStyle(Color.textTertiary)
                    }
                    Image(systemName: isDoneTodayExpanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: FontSize.t11, weight: .semibold))
                        .foregroundStyle(Color.textTertiary)
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Done today, \(moments.count) items, \(isDoneTodayExpanded ? "expanded" : "collapsed")")

            if isDoneTodayExpanded {
                if moments.isEmpty {
                    emptyState(
                        title: "Nothing done today yet.",
                        subtitle: "Accepted Moments land here."
                    )
                } else {
                    VStack(spacing: Spacing.s2) {
                        ForEach(moments) { moment in
                            doneRow(moment: moment)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Row builders

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption11.weight(FontWeightToken.semibold))
            .tracking(LetterSpacing.caps)
            .foregroundStyle(Color.textTertiary)
    }

    private func upNextRow(moment: Moment) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.s3) {
            Text(Self.upNextPrefix(for: moment))
                .font(.meta13.weight(FontWeightToken.medium))
                .foregroundStyle(Color.textSecondary)
                .frame(minWidth: 72, alignment: .leading)
            Text(moment.insight)
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, Spacing.s3)
        .padding(.vertical, Spacing.s2)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
    }

    private func doneRow(moment: Moment) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.s3) {
            Image(systemName: "checkmark")
                .font(.system(size: FontSize.t13, weight: .medium))
                .foregroundStyle(Color.statusSuccess)
            Text(moment.insight)
                .font(.body15)
                .foregroundStyle(Color.textSecondary)
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, Spacing.s3)
        .padding(.vertical, Spacing.s2)
    }

    private func emptyState(title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s1) {
            Text(title)
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
            Text(subtitle)
                .font(.meta13)
                .foregroundStyle(Color.textTertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Spacing.s4)
        .padding(.vertical, Spacing.s5)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.md))
    }

    // MARK: - Pure helpers (testable without rendering)

    /// Up to three pending Moments, which populate the full MomentCardView rows
    /// in the NOW section. DESIGN.md calls for "2-3 cards"; we take the upper
    /// bound as a soft cap.
    static func nowSectionMoments(from feed: MomentFeed, limit: Int = nowSectionLimit) -> [Moment] {
        Array(feed.pending.prefix(limit))
    }

    /// Every scheduled Moment, in feed order. UP NEXT stays compact so no
    /// prefix cap is applied — the user scrolls if the list gets long.
    static func upNextSectionMoments(from feed: MomentFeed) -> [Moment] {
        feed.scheduled
    }

    static func doneTodaySectionMoments(from feed: MomentFeed) -> [Moment] {
        feed.done
    }

    /// Leading metadata string shown on an UP NEXT row. Prefers the
    /// context-trigger expression (e.g. `"arrive:home"`) when present,
    /// else a `HH:mm` rendering of `scheduled_for`, else a fallback label.
    static func upNextPrefix(for moment: Moment) -> String {
        if let trigger = moment.contextTrigger?.expression, !trigger.isEmpty {
            return trigger
        }
        if let ts = moment.scheduledFor {
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "HH:mm"
            return f.string(from: ts)
        }
        return "later"
    }
}

// MARK: - Previews

#Preview("Populated feed") {
    NowTabView(feed: MockData.feed)
        .preferredColorScheme(.dark)
}

#Preview("Empty feed") {
    NowTabView(feed: MockData.emptyFeed)
        .preferredColorScheme(.dark)
}

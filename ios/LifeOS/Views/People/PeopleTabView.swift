//
//  PeopleTabView.swift
//  Life OS — People tab (contact list)
//
//  Wireframe (DESIGN.md §Information Architecture):
//
//      ┌ People ─────────────────────────┐
//      │ [ Search people…               ]│
//      ├ YOU ────────────────────────────┤
//      │ Observed 9 months · 1,842 …     │
//      ├ NEEDS ATTENTION ────────────────┤
//      │ Dad     9d ago      +4d         │
//      │ Maya    11d ago     +6d         │
//      ├ ACTIVE THIS WEEK ───────────────┤
//      │ Sam     2d ago                  │
//      │ Devon   3d ago                  │
//      └─────────────────────────────────┘
//
//  Hard rules from DESIGN.md:
//  - Right-aligned stats use SF Mono so the columns line up (tabular
//    numerals only — no pie charts, no avatars per §"What's NOT in
//    Phase 1").
//  - YOU row is pinned to the top; the two directory sections sit below.
//  - Tapping a contact row pushes `ContactDossierView` on the navigation
//    stack.
//  - Calm tone — "9d ago (+4d)", never "⚠ overdue".
//
//  The pure helpers (filter / last-seen label / cadence label / YOU
//  subtitle) are exposed statically so `PeopleTabViewTests` can pin
//  copy without spinning up a SwiftUI render tree.
//

import SwiftUI

struct PeopleTabView: View {

    // MARK: - Inputs

    /// Underlying roster. `@State` so previews and tests can seed
    /// different fixtures without a view model.
    @State private var peopleList: PeopleList

    /// User-entered search text. Filters the two directory sections
    /// (YOU is always shown — it's identity, not directory).
    @State private var query: String = ""

    /// Anchor timestamp used to format "last contact" labels. Tests
    /// freeze this so copy is deterministic; the default is `Date()`.
    private let anchor: Date

    init(
        peopleList: PeopleList = MockData.peopleList,
        anchor: Date = Date()
    ) {
        _peopleList = State(initialValue: peopleList)
        self.anchor = anchor
    }

    // MARK: - Body

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.sectionGap) {
                    youSection
                    needsAttentionSection
                    activeThisWeekSection
                }
                .padding(.horizontal, Spacing.s4)
                .padding(.vertical, Spacing.s6)
            }
            .background(Color.bgBase.ignoresSafeArea())
            .navigationTitle("People")
            .navigationBarTitleDisplayMode(.large)
            .navigationDestination(for: ContactSummary.self) { contact in
                ContactDossierView(dossier: Self.dossier(for: contact), anchor: anchor)
            }
            .searchable(text: $query, prompt: Text("Search people"))
        }
    }

    // MARK: - YOU (identity row)

    @ViewBuilder
    private var youSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("YOU")
            VStack(alignment: .leading, spacing: Spacing.s1) {
                Text("You")
                    .font(.body15.weight(FontWeightToken.semibold))
                    .foregroundStyle(Color.textPrimary)
                Text(Self.youSubtitle(for: peopleList.you))
                    .font(.meta13)
                    .foregroundStyle(Color.textSecondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, Spacing.s3)
            .padding(.vertical, Spacing.s3)
            .background(Color.bgRaised)
            .clipShape(RoundedRectangle(cornerRadius: Radius.md))
        }
    }

    // MARK: - NEEDS ATTENTION

    @ViewBuilder
    private var needsAttentionSection: some View {
        let rows = Self.filter(peopleList.needsAttention, query: query)
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("NEEDS ATTENTION")
            if peopleList.needsAttention.isEmpty {
                emptyState(
                    title: "Nobody is drifting.",
                    subtitle: "Cadence drift surfaces here once a contact slips past their usual gap."
                )
            } else if rows.isEmpty {
                emptyState(
                    title: "No matches.",
                    subtitle: "Nothing in NEEDS ATTENTION matches \"\(query)\"."
                )
            } else {
                VStack(spacing: Spacing.s2) {
                    ForEach(rows) { contact in
                        contactRow(contact, showCadence: true)
                    }
                }
            }
        }
    }

    // MARK: - ACTIVE THIS WEEK

    @ViewBuilder
    private var activeThisWeekSection: some View {
        let rows = Self.filter(peopleList.activeThisWeek, query: query)
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("ACTIVE THIS WEEK")
            if peopleList.activeThisWeek.isEmpty {
                emptyState(
                    title: "Nothing this week.",
                    subtitle: "Conversations from the last 7 days land here."
                )
            } else if rows.isEmpty {
                emptyState(
                    title: "No matches.",
                    subtitle: "Nothing in ACTIVE THIS WEEK matches \"\(query)\"."
                )
            } else {
                VStack(spacing: Spacing.s2) {
                    ForEach(rows) { contact in
                        contactRow(contact, showCadence: false)
                    }
                }
            }
        }
    }

    // MARK: - Row

    private func contactRow(_ contact: ContactSummary, showCadence: Bool) -> some View {
        NavigationLink(value: contact) {
            HStack(alignment: .firstTextBaseline, spacing: Spacing.s3) {
                Text(contact.name)
                    .font(.body15.weight(FontWeightToken.medium))
                    .foregroundStyle(Color.textPrimary)
                    .frame(minWidth: 96, alignment: .leading)
                Spacer(minLength: 0)
                Text(Self.lastSeenLabel(for: contact, anchor: anchor))
                    .font(.mono(size: FontSize.t13))
                    .foregroundStyle(Color.textSecondary)
                if showCadence, let cadence = Self.cadenceLabel(for: contact) {
                    Text(cadence)
                        .font(.mono(size: FontSize.t13))
                        .foregroundStyle(Color.textTertiary)
                        .frame(minWidth: 44, alignment: .trailing)
                }
            }
            .padding(.horizontal, Spacing.s3)
            .padding(.vertical, Spacing.s3)
            .background(Color.bgRaised)
            .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Self.accessibilityLabel(for: contact, anchor: anchor))
    }

    // MARK: - Chrome

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption11.weight(FontWeightToken.semibold))
            .tracking(LetterSpacing.caps)
            .foregroundStyle(Color.textTertiary)
            .accessibilityAddTraits(.isHeader)
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

    /// Case-insensitive name filter. Trims whitespace on the query and
    /// returns the unfiltered list when the query is empty.
    static func filter(_ contacts: [ContactSummary], query: String) -> [ContactSummary] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return contacts }
        let needle = trimmed.lowercased()
        return contacts.filter { $0.name.lowercased().contains(needle) }
    }

    /// Right-aligned monospace label showing how long ago the last
    /// contact was. Returns an em-dash sentinel when `lastContactTs`
    /// is nil. Uses whole-day granularity because the People list is
    /// a glanceable surface.
    static func lastSeenLabel(for contact: ContactSummary, anchor: Date) -> String {
        guard let ts = contact.lastContactTs else { return "—" }
        let last = Date(timeIntervalSince1970: TimeInterval(ts))
        let seconds = anchor.timeIntervalSince(last)
        if seconds < 0 { return "today" }
        let days = Int(seconds / 86_400)
        switch days {
        case 0: return "today"
        case 1: return "1d ago"
        default: return "\(days)d ago"
        }
    }

    /// Right-aligned monospace label summarizing cadence deviation.
    /// Returns nil when the row has no deviation recorded OR when the
    /// deviation is zero (calm default — nothing to flag).
    static func cadenceLabel(for contact: ContactSummary) -> String? {
        guard let deviation = contact.cadenceDeviationDays, deviation != 0 else {
            return nil
        }
        let sign = deviation > 0 ? "+" : "−"
        return "\(sign)\(abs(deviation))d"
    }

    /// Subtitle line for the pinned YOU row. Mirrors the You-tab header
    /// but stays compact so it fits in a single row. Falls back to a
    /// calm string on a fresh install.
    static func youSubtitle(for portrait: SelfPortrait) -> String {
        if portrait.observedMonths == 0 && portrait.interactionsCount == 0 {
            return "Just getting started — no observations yet."
        }
        let monthsLabel = portrait.observedMonths == 1 ? "month" : "months"
        let interactionsLabel = portrait.interactionsCount == 1 ? "interaction" : "interactions"
        return "Observed \(portrait.observedMonths) \(monthsLabel) · \(portrait.interactionsCount) \(interactionsLabel)"
    }

    /// VoiceOver label for a contact row — assembled from name + last
    /// seen so the screen-reader announcement reads as a sentence.
    static func accessibilityLabel(for contact: ContactSummary, anchor: Date) -> String {
        let last = lastSeenLabel(for: contact, anchor: anchor)
        if let cadence = cadenceLabel(for: contact) {
            return "\(contact.name). Last contact \(last). Cadence \(cadence)."
        }
        return "\(contact.name). Last contact \(last)."
    }

    /// Resolve a `ContactSummary` into a `ContactDossier` for the push
    /// destination. Today this is a static lookup because the People
    /// tab doesn't yet ship a ViewModel — a follow-up ViewModels task
    /// will replace this with an `APIClient` fetch.
    static func dossier(for contact: ContactSummary) -> ContactDossier {
        ContactDossier(
            contactId: contact.contactId,
            name: contact.name,
            lastContactTs: contact.lastContactTs,
            usualCadenceDays: contact.cadenceDeviationDays.map { abs($0) + 5 },
            commTemplate: nil,
            cadenceSparkline: [],
            recentTopics: [],
            predictedNext: nil
        )
    }
}

// MARK: - ContactSummary Hashable (NavigationLink value)

/// `NavigationLink(value:)` requires `Hashable` on the route. `ContactSummary`
/// conforms to `Equatable` for free via synthesis; we extend it to
/// `Hashable` so the pushed route can round-trip through the
/// `NavigationPath`.
extension ContactSummary: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(contactId)
    }
}

// MARK: - Previews

#Preview("Populated") {
    PeopleTabView(peopleList: MockData.peopleList, anchor: MockData.anchorDate)
        .preferredColorScheme(.dark)
}

#Preview("Empty install") {
    PeopleTabView(peopleList: MockData.emptyPeopleList, anchor: MockData.anchorDate)
        .preferredColorScheme(.dark)
}

//
//  YouTabView.swift
//  Life OS — You tab (self-portrait)
//
//  Wireframe (DESIGN.md §Information Architecture + §Copy voice):
//
//      ┌ You ────────────────────────────┐
//      │ Observed 9 months · 1,842       │
//      │ interactions                    │
//      ├ WHEN YOU'RE AT YOUR BEST ───────┤
//      │ Tuesday & Wednesday mornings…   │
//      │ After a 20-minute walk…         │
//      ├ HOW YOU WRITE ──────────────────┤
//      │ Family — Warm, short sentences  │
//      │ Work   — Direct, structured     │
//      ├ YOUR ROUTINES ──────────────────┤
//      │ Tuesday morning run · 9:15–9:45 │
//      │ Sunday review · 60-90m late aft │
//      │ Evening reading — not detected  │
//      ├ DRIFTING ───────────────────────┤
//      │ Dad   · 9 days (usual 5)        │
//      │ Maya  · 11 days (usual 5)       │
//      └─────────────────────────────────┘
//
//  Hard rules from DESIGN.md / NEXT_TASKS.md:
//  - NO mood bars, NO progress bars, NO pie charts (plain text only)
//  - Section order is locked: when_at_best → how_you_write →
//    your_routines → drifting
//  - Calm tone — "9 days (usual 5)", never "OVERDUE"
//  - Empty-state language is warm and tertiary, no cartoon CTAs
//
//  A full ViewModel lands in the ViewModels task further down
//  NEXT_TASKS.md; for now this view owns its `SelfPortrait` as
//  `@State` seeded from `MockData.selfPortrait` so previews and
//  XCTest can drive different fixtures through `init(portrait:)`.
//

import SwiftUI

// MARK: - Section enum (locked order)

/// Identity for the four You-tab sections. The enum's `allCases` order is
/// the *only* source of truth for section ordering — both the view body
/// and the XCTest suite read from it, so a wireframe change is one
/// surface away from a failing test.
enum YouSection: String, CaseIterable, Identifiable {
    case whenAtBest = "WHEN YOU'RE AT YOUR BEST"
    case howYouWrite = "HOW YOU WRITE"
    case yourRoutines = "YOUR ROUTINES"
    case drifting = "DRIFTING"

    var id: String { rawValue }

    /// Empty-state title for the section (shown when the underlying
    /// list is empty). Tertiary copy — no "Get started" CTA per
    /// DESIGN.md § "Empty states".
    var emptyTitle: String {
        switch self {
        case .whenAtBest:   return "Not enough signal yet."
        case .howYouWrite:  return "No writing patterns observed yet."
        case .yourRoutines: return "No routines detected yet."
        case .drifting:     return "Nobody is drifting."
        }
    }

    /// Tertiary subtitle paired with `emptyTitle`.
    var emptySubtitle: String {
        switch self {
        case .whenAtBest:   return "A pattern needs ~14 days of activity."
        case .howYouWrite:  return "Style summaries land after ~50 messages per audience."
        case .yourRoutines: return "Routines need 4+ matching weekly sessions."
        case .drifting:     return "Cadence drift shows up here once it slips past your usual gap."
        }
    }
}

// MARK: - View

struct YouTabView: View {

    // MARK: Inputs

    /// Self-portrait driving the view. `@State` so previews and tests
    /// can seed different fixtures without a view model.
    @State private var portrait: SelfPortrait

    init(portrait: SelfPortrait = MockData.selfPortrait) {
        _portrait = State(initialValue: portrait)
    }

    // MARK: Body

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.sectionGap) {
                    headerView
                    ForEach(YouSection.allCases) { section in
                        self.section(section)
                    }
                }
                .padding(.horizontal, Spacing.s4)
                .padding(.vertical, Spacing.s6)
            }
            .background(Color.bgBase.ignoresSafeArea())
            .navigationTitle("You")
            .navigationBarTitleDisplayMode(.large)
        }
    }

    // MARK: Header

    @ViewBuilder
    private var headerView: some View {
        VStack(alignment: .leading, spacing: Spacing.s1) {
            Text(Self.headerLine(for: portrait))
                .font(.body15)
                .foregroundStyle(Color.textSecondary)
                .accessibilityIdentifier("you.header")
            if portrait.confidencePct > 0 {
                Text("Confidence \(portrait.confidencePct)%")
                    .font(.meta13)
                    .foregroundStyle(Color.textTertiary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Section dispatch

    @ViewBuilder
    private func section(_ section: YouSection) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader(section.rawValue)
            switch section {
            case .whenAtBest:   whenAtBestSection
            case .howYouWrite:  howYouWriteSection
            case .yourRoutines: yourRoutinesSection
            case .drifting:     driftingSection
            }
        }
    }

    // MARK: WHEN YOU'RE AT YOUR BEST

    @ViewBuilder
    private var whenAtBestSection: some View {
        if portrait.whenAtBest.isEmpty {
            emptyState(for: .whenAtBest)
        } else {
            VStack(alignment: .leading, spacing: Spacing.s2) {
                ForEach(Array(portrait.whenAtBest.enumerated()), id: \.offset) { _, line in
                    plainTextRow(line)
                }
            }
        }
    }

    // MARK: HOW YOU WRITE

    @ViewBuilder
    private var howYouWriteSection: some View {
        if portrait.howYouWrite.isEmpty {
            emptyState(for: .howYouWrite)
        } else {
            VStack(alignment: .leading, spacing: Spacing.s2) {
                ForEach(Array(portrait.howYouWrite.enumerated()), id: \.offset) { _, style in
                    personaStyleRow(style)
                }
            }
        }
    }

    private func personaStyleRow(_ style: PersonaStyle) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.s3) {
            Text(style.audience)
                .font(.body15.weight(FontWeightToken.semibold))
                .foregroundStyle(Color.textPrimary)
                .frame(minWidth: 96, alignment: .leading)
            Text(style.tone)
                .font(.body15)
                .foregroundStyle(Color.textSecondary)
                .lineLimit(2)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, Spacing.s3)
        .padding(.vertical, Spacing.s2)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
    }

    // MARK: YOUR ROUTINES

    @ViewBuilder
    private var yourRoutinesSection: some View {
        if portrait.yourRoutines.isEmpty {
            emptyState(for: .yourRoutines)
        } else {
            VStack(alignment: .leading, spacing: Spacing.s2) {
                ForEach(Array(portrait.yourRoutines.enumerated()), id: \.offset) { _, routine in
                    routineRow(routine)
                }
            }
        }
    }

    private func routineRow(_ routine: Routine) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s1) {
            Text(routine.name)
                .font(.body15.weight(FontWeightToken.medium))
                .foregroundStyle(routine.detected ? Color.textPrimary : Color.textTertiary)
            Text(Self.routineSubtitle(for: routine))
                .font(.meta13)
                .foregroundStyle(Color.textTertiary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Spacing.s3)
        .padding(.vertical, Spacing.s2)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
    }

    // MARK: DRIFTING

    @ViewBuilder
    private var driftingSection: some View {
        if portrait.drifting.isEmpty {
            emptyState(for: .drifting)
        } else {
            VStack(alignment: .leading, spacing: Spacing.s2) {
                ForEach(portrait.drifting) { contact in
                    driftingRow(contact)
                }
            }
        }
    }

    private func driftingRow(_ contact: DriftingContact) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.s3) {
            Text(contact.name)
                .font(.body15.weight(FontWeightToken.medium))
                .foregroundStyle(Color.textPrimary)
                .frame(minWidth: 96, alignment: .leading)
            Text(Self.driftingDetail(for: contact))
                .font(.body15)
                .foregroundStyle(Color.textSecondary)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, Spacing.s3)
        .padding(.vertical, Spacing.s2)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
    }

    // MARK: Common chrome

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption11.weight(FontWeightToken.semibold))
            .tracking(LetterSpacing.caps)
            .foregroundStyle(Color.textTertiary)
            .accessibilityAddTraits(.isHeader)
    }

    private func plainTextRow(_ text: String) -> some View {
        Text(text)
            .font(.body15)
            .foregroundStyle(Color.textPrimary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, Spacing.s3)
            .padding(.vertical, Spacing.s2)
            .background(Color.bgRaised)
            .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
    }

    private func emptyState(for section: YouSection) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s1) {
            Text(section.emptyTitle)
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
            Text(section.emptySubtitle)
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

    /// Human header line shown under the navigation title:
    ///
    ///     "Observed 9 months · 1,842 interactions"
    ///
    /// Falls back to a calm phrasing for fresh installs ("Just getting
    /// started — no observations yet."). Numeric grouping uses the
    /// current locale so `1842` becomes `1,842` on en-US, `1.842` on
    /// de-DE, etc.
    static func headerLine(for portrait: SelfPortrait) -> String {
        if portrait.observedMonths == 0 && portrait.interactionsCount == 0 {
            return "Just getting started — no observations yet."
        }
        let monthsLabel = portrait.observedMonths == 1 ? "month" : "months"
        let interactions = numberFormatter.string(from: NSNumber(value: portrait.interactionsCount))
            ?? "\(portrait.interactionsCount)"
        let interactionsLabel = portrait.interactionsCount == 1 ? "interaction" : "interactions"
        return "Observed \(portrait.observedMonths) \(monthsLabel) · \(interactions) \(interactionsLabel)"
    }

    /// Subtitle line for a routine row — shows the description for
    /// detected routines, "Not detected yet." otherwise.
    static func routineSubtitle(for routine: Routine) -> String {
        if !routine.detected {
            return "Not detected yet."
        }
        return routine.description ?? "Detected."
    }

    /// Drifting-row right-hand text. Calm phrasing per DESIGN.md
    /// (`"Dad · 9 days (usual 5)"`, never `"⚠ OVERDUE"`).
    static func driftingDetail(for contact: DriftingContact) -> String {
        let dayLabel = contact.daysSinceLast == 1 ? "day" : "days"
        return "\(contact.daysSinceLast) \(dayLabel) (usual \(contact.usualCadenceDays))"
    }

    /// Locale-aware integer formatter shared by the header builder.
    private static let numberFormatter: NumberFormatter = {
        let f = NumberFormatter()
        f.numberStyle = .decimal
        return f
    }()
}

// MARK: - Previews

#Preview("Populated") {
    YouTabView(portrait: MockData.selfPortrait)
        .preferredColorScheme(.dark)
}

#Preview("Empty install") {
    YouTabView(portrait: MockData.emptySelfPortrait)
        .preferredColorScheme(.dark)
}

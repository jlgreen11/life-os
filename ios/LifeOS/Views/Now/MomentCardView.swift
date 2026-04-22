//
//  MomentCardView.swift
//  Life OS — Now tab
//
//  The single building block of the Now tab's action queue. Each card
//  renders one `Moment`:
//
//    • Insight line — 22pt display weight, the headline of the card.
//    • Evidence link — tap reveals a sheet listing raw evidence strings.
//      DESIGN.md §Principles: "Evidence is a feature."
//    • Draft block — shown only when the proposed action is a
//      `draftMessage` *and* the params carry a `body` string. Uses the
//      draft tint + `Radius.lg` corner per DESIGN.md.
//    • Action row — exactly one primary button, optional secondaries
//      (Edit / Snooze), a rightmost ghost dismiss button.
//
//  View presentation uses card REST state only (`--bg-raised` +
//  `--elev-1`). Hover / pressed / focus states land when the iOS UI
//  layer gets real interaction and keyboard support — the scaffolding
//  is in `Elevation.swift`.
//

import SwiftUI

struct MomentCardView: View {

    // MARK: - Inputs

    let moment: Moment

    /// Callback when the user taps the primary action. Default no-op so
    /// the scaffold + previews compile without a view model.
    var onPrimary: (Moment) -> Void = { _ in }
    /// Callback when the user taps the ghost dismiss button.
    var onDismiss: (Moment) -> Void = { _ in }
    /// Callback when the user taps "Snooze 3d".
    var onSnooze: (Moment) -> Void = { _ in }
    /// Callback when the user taps "Edit" on a draft.
    var onEditDraft: (Moment) -> Void = { _ in }

    // MARK: - Internal state

    @State private var isEvidenceSheetPresented = false

    // MARK: - Body

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.s4) {
            insightText

            if !moment.evidence.isEmpty {
                evidenceLink
            }

            if let draft = Self.draftBody(from: moment) {
                draftBlock(text: draft)
            }

            actionRow
        }
        .padding(.horizontal, Spacing.cardPaddingHorizontal)
        .padding(.vertical, Spacing.cardPaddingVertical)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.md))
        .elevation(.rest)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Self.accessibilityLabel(for: moment))
        .sheet(isPresented: $isEvidenceSheetPresented) {
            EvidenceSheet(moment: moment)
                .presentationDetents([.medium, .large])
        }
    }

    // MARK: - Subviews

    private var insightText: some View {
        Text(moment.insight)
            .font(.headline22)
            .foregroundStyle(Color.textPrimary)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var evidenceLink: some View {
        Button {
            isEvidenceSheetPresented = true
        } label: {
            HStack(spacing: Spacing.s1) {
                Text(Self.evidenceLinkLabel(for: moment))
                    .font(.meta13)
                Image(systemName: "chevron.right")
                    .font(.system(size: FontSize.t11, weight: .semibold))
            }
            .foregroundStyle(Color.textSecondary)
        }
        .buttonStyle(.plain)
        .accessibilityHint("Shows the raw evidence behind this insight.")
    }

    private func draftBlock(text: String) -> some View {
        VStack(alignment: .leading, spacing: Spacing.s2) {
            Text("DRAFT")
                .font(.caption11.weight(FontWeightToken.semibold))
                .tracking(LetterSpacing.caps)
                .foregroundStyle(Color.textTertiary)
            Text(text)
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(Spacing.s4)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.draftBg)
        .overlay(
            RoundedRectangle(cornerRadius: Radius.lg)
                .strokeBorder(Color.draftBorder, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: Radius.lg))
    }

    private var actionRow: some View {
        HStack(spacing: Spacing.s2) {
            primaryButton

            if Self.hasDraft(moment) {
                ghostButton(title: "Edit") { onEditDraft(moment) }
            }

            ghostButton(title: "Snooze 3d") { onSnooze(moment) }

            Spacer(minLength: 0)

            dismissButton
        }
    }

    private var primaryButton: some View {
        Button {
            onPrimary(moment)
        } label: {
            Text(Self.primaryActionLabel(for: moment.proposedAction.kind))
                .font(.body15.weight(FontWeightToken.medium))
                .foregroundStyle(.white)
                .padding(.horizontal, Spacing.s3)
                .padding(.vertical, Spacing.s2)
                .background(Color.primaryAction)
                .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Self.primaryActionLabel(for: moment.proposedAction.kind))
    }

    private func ghostButton(title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.body15.weight(FontWeightToken.medium))
                .foregroundStyle(Color.textPrimary)
                .padding(.horizontal, Spacing.s3)
                .padding(.vertical, Spacing.s2)
                .overlay(
                    RoundedRectangle(cornerRadius: Radius.sm)
                        .strokeBorder(Color.borderStrong, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }

    private var dismissButton: some View {
        Button {
            onDismiss(moment)
        } label: {
            Image(systemName: "xmark")
                .font(.system(size: FontSize.t13, weight: .medium))
                .foregroundStyle(Color.textTertiary)
                .frame(width: 36, height: 36)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Dismiss")
    }

    // MARK: - Pure helpers (testable without rendering)

    /// The primary-action label shown on a Moment card. The mapping lives
    /// here because it's small, stable, and lets tests pin it down without
    /// spinning up a SwiftUI render tree.
    static func primaryActionLabel(for kind: ActionKind) -> String {
        switch kind {
        case .draftMessage: return "Start a message"
        case .sendMessage: return "Send"
        case .scheduleBlock: return "Schedule block"
        case .archiveEvent: return "Archive"
        case .nudge: return "Start timer"
        case .setReminder: return "Remind me"
        case .createCalendarEntry: return "Add to calendar"
        case .noteObservation: return "Note it"
        }
    }

    /// Body text to render inside the draft block, or `nil` when no draft
    /// is applicable. Returns nil for any non-`draftMessage` action and
    /// for drafts that ship without a `body` param.
    static func draftBody(from moment: Moment) -> String? {
        guard moment.proposedAction.kind == .draftMessage else { return nil }
        guard let raw = moment.proposedAction.params["body"]?.value as? String else {
            return nil
        }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    /// True when the card should expose an "Edit" secondary action — i.e.
    /// there's a draft body the user could edit. Same predicate as
    /// `draftBody(from:) != nil`; named separately so call sites read clean.
    static func hasDraft(_ moment: Moment) -> Bool {
        draftBody(from: moment) != nil
    }

    /// Human-readable evidence-link label. DESIGN.md §"Copy voice":
    /// "From 4 conversations · Feb 12 – Apr 3." We don't have date ranges
    /// here so we emit the count form.
    static func evidenceLinkLabel(for moment: Moment) -> String {
        let n = moment.evidence.count
        if n == 1 { return "From 1 source" }
        return "From \(n) sources"
    }

    /// Accessibility label combining insight + primary action. Screen
    /// readers should hear the full sentence + what happens on activation.
    static func accessibilityLabel(for moment: Moment) -> String {
        "\(moment.insight). Primary action: \(primaryActionLabel(for: moment.proposedAction.kind))."
    }
}

// MARK: - Evidence sheet

/// Bottom sheet presented when the evidence link is tapped. Just a list
/// of raw evidence strings — rendering matches DESIGN.md body type.
struct EvidenceSheet: View {
    let moment: Moment

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(Array(moment.evidence.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.body15)
                            .foregroundStyle(Color.textPrimary)
                            .listRowBackground(Color.bgRaised)
                    }
                } header: {
                    Text("EVIDENCE")
                        .font(.caption11.weight(FontWeightToken.semibold))
                        .tracking(LetterSpacing.caps)
                        .foregroundStyle(Color.textTertiary)
                }
            }
            .scrollContentBackground(.hidden)
            .background(Color.bgBase.ignoresSafeArea())
            .navigationTitle("Evidence")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

// MARK: - Previews

#Preview("Draft + evidence") {
    MomentCardView(moment: MockData.momentCadence)
        .padding()
        .background(Color.bgBase)
        .preferredColorScheme(.dark)
}

#Preview("No draft") {
    MomentCardView(moment: MockData.momentTemporal)
        .padding()
        .background(Color.bgBase)
        .preferredColorScheme(.dark)
}

#Preview("Done (archived)") {
    MomentCardView(moment: MockData.momentDone)
        .padding()
        .background(Color.bgBase)
        .preferredColorScheme(.dark)
}

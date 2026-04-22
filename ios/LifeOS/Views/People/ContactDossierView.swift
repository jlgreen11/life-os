//
//  ContactDossierView.swift
//  Life OS — People tab / per-contact detail pane
//
//  Wireframe (DESIGN.md §People + §"Copy voice"):
//
//      ┌ Dad ────────────────────────────┐
//      │ Last contact 9d ago             │
//      ├ COMMUNICATION STYLE ────────────┤
//      │ Short, warm check-ins. …        │
//      ├ RECENT TOPICS ──────────────────┤
//      │ · Garden tomato harvest         │
//      │ · Sunday dinner plans           │
//      ├ CADENCE ────────────────────────┤
//      │ ╱╲╱╲   (sparkline)              │
//      │ 14 days · usual every 5 days    │
//      ├ PREDICTED NEXT ─────────────────┤
//      │ Likely worth reaching out in…   │
//      ├─────────────────────────────────┤
//      │ [Start a message]               │
//      └─────────────────────────────────┘
//
//  Hard rules from DESIGN.md:
//  - Sparkline is a plain SwiftUI `Path` — NO chart library, NO bars,
//    NO dots. One smooth line so fresh-install renders as a flat line.
//  - NO avatars. Name is rendered in the navigation title; below it is
//    meta text only.
//  - Exactly one primary action: `[Start a message]` (per DESIGN.md
//    §Action button hierarchy — "Start a message" is the canonical
//    draft-message primary label).
//
//  Every copy-producing helper is static so tests can assert against
//  the exact strings without rendering.
//

import SwiftUI

struct ContactDossierView: View {

    // MARK: - Inputs

    let dossier: ContactDossier

    /// Callback when the user taps the primary action. Default no-op
    /// so previews + stub navigation compile without a view model.
    var onStartMessage: (ContactDossier) -> Void = { _ in }

    /// Anchor date used to format the "last contact" line. Tests freeze
    /// this; production uses `Date()`.
    private let anchor: Date

    init(
        dossier: ContactDossier,
        anchor: Date = Date(),
        onStartMessage: @escaping (ContactDossier) -> Void = { _ in }
    ) {
        self.dossier = dossier
        self.anchor = anchor
        self.onStartMessage = onStartMessage
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.sectionGap) {
                headerBlock
                commStyleSection
                recentTopicsSection
                cadenceSection
                predictedNextSection
                primaryActionRow
            }
            .padding(.horizontal, Spacing.s4)
            .padding(.vertical, Spacing.s6)
        }
        .background(Color.bgBase.ignoresSafeArea())
        .navigationTitle(dossier.name)
        .navigationBarTitleDisplayMode(.large)
    }

    // MARK: - Sections

    @ViewBuilder
    private var headerBlock: some View {
        Text(Self.lastSeenLine(for: dossier, anchor: anchor))
            .font(.body15)
            .foregroundStyle(Color.textSecondary)
            .accessibilityIdentifier("dossier.lastSeen")
    }

    @ViewBuilder
    private var commStyleSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("COMMUNICATION STYLE")
            Text(Self.commStyleOrFallback(for: dossier))
                .font(.body15)
                .foregroundStyle(dossier.commTemplate == nil
                                 ? Color.textTertiary
                                 : Color.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(Spacing.s4)
                .background(Color.bgRaised)
                .clipShape(RoundedRectangle(cornerRadius: Radius.md))
        }
    }

    @ViewBuilder
    private var recentTopicsSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("RECENT TOPICS")
            if dossier.recentTopics.isEmpty {
                Text("No topics extracted yet.")
                    .font(.body15)
                    .foregroundStyle(Color.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(Spacing.s4)
                    .background(Color.bgRaised)
                    .clipShape(RoundedRectangle(cornerRadius: Radius.md))
            } else {
                VStack(alignment: .leading, spacing: Spacing.s2) {
                    ForEach(Array(dossier.recentTopics.enumerated()), id: \.offset) { _, topic in
                        topicRow(topic)
                    }
                }
            }
        }
    }

    private func topicRow(_ topic: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.s3) {
            Text("·")
                .font(.body15)
                .foregroundStyle(Color.textTertiary)
            Text(topic)
                .font(.body15)
                .foregroundStyle(Color.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, Spacing.s3)
        .padding(.vertical, Spacing.s2)
        .background(Color.bgRaised)
        .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
    }

    @ViewBuilder
    private var cadenceSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("CADENCE")
            VStack(alignment: .leading, spacing: Spacing.s3) {
                CadenceSparkline(values: dossier.cadenceSparkline)
                    .frame(height: 48)
                    .accessibilityLabel(Self.sparklineAccessibilityLabel(for: dossier))
                Text(Self.cadenceFooter(for: dossier))
                    .font(.meta13)
                    .foregroundStyle(Color.textTertiary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(Spacing.s4)
            .background(Color.bgRaised)
            .clipShape(RoundedRectangle(cornerRadius: Radius.md))
        }
    }

    @ViewBuilder
    private var predictedNextSection: some View {
        VStack(alignment: .leading, spacing: Spacing.s3) {
            sectionHeader("PREDICTED NEXT")
            Text(Self.predictedNextLine(for: dossier))
                .font(.body15)
                .foregroundStyle(dossier.predictedNext == nil
                                 ? Color.textTertiary
                                 : Color.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(Spacing.s4)
                .background(Color.bgRaised)
                .clipShape(RoundedRectangle(cornerRadius: Radius.md))
        }
    }

    @ViewBuilder
    private var primaryActionRow: some View {
        HStack {
            Button {
                onStartMessage(dossier)
            } label: {
                Text("Start a message")
                    .font(.body15.weight(FontWeightToken.medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, Spacing.s4)
                    .padding(.vertical, Spacing.s3)
                    .background(Color.primaryAction)
                    .clipShape(RoundedRectangle(cornerRadius: Radius.sm))
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Start a message")
            Spacer(minLength: 0)
        }
    }

    // MARK: - Chrome

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption11.weight(FontWeightToken.semibold))
            .tracking(LetterSpacing.caps)
            .foregroundStyle(Color.textTertiary)
            .accessibilityAddTraits(.isHeader)
    }

    // MARK: - Pure helpers (testable without rendering)

    /// "Last contact 9d ago" / "No contact recorded yet." Derived the
    /// same way as the row label on the People list, except rendered
    /// as a full sentence.
    static func lastSeenLine(for dossier: ContactDossier, anchor: Date) -> String {
        guard let ts = dossier.lastContactTs else {
            return "No contact recorded yet."
        }
        let last = Date(timeIntervalSince1970: TimeInterval(ts))
        let seconds = anchor.timeIntervalSince(last)
        if seconds < 0 { return "Last contact today." }
        let days = Int(seconds / 86_400)
        switch days {
        case 0: return "Last contact today."
        case 1: return "Last contact 1 day ago."
        default: return "Last contact \(days) days ago."
        }
    }

    /// Communication-style copy. Returns the stored template verbatim
    /// when present, else a tertiary fallback.
    static func commStyleOrFallback(for dossier: ContactDossier) -> String {
        if let template = dossier.commTemplate, !template.isEmpty {
            return template
        }
        return "Style summary lands after enough messages to establish a voice."
    }

    /// Predicted-next copy. Returns the stored hint verbatim when
    /// present, else a tertiary fallback. Never alarm copy.
    static func predictedNextLine(for dossier: ContactDossier) -> String {
        if let predicted = dossier.predictedNext, !predicted.isEmpty {
            return predicted
        }
        return "No prediction yet — waiting on more observed cadence."
    }

    /// Footer under the sparkline. Composes `"14 days · usual every 5
    /// days"` when both the sparkline and a `usualCadenceDays` are
    /// present; degrades gracefully to the parts that ARE populated.
    static func cadenceFooter(for dossier: ContactDossier) -> String {
        let sparkPart = dossier.cadenceSparkline.isEmpty
            ? "No recent contact data."
            : "\(dossier.cadenceSparkline.count) days"
        guard let usual = dossier.usualCadenceDays else { return sparkPart }
        let dayWord = usual == 1 ? "day" : "days"
        let usualPart = "usual every \(usual) \(dayWord)"
        if dossier.cadenceSparkline.isEmpty {
            return "\(sparkPart) Usual every \(usual) \(dayWord)."
        }
        return "\(sparkPart) · \(usualPart)"
    }

    /// Map a `[Int]` cadence series into a set of `CGPoint`s spanning
    /// `size`. Used by `CadenceSparkline` and locked here so tests can
    /// assert shape invariants without rendering.
    ///
    /// Invariants:
    ///   - Empty series → `[]`.
    ///   - Single-value series → one point at `(0, midY)`.
    ///   - Multi-value series → `values.count` points, X evenly spaced
    ///     from `0` to `size.width`, Y scaled so min maps to `size.height`
    ///     and max maps to `0` (SwiftUI's inverted Y).
    static func sparklinePoints(for values: [Int], in size: CGSize) -> [CGPoint] {
        if values.isEmpty { return [] }
        if values.count == 1 {
            return [CGPoint(x: 0, y: size.height / 2)]
        }
        let minVal = values.min() ?? 0
        let maxVal = values.max() ?? 0
        let range = maxVal - minVal
        let stepX = size.width / CGFloat(values.count - 1)
        return values.enumerated().map { index, value in
            let x = CGFloat(index) * stepX
            let normalized: CGFloat
            if range == 0 {
                normalized = 0.5
            } else {
                normalized = CGFloat(value - minVal) / CGFloat(range)
            }
            let y = size.height - (normalized * size.height)
            return CGPoint(x: x, y: y)
        }
    }

    /// VoiceOver label for the sparkline. The chart is decorative for
    /// sighted users; screen readers hear a plain summary instead.
    static func sparklineAccessibilityLabel(for dossier: ContactDossier) -> String {
        if dossier.cadenceSparkline.isEmpty {
            return "No recent cadence data."
        }
        let total = dossier.cadenceSparkline.reduce(0, +)
        let days = dossier.cadenceSparkline.count
        let contactWord = total == 1 ? "contact" : "contacts"
        return "Cadence over \(days) days. \(total) \(contactWord) recorded."
    }
}

// MARK: - CadenceSparkline

/// Plain SwiftUI `Path` rendering of a cadence series. No library, no
/// ornaments — one line segment per data point, stroked in
/// `--primary-action`. Falls back to a horizontal baseline when the
/// series is empty so the section still occupies the same vertical
/// rhythm as a populated one.
struct CadenceSparkline: View {
    let values: [Int]

    var body: some View {
        GeometryReader { geo in
            Path { path in
                let points = ContactDossierView.sparklinePoints(for: values, in: geo.size)
                guard let first = points.first else {
                    path.move(to: CGPoint(x: 0, y: geo.size.height / 2))
                    path.addLine(to: CGPoint(x: geo.size.width, y: geo.size.height / 2))
                    return
                }
                path.move(to: first)
                for point in points.dropFirst() {
                    path.addLine(to: point)
                }
            }
            .stroke(
                values.isEmpty ? Color.borderStrong : Color.primaryAction,
                style: StrokeStyle(lineWidth: 1.5, lineCap: .round, lineJoin: .round)
            )
        }
    }
}

// MARK: - Previews

#Preview("Populated") {
    NavigationStack {
        ContactDossierView(dossier: MockData.contactDossierDad, anchor: MockData.anchorDate)
    }
    .preferredColorScheme(.dark)
}

#Preview("Sparse") {
    NavigationStack {
        ContactDossierView(dossier: MockData.contactDossierSparse, anchor: MockData.anchorDate)
    }
    .preferredColorScheme(.dark)
}

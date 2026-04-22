//
//  Font+Tokens.swift
//  Life OS — Design System
//
//  Type scale and weight ladder ported from DESIGN.md (Phase 1).
//  Pair rule: use display wrapper at `t22` / `t28`; text wrapper everywhere else.
//

import SwiftUI

/// Point sizes for the six-step type scale (11 / 13 / 15 / 17 / 22 / 28).
enum FontSize {
    /// Caption.
    static let t11: CGFloat = 11
    /// Meta.
    static let t13: CGFloat = 13
    /// Body.
    static let t15: CGFloat = 15
    /// Callout.
    static let t17: CGFloat = 17
    /// Headline (Moment insight).
    static let t22: CGFloat = 22
    /// Title (tab header).
    static let t28: CGFloat = 28
}

/// Weight ladder. DESIGN.md names map to SwiftUI `Font.Weight`.
enum FontWeightToken {
    static let regular: Font.Weight = .regular
    static let medium: Font.Weight = .medium
    static let semibold: Font.Weight = .semibold
    static let bold: Font.Weight = .bold
}

/// Line-height multipliers. SwiftUI uses point spacing, so most callers want
/// `lineSpacing = size * (multiplier - 1)`.
enum LineHeight {
    static let tight: CGFloat = 1.15
    static let snug: CGFloat = 1.30
    /// Default line height (named to avoid colliding with Swift's `default`).
    static let standard: CGFloat = 1.50
    static let loose: CGFloat = 1.60
}

/// Letter spacing (em). `caps` applies to uppercase section labels.
enum LetterSpacing {
    static let tight: CGFloat = -0.01
    static let normal: CGFloat = 0
    static let caps: CGFloat = 0.08
}

// MARK: - Font wrappers

extension Font {
    /// Display-tier wrapper (for the `t22` and `t28` steps). Uses the system
    /// SF Pro Display face via `.default` design on Apple platforms.
    static func display(size: CGFloat, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    /// Text-tier wrapper for all sub-22pt steps. Resolves to SF Pro Text.
    static func text(size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    /// SF Mono wrapper (used by People-tab stats).
    static func mono(size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }

    // Named roles — lean on these instead of ad-hoc sizes in views.
    static var caption11: Font { .text(size: FontSize.t11) }
    static var meta13: Font { .text(size: FontSize.t13) }
    static var body15: Font { .text(size: FontSize.t15) }
    static var callout17: Font { .text(size: FontSize.t17) }
    static var headline22: Font { .display(size: FontSize.t22, weight: .semibold) }
    static var title28: Font { .display(size: FontSize.t28, weight: .bold) }
}

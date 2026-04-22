//
//  Radius.swift
//  Life OS — Design System
//
//  Corner radii ported from DESIGN.md. Cards use `md`, buttons use `sm`
//  (or `pill` for an isolated primary), drafts use `lg`.
//

import CoreGraphics

enum Radius {
    static let sm: CGFloat = 6
    static let md: CGFloat = 10
    static let lg: CGFloat = 14
    /// Matches CSS `--r-pill` (999px) — large enough to render as a pill at
    /// any realistic control height.
    static let pill: CGFloat = 999
}

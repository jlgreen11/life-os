//
//  Spacing.swift
//  Life OS — Design System
//
//  8pt-base spacing scale ported from DESIGN.md. Step numbers mirror the
//  CSS custom properties (`--s-4` → `Spacing.s4`) so designer/engineer
//  dialogue stays synchronized.
//

import CoreGraphics

enum Spacing {
    static let s0: CGFloat = 0
    static let s1: CGFloat = 4
    static let s2: CGFloat = 8
    static let s3: CGFloat = 12
    static let s4: CGFloat = 16
    static let s5: CGFloat = 20
    static let s6: CGFloat = 24
    static let s8: CGFloat = 32
    static let s10: CGFloat = 40
    static let s12: CGFloat = 48
    static let s16: CGFloat = 64

    /// Card inner padding (horizontal).
    static let cardPaddingHorizontal: CGFloat = s4
    /// Card inner padding (vertical).
    static let cardPaddingVertical: CGFloat = s5
    /// Gap between Moment cards.
    static let cardGap: CGFloat = s6
    /// Gap between sections (NOW / UP NEXT / DONE TODAY).
    static let sectionGap: CGFloat = s8
}

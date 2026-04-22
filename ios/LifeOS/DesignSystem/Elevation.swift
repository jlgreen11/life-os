//
//  Elevation.swift
//  Life OS — Design System
//
//  Shadow + focus-ring helpers ported from DESIGN.md:
//    --elev-1:     0 1px 2px rgba(0,0,0,0.32);             // card rest
//    --elev-2:     0 4px 12px rgba(0,0,0,0.40);            // card hover
//    --elev-focus: 0 0 0 2px var(--border-focus);
//    --elev-modal: 0 16px 48px rgba(0,0,0,0.60);
//

import SwiftUI

struct ElevationStyle: Equatable {
    let color: Color
    let radius: CGFloat
    let x: CGFloat
    let y: CGFloat

    /// No shadow (the `--elev-0` sentinel).
    static let none = ElevationStyle(color: .clear, radius: 0, x: 0, y: 0)
    /// Card rest state — `--elev-1`.
    static let rest = ElevationStyle(color: .black.opacity(0.32), radius: 2, x: 0, y: 1)
    /// Card hover state — `--elev-2`.
    static let hover = ElevationStyle(color: .black.opacity(0.40), radius: 12, x: 0, y: 4)
    /// Modal surface — `--elev-modal`.
    static let modal = ElevationStyle(color: .black.opacity(0.60), radius: 48, x: 0, y: 16)
}

extension View {
    /// Apply one of the DESIGN.md elevation tokens as a drop shadow.
    func elevation(_ style: ElevationStyle) -> some View {
        shadow(color: style.color, radius: style.radius, x: style.x, y: style.y)
    }

    /// Apply the `--elev-focus` ring. Pair with a matching corner radius on
    /// the host view; defaults to `Radius.md` (card corner).
    func focusRing(cornerRadius: CGFloat = Radius.md, color: Color = .borderFocus) -> some View {
        overlay(
            RoundedRectangle(cornerRadius: cornerRadius)
                .strokeBorder(color, lineWidth: 2)
        )
    }
}

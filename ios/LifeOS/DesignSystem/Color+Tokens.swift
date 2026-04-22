//
//  Color+Tokens.swift
//  Life OS — Design System
//
//  Semantic color tokens ported from DESIGN.md (dark theme, Phase 1).
//  Hex values must match DESIGN.md exactly; keep in sync.
//

import SwiftUI

// MARK: - Hex source-of-truth

/// Raw hex values for every opaque design token. Exposed separately from
/// `Color` so tests can assert each value without reconstructing floats from
/// rendered sRGB components. Translucent tokens (overlays, borders, draft
/// tints) live in ``DesignColorRGBA`` below because they carry alpha.
enum DesignColorHex {
    // Backgrounds
    static let bgBase: UInt32 = 0x0A0A0C
    static let bgRaised: UInt32 = 0x141417
    static let bgRaisedHover: UInt32 = 0x1B1B1F
    static let bgSunken: UInt32 = 0x060607

    // Text
    static let textPrimary: UInt32 = 0xF2F2F5
    static let textSecondary: UInt32 = 0xA8A8AE
    static let textTertiary: UInt32 = 0x6A6A70
    static let textDisabled: UInt32 = 0x44444A

    // Primary action
    static let primaryAction: UInt32 = 0x0A84FF
    static let primaryActionHover: UInt32 = 0x1F93FF
    static let primaryActionPressed: UInt32 = 0x006EDC

    // Status
    static let success: UInt32 = 0x30D158
    static let warning: UInt32 = 0xFF9F0A
    static let error: UInt32 = 0xFF453A
    static let info: UInt32 = 0x64D2FF
}

/// Translucent tokens. `(r, g, b, a)` on a 0–1 scale to match SwiftUI.
enum DesignColorRGBA {
    /// `rgba(255, 255, 255, 0.04)` from DESIGN.md.
    static let bgOverlay: (r: Double, g: Double, b: Double, a: Double) = (1.0, 1.0, 1.0, 0.04)
    /// `rgba(255, 255, 255, 0.06)`.
    static let borderSubtle: (r: Double, g: Double, b: Double, a: Double) = (1.0, 1.0, 1.0, 0.06)
    /// `rgba(255, 255, 255, 0.12)`.
    static let borderStrong: (r: Double, g: Double, b: Double, a: Double) = (1.0, 1.0, 1.0, 0.12)
    /// `rgba(10, 132, 255, 0.06)` — draft card background.
    static let draftBg: (r: Double, g: Double, b: Double, a: Double) = (10.0 / 255.0, 132.0 / 255.0, 255.0 / 255.0, 0.06)
    /// `rgba(10, 132, 255, 0.12)` — draft card border.
    static let draftBorder: (r: Double, g: Double, b: Double, a: Double) = (10.0 / 255.0, 132.0 / 255.0, 255.0 / 255.0, 0.12)
}

// MARK: - Color helpers

extension Color {
    /// Construct an sRGB color from a 24-bit hex value (e.g. `0x0A84FF`).
    init(hex: UInt32, opacity: Double = 1.0) {
        let r = Double((hex >> 16) & 0xFF) / 255.0
        let g = Double((hex >> 8) & 0xFF) / 255.0
        let b = Double(hex & 0xFF) / 255.0
        self.init(.sRGB, red: r, green: g, blue: b, opacity: opacity)
    }

    fileprivate init(rgba: (r: Double, g: Double, b: Double, a: Double)) {
        self.init(.sRGB, red: rgba.r, green: rgba.g, blue: rgba.b, opacity: rgba.a)
    }
}

// MARK: - Semantic tokens

extension Color {
    // Backgrounds
    static let bgBase = Color(hex: DesignColorHex.bgBase)
    static let bgRaised = Color(hex: DesignColorHex.bgRaised)
    static let bgRaisedHover = Color(hex: DesignColorHex.bgRaisedHover)
    static let bgSunken = Color(hex: DesignColorHex.bgSunken)
    static let bgOverlay = Color(rgba: DesignColorRGBA.bgOverlay)

    // Text
    static let textPrimary = Color(hex: DesignColorHex.textPrimary)
    static let textSecondary = Color(hex: DesignColorHex.textSecondary)
    static let textTertiary = Color(hex: DesignColorHex.textTertiary)
    static let textDisabled = Color(hex: DesignColorHex.textDisabled)

    // Primary action
    static let primaryAction = Color(hex: DesignColorHex.primaryAction)
    static let primaryActionHover = Color(hex: DesignColorHex.primaryActionHover)
    static let primaryActionPressed = Color(hex: DesignColorHex.primaryActionPressed)

    // Status
    static let statusSuccess = Color(hex: DesignColorHex.success)
    static let statusWarning = Color(hex: DesignColorHex.warning)
    static let statusError = Color(hex: DesignColorHex.error)
    static let statusInfo = Color(hex: DesignColorHex.info)

    // Borders
    static let borderSubtle = Color(rgba: DesignColorRGBA.borderSubtle)
    static let borderStrong = Color(rgba: DesignColorRGBA.borderStrong)
    /// Focus ring matches `--primary-action` per DESIGN.md.
    static var borderFocus: Color { .primaryAction }

    // Draft tints
    static let draftBg = Color(rgba: DesignColorRGBA.draftBg)
    static let draftBorder = Color(rgba: DesignColorRGBA.draftBorder)
}

//
//  DesignSystemTests.swift
//  Life OS — Design System Tests
//
//  Verifies the DesignSystem constants match DESIGN.md exactly. These tests
//  do not need an iOS device — they assert on raw numeric constants.
//

import XCTest
import SwiftUI
@testable import LifeOS

final class DesignColorHexTests: XCTestCase {

    // MARK: - Backgrounds (DESIGN.md §Color)

    func test_bgBase_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.bgBase, 0x0A0A0C)
    }

    func test_bgRaised_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.bgRaised, 0x141417)
    }

    func test_bgRaisedHover_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.bgRaisedHover, 0x1B1B1F)
    }

    func test_bgSunken_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.bgSunken, 0x060607)
    }

    // MARK: - Text

    func test_textPrimary_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.textPrimary, 0xF2F2F5)
    }

    func test_textSecondary_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.textSecondary, 0xA8A8AE)
    }

    func test_textTertiary_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.textTertiary, 0x6A6A70)
    }

    func test_textDisabled_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.textDisabled, 0x44444A)
    }

    // MARK: - Primary action

    func test_primaryAction_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.primaryAction, 0x0A84FF)
    }

    func test_primaryActionHover_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.primaryActionHover, 0x1F93FF)
    }

    func test_primaryActionPressed_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.primaryActionPressed, 0x006EDC)
    }

    // MARK: - Status

    func test_success_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.success, 0x30D158)
    }

    func test_warning_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.warning, 0xFF9F0A)
    }

    func test_error_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.error, 0xFF453A)
    }

    func test_info_matchesDesignMd() {
        XCTAssertEqual(DesignColorHex.info, 0x64D2FF)
    }
}

final class DesignColorRGBATests: XCTestCase {

    func test_bgOverlay_isWhiteAt4Percent() {
        let t = DesignColorRGBA.bgOverlay
        XCTAssertEqual(t.r, 1.0, accuracy: 0.001)
        XCTAssertEqual(t.g, 1.0, accuracy: 0.001)
        XCTAssertEqual(t.b, 1.0, accuracy: 0.001)
        XCTAssertEqual(t.a, 0.04, accuracy: 0.001)
    }

    func test_borderSubtle_isWhiteAt6Percent() {
        let t = DesignColorRGBA.borderSubtle
        XCTAssertEqual(t.a, 0.06, accuracy: 0.001)
        XCTAssertEqual(t.r, 1.0, accuracy: 0.001)
    }

    func test_borderStrong_isWhiteAt12Percent() {
        let t = DesignColorRGBA.borderStrong
        XCTAssertEqual(t.a, 0.12, accuracy: 0.001)
        XCTAssertEqual(t.r, 1.0, accuracy: 0.001)
    }

    func test_draftBg_usesPrimaryActionAt6Percent() {
        let t = DesignColorRGBA.draftBg
        // rgba(10, 132, 255, 0.06)
        XCTAssertEqual(t.r, 10.0 / 255.0, accuracy: 0.001)
        XCTAssertEqual(t.g, 132.0 / 255.0, accuracy: 0.001)
        XCTAssertEqual(t.b, 255.0 / 255.0, accuracy: 0.001)
        XCTAssertEqual(t.a, 0.06, accuracy: 0.001)
    }

    func test_draftBorder_usesPrimaryActionAt12Percent() {
        let t = DesignColorRGBA.draftBorder
        XCTAssertEqual(t.r, 10.0 / 255.0, accuracy: 0.001)
        XCTAssertEqual(t.g, 132.0 / 255.0, accuracy: 0.001)
        XCTAssertEqual(t.b, 255.0 / 255.0, accuracy: 0.001)
        XCTAssertEqual(t.a, 0.12, accuracy: 0.001)
    }
}

final class ColorHexInitTests: XCTestCase {

    /// Smoke test that the `Color(hex:)` helper round-trips a well-known value.
    /// We can't introspect the sRGB components of a SwiftUI `Color` directly
    /// on every platform, so the smoke test just asserts construction
    /// succeeds and two identical hex inits produce equal colors.
    func test_hexInit_isDeterministic() {
        let a = Color(hex: 0x0A84FF)
        let b = Color(hex: 0x0A84FF)
        XCTAssertEqual(a, b)
    }

    func test_semanticTokens_resolvedFromHexConstants() {
        // Tokens should be constructible without throwing / trapping.
        _ = Color.bgBase
        _ = Color.bgRaised
        _ = Color.textPrimary
        _ = Color.primaryAction
        _ = Color.statusSuccess
        _ = Color.statusError
        _ = Color.borderSubtle
        _ = Color.draftBg
    }
}

final class FontTokenTests: XCTestCase {

    func test_typeScale_matchesDesignMd() {
        XCTAssertEqual(FontSize.t11, 11)
        XCTAssertEqual(FontSize.t13, 13)
        XCTAssertEqual(FontSize.t15, 15)
        XCTAssertEqual(FontSize.t17, 17)
        XCTAssertEqual(FontSize.t22, 22)
        XCTAssertEqual(FontSize.t28, 28)
    }

    func test_weightLadder_mapsToSwiftUI() {
        XCTAssertEqual(FontWeightToken.regular, .regular)
        XCTAssertEqual(FontWeightToken.medium, .medium)
        XCTAssertEqual(FontWeightToken.semibold, .semibold)
        XCTAssertEqual(FontWeightToken.bold, .bold)
    }

    func test_lineHeight_multipliersMatchDesignMd() {
        XCTAssertEqual(LineHeight.tight, 1.15, accuracy: 0.0001)
        XCTAssertEqual(LineHeight.snug, 1.30, accuracy: 0.0001)
        XCTAssertEqual(LineHeight.standard, 1.50, accuracy: 0.0001)
        XCTAssertEqual(LineHeight.loose, 1.60, accuracy: 0.0001)
    }

    func test_letterSpacing_matchesDesignMd() {
        XCTAssertEqual(LetterSpacing.tight, -0.01, accuracy: 0.0001)
        XCTAssertEqual(LetterSpacing.normal, 0, accuracy: 0.0001)
        XCTAssertEqual(LetterSpacing.caps, 0.08, accuracy: 0.0001)
    }

    func test_namedRoles_exist() {
        _ = Font.caption11
        _ = Font.meta13
        _ = Font.body15
        _ = Font.callout17
        _ = Font.headline22
        _ = Font.title28
    }
}

final class SpacingTests: XCTestCase {

    func test_scale_matchesDesignMd() {
        XCTAssertEqual(Spacing.s0, 0)
        XCTAssertEqual(Spacing.s1, 4)
        XCTAssertEqual(Spacing.s2, 8)
        XCTAssertEqual(Spacing.s3, 12)
        XCTAssertEqual(Spacing.s4, 16)
        XCTAssertEqual(Spacing.s5, 20)
        XCTAssertEqual(Spacing.s6, 24)
        XCTAssertEqual(Spacing.s8, 32)
        XCTAssertEqual(Spacing.s10, 40)
        XCTAssertEqual(Spacing.s12, 48)
        XCTAssertEqual(Spacing.s16, 64)
    }

    func test_cardPadding_matchesDesignMdRule() {
        // "Card inner padding: --s-4 horizontal × --s-5 vertical."
        XCTAssertEqual(Spacing.cardPaddingHorizontal, Spacing.s4)
        XCTAssertEqual(Spacing.cardPaddingVertical, Spacing.s5)
    }

    func test_cardGap_and_sectionGap_matchDesignMdRule() {
        XCTAssertEqual(Spacing.cardGap, Spacing.s6)
        XCTAssertEqual(Spacing.sectionGap, Spacing.s8)
    }
}

final class RadiusTests: XCTestCase {

    func test_radii_matchDesignMd() {
        XCTAssertEqual(Radius.sm, 6)
        XCTAssertEqual(Radius.md, 10)
        XCTAssertEqual(Radius.lg, 14)
        XCTAssertEqual(Radius.pill, 999)
    }
}

final class ElevationTests: XCTestCase {

    func test_restElevation_matchesDesignMd() {
        let s = ElevationStyle.rest
        XCTAssertEqual(s.x, 0)
        XCTAssertEqual(s.y, 1)
        XCTAssertEqual(s.radius, 2)
        XCTAssertEqual(s.color, Color.black.opacity(0.32))
    }

    func test_hoverElevation_matchesDesignMd() {
        let s = ElevationStyle.hover
        XCTAssertEqual(s.x, 0)
        XCTAssertEqual(s.y, 4)
        XCTAssertEqual(s.radius, 12)
        XCTAssertEqual(s.color, Color.black.opacity(0.40))
    }

    func test_modalElevation_matchesDesignMd() {
        let s = ElevationStyle.modal
        XCTAssertEqual(s.x, 0)
        XCTAssertEqual(s.y, 16)
        XCTAssertEqual(s.radius, 48)
        XCTAssertEqual(s.color, Color.black.opacity(0.60))
    }

    func test_noneElevation_isNoop() {
        let s = ElevationStyle.none
        XCTAssertEqual(s.radius, 0)
        XCTAssertEqual(s.x, 0)
        XCTAssertEqual(s.y, 0)
        XCTAssertEqual(s.color, .clear)
    }
}

//
//  RootTabViewTests.swift
//  Life OS — Root tab shell tests
//
//  SwiftUI's `TabView` doesn't expose an introspection surface that
//  works on every Xcode/SDK pair without a third-party library. We
//  drive verification through the `RootTab` enum instead — it's the
//  single source of truth `RootTabView` reads from, so an enum
//  mismatch == a tab-bar mismatch.
//
//  These tests are pure value-type assertions; they need no simulator.
//

import XCTest
@testable import LifeOS

final class RootTabViewTests: XCTestCase {

    // MARK: - Tab order

    func test_fourTabsPresent_inWireframeOrder() {
        // DESIGN.md §"Information Architecture":
        //   "4 tabs: Now · You · People · Settings (gear)."
        XCTAssertEqual(RootTab.allCases.count, 4)
        XCTAssertEqual(RootTab.allCases, [.now, .you, .people, .settings])
    }

    // MARK: - Titles

    func test_tabTitles_matchDESIGNmd() {
        XCTAssertEqual(RootTab.now.title, "Now")
        XCTAssertEqual(RootTab.you.title, "You")
        XCTAssertEqual(RootTab.people.title, "People")
        XCTAssertEqual(RootTab.settings.title, "Settings")
    }

    func test_titleEqualsRawValue_soStorageRoundTrips() {
        for tab in RootTab.allCases {
            XCTAssertEqual(tab.title, tab.rawValue)
        }
    }

    // MARK: - SF Symbols

    func test_eachTab_hasNonEmptySFSymbol() {
        for tab in RootTab.allCases {
            XCTAssertFalse(
                tab.systemImage.isEmpty,
                "RootTab.\(tab) is missing a tab-bar SF Symbol"
            )
        }
    }

    func test_settingsTab_usesGearGlyph_perDESIGNmd() {
        // DESIGN.md explicitly pins "Settings (gear)".
        XCTAssertEqual(RootTab.settings.systemImage, "gear")
    }

    func test_iconAssignments_areStable() {
        // Lock the chosen glyphs in. Changing any of these is a UX
        // change and should invalidate this test on purpose.
        XCTAssertEqual(RootTab.now.systemImage, "tray.full")
        XCTAssertEqual(RootTab.you.systemImage, "person.crop.circle")
        XCTAssertEqual(RootTab.people.systemImage, "person.2")
        XCTAssertEqual(RootTab.settings.systemImage, "gear")
    }

    func test_iconsAreUnique_acrossTabs() {
        let icons = RootTab.allCases.map(\.systemImage)
        XCTAssertEqual(
            Set(icons).count,
            icons.count,
            "Two tabs share an SF Symbol — riders won't be able to tell them apart"
        )
    }

    // MARK: - Identifiable

    func test_idEqualsRawValue() {
        for tab in RootTab.allCases {
            XCTAssertEqual(tab.id, tab.rawValue)
        }
    }
}

// MARK: - AppState wiring

final class AppStateRootTabTests: XCTestCase {

    @MainActor
    func test_appState_defaultsToNowTab() {
        let state = AppState()
        XCTAssertEqual(state.currentTab, .now)
    }

    @MainActor
    func test_appState_currentTabIsMutable() {
        let state = AppState()
        state.currentTab = .people
        XCTAssertEqual(state.currentTab, .people)
        state.currentTab = .settings
        XCTAssertEqual(state.currentTab, .settings)
    }
}

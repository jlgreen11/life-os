//
//  APIClientProtocol.swift
//  Life OS — testable seam over `APIClient`
//
//  Every per-tab view model holds an `APIClientProtocol` instead of a
//  concrete `APIClient`. Production wiring passes the live actor;
//  XCTest passes a hand-rolled mock that records calls and lets each
//  test pre-stage the response (success or error) for one method at a
//  time.
//
//  The protocol covers exactly the surface the four view models need:
//
//    Now      — getNow / acceptMoment / dismissMoment / snoozeMoment /
//               undoMoment / editMoment
//    You      — getYou
//    People   — getPeople / getContact
//    Settings — getConnectors / updateConnector / updatePreference
//
//  Endpoints used elsewhere in the app (health, status, briefing,
//  context pipeline) intentionally stay off the protocol — they are
//  consumed by `AppState` / `ContextEngine` and don't need view-model
//  isolation. Adding them later is a one-line change here plus an
//  extension on the mock.
//

import Foundation

/// Async surface every view model uses to talk to the v2 backend.
///
/// `APIClient` (the production actor) gets a free conformance via the
/// extension at the bottom of this file — its method signatures already
/// match. The mock used in `ViewModelsTests` is a plain class so test
/// state can be poked synchronously between calls.
///
/// Methods that take an optional `annotation` / `pageSize` mirror the
/// defaults on `APIClient` itself; callers that want defaults pass
/// `nil` explicitly because protocols cannot carry default arguments.
protocol APIClientProtocol {
    // MARK: Now tab

    func getNow() async throws -> MomentFeed
    func acceptMoment(id: String, annotation: String?) async throws -> Moment
    func dismissMoment(id: String, annotation: String?) async throws -> Moment
    func snoozeMoment(id: String, snoozeUntil: Int, annotation: String?) async throws -> Moment
    func undoMoment(id: String) async throws -> Moment
    func editMoment(id: String, actionParams: [String: AnyCodable]) async throws -> Moment

    // MARK: You tab

    func getYou() async throws -> SelfPortrait

    // MARK: People tab

    func getPeople(query: String?, page: Int, pageSize: Int?) async throws -> PeopleList
    func getContact(id: String) async throws -> ContactDossier

    // MARK: Settings tab

    func getConnectors() async throws -> [Connector]
    func updateConnector(id: String, update: ConnectorConfigUpdate) async throws -> Connector
    func updatePreference(key: String, value: AnyCodable) async throws
}

// `APIClient` already exposes every method on the protocol with the
// matching signature — declare the conformance here so view-model
// constructors can accept the actor without any glue code.
extension APIClient: APIClientProtocol {}

import SwiftUI

struct ChatView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var contextEngine: ContextEngine
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var isProcessing = false
    @FocusState private var isInputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Context banner
                contextBanner

                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(messages) { message in
                                MessageBubble(message: message) { suggestion in
                                    inputText = suggestion
                                    sendMessage()
                                }
                            }
                        }
                        .padding()
                    }
                    .onChange(of: messages.count) { _, _ in
                        if let last = messages.last {
                            withAnimation {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }

                Divider()

                // Input bar
                inputBar
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Assistant")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("Morning Briefing") { sendPrefilled("briefing") }
                        Button("What should I do next?") { sendPrefilled("What should I focus on right now?") }
                        Button("Summarize my day") { sendPrefilled("Summarize my day so far") }
                        Button("Draft a message") { sendPrefilled("Help me draft a message") }
                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }
            }
            .onAppear {
                if messages.isEmpty {
                    addSystemMessage()
                }
            }
        }
    }

    // MARK: - Context Banner

    private var contextBanner: some View {
        HStack(spacing: 8) {
            if let loc = contextEngine.locationManager.currentLocation {
                Label(loc.placeName ?? "Unknown", systemImage: "location.fill")
                    .font(.caption2)
                    .foregroundStyle(.blue)
            }
            Label("\(contextEngine.deviceDiscovery.nearbyDevices.count) devices", systemImage: "antenna.radiowaves.left.and.right")
                .font(.caption2)
                .foregroundStyle(.purple)
            Spacer()
            Text("AI uses this context")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
        .background(Color(.secondarySystemGroupedBackground))
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Ask anything...", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .focused($isInputFocused)
                .onSubmit { sendMessage() }

            Button {
                sendMessage()
            } label: {
                Image(systemName: isProcessing ? "stop.circle.fill" : "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundStyle(inputText.trimmingCharacters(in: .whitespaces).isEmpty ? .gray : .blue)
            }
            .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty && !isProcessing)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color(.secondarySystemGroupedBackground))
    }

    // MARK: - Actions

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        inputText = ""

        let userMessage = ChatMessage(
            role: .user,
            content: text,
            timestamp: Date(),
            suggestions: nil,
            actions: nil
        )
        messages.append(userMessage)

        isProcessing = true

        // Build context-enriched command
        let contextPrefix = buildContextPrefix()
        let enrichedCommand = contextPrefix.isEmpty ? text : "\(contextPrefix)\n\(text)"

        Task {
            if let response = await appState.sendCommand(enrichedCommand) {
                let assistantMessage = ChatMessage(
                    role: .assistant,
                    content: response.content,
                    timestamp: Date(),
                    suggestions: response.suggestions,
                    actions: response.actions
                )
                messages.append(assistantMessage)
            } else {
                let errorMessage = ChatMessage(
                    role: .system,
                    content: "Failed to get a response. Check your connection.",
                    timestamp: Date(),
                    suggestions: nil,
                    actions: nil
                )
                messages.append(errorMessage)
            }
            isProcessing = false
        }
    }

    private func sendPrefilled(_ text: String) {
        inputText = text
        sendMessage()
    }

    private func buildContextPrefix() -> String {
        var parts: [String] = []

        if let loc = contextEngine.locationManager.currentLocation {
            let place = loc.placeName ?? "(\(String(format: "%.4f", loc.coordinate.latitude)), \(String(format: "%.4f", loc.coordinate.longitude)))"
            parts.append("[Location: \(place)]")
        }

        let devices = contextEngine.deviceDiscovery.nearbyDevices
        let attributed = devices.compactMap { $0.attributedTo }
        if !attributed.isEmpty {
            parts.append("[Nearby people: \(attributed.joined(separator: ", "))]")
        }

        if let snapshot = contextEngine.latestSnapshot {
            parts.append("[Time: \(snapshot.timeContext.dayOfWeek) \(snapshot.timeContext.partOfDay)]")
        }

        return parts.isEmpty ? "" : "[Context] " + parts.joined(separator: " ")
    }

    private func addSystemMessage() {
        let msg = ChatMessage(
            role: .system,
            content: "Life OS Assistant is ready. I have access to your location, nearby devices, and time context to provide personalized help.",
            timestamp: Date(),
            suggestions: ["What's my schedule today?", "Summarize notifications", "Who's nearby?"],
            actions: nil
        )
        messages.append(msg)
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ChatMessage
    var onSuggestionTap: ((String) -> Void)?

    var body: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
            HStack {
                if message.role == .user { Spacer() }

                VStack(alignment: .leading, spacing: 6) {
                    Text(message.content)
                        .font(.body)
                        .foregroundStyle(message.role == .system ? .secondary : .primary)

                    if let suggestions = message.suggestions, !suggestions.isEmpty {
                        FlowLayout(spacing: 6) {
                            ForEach(suggestions, id: \.self) { suggestion in
                                Button {
                                    onSuggestionTap?(suggestion)
                                } label: {
                                    Text(suggestion)
                                        .font(.caption)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(Color.blue.opacity(0.1))
                                        .clipShape(Capsule())
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                }
                .padding(12)
                .background(bubbleBackground)
                .clipShape(RoundedRectangle(cornerRadius: 16))

                if message.role != .user { Spacer() }
            }

            Text(timeString)
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .padding(.horizontal, 4)
        }
    }

    private var bubbleBackground: Color {
        switch message.role {
        case .user: return Color.blue.opacity(0.2)
        case .assistant: return Color(.secondarySystemGroupedBackground)
        case .system: return Color(.tertiarySystemGroupedBackground)
        }
    }

    private var timeString: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: message.timestamp)
    }
}

// MARK: - Flow Layout

struct FlowLayout: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = computeLayout(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = computeLayout(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y),
                                  proposal: .unspecified)
        }
    }

    private func computeLayout(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var currentX: CGFloat = 0
        var currentY: CGFloat = 0
        var lineHeight: CGFloat = 0
        var maxX: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if currentX + size.width > maxWidth && currentX > 0 {
                currentX = 0
                currentY += lineHeight + spacing
                lineHeight = 0
            }
            positions.append(CGPoint(x: currentX, y: currentY))
            lineHeight = max(lineHeight, size.height)
            currentX += size.width + spacing
            maxX = max(maxX, currentX)
        }

        return (CGSize(width: maxX, height: currentY + lineHeight), positions)
    }
}

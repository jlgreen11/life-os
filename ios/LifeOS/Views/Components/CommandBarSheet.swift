import SwiftUI

struct CommandBarSheet: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var result: CommandResponse?
    @State private var isSearching = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search your life or ask anything...", text: $query)
                        .textFieldStyle(.plain)
                        .onSubmit { executeSearch() }
                    if isSearching {
                        ProgressView()
                            .controlSize(.small)
                    }
                }
                .padding()
                .background(Color(.secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal)

                if let result = result {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            Text(result.content)
                                .font(.body)
                                .padding()
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(.secondarySystemGroupedBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 12))

                            if let suggestions = result.suggestions, !suggestions.isEmpty {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("Suggestions")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    ForEach(suggestions, id: \.self) { suggestion in
                                        Button {
                                            query = suggestion
                                            executeSearch()
                                        } label: {
                                            Text(suggestion)
                                                .font(.subheadline)
                                                .frame(maxWidth: .infinity, alignment: .leading)
                                                .padding(8)
                                                .background(Color(.tertiarySystemGroupedBackground))
                                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                        }
                                        .buttonStyle(.plain)
                                    }
                                }
                                .padding(.horizontal)
                            }
                        }
                        .padding(.horizontal)
                    }
                }

                Spacer()
            }
            .navigationTitle("Search")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func executeSearch() {
        guard !query.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        isSearching = true
        Task {
            result = await appState.sendCommand(query)
            isSearching = false
        }
    }
}

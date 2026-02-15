import SwiftUI

struct NotificationRow: View {
    let notification: LifeOSNotification
    @EnvironmentObject var appState: AppState

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Circle()
                .fill(priorityColor)
                .frame(width: 8, height: 8)
                .padding(.top, 6)

            VStack(alignment: .leading, spacing: 4) {
                Text(notification.title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Text(notification.body)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)

                HStack(spacing: 8) {
                    if let source = notification.source {
                        Text(source)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    Text(notification.relativeTime)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                if let actions = notification.actions, !actions.isEmpty {
                    HStack(spacing: 8) {
                        ForEach(actions) { action in
                            Button(action.label) {
                                Task {
                                    await appState.submitFeedback(FeedbackPayload(
                                        type: "engaged",
                                        targetId: notification.id,
                                        targetType: "notification",
                                        value: action.type
                                    ))
                                }
                            }
                            .font(.caption2)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.blue.opacity(0.15))
                            .clipShape(Capsule())
                        }
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var priorityColor: Color {
        switch notification.priority {
        case "critical": return .red
        case "high": return .orange
        case "normal": return .blue
        default: return .gray
        }
    }
}

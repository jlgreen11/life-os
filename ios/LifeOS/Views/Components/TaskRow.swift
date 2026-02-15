import SwiftUI

struct TaskRow: View {
    let task: LifeOSTask

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: task.statusIcon)
                .foregroundStyle(statusColor)
                .font(.body)

            VStack(alignment: .leading, spacing: 2) {
                Text(task.title)
                    .font(.subheadline)
                    .strikethrough(task.isComplete)
                    .foregroundStyle(task.isComplete ? .secondary : .primary)

                HStack(spacing: 6) {
                    Text(task.priority)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(priorityColor.opacity(0.15))
                        .foregroundStyle(priorityColor)
                        .clipShape(Capsule())

                    if let due = task.dueDate {
                        Label(due, systemImage: "calendar")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    if let contacts = task.relatedContacts, !contacts.isEmpty {
                        Label(contacts.first ?? "", systemImage: "person")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Spacer()
        }
        .padding(.vertical, 4)
    }

    private var statusColor: Color {
        switch task.status {
        case "completed": return .green
        case "in_progress": return .blue
        default: return .gray
        }
    }

    private var priorityColor: Color {
        switch task.priority {
        case "critical": return .red
        case "high": return .orange
        case "normal": return .blue
        default: return .gray
        }
    }
}

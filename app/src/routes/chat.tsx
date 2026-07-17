import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/chat')({
  component: ChatComponent,
})

function ChatComponent() {
  return <div>Hello "/Chat"!</div>
}

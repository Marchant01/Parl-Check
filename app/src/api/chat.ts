import type { ChatRequest, ChatResponse } from '../types/chat';

const API_BASE = import.meta.env.API_BASE

export async function askQuestion(question: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question } satisfies ChatRequest),
  })

  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`)
  }

  return res.json()
}
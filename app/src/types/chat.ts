export interface  ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  question: string
}

export interface ChatResponse {
  answer: string
}

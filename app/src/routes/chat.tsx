import { createFileRoute } from '@tanstack/react-router';
import { useMutation } from '@tanstack/react-query';
import { useState } from 'react';
import { askQuestion } from '../api/chat';
import type { ChatMessage } from '../types/chat';

export const Route = createFileRoute('/chat')({
  component: ChatComponent,
});

function ChatComponent() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])

  const mutation = useMutation({
    mutationFn: askQuestion,

    onSuccess: (data) => {
      setMessages(prev => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: data.answer,
        },
      ])
    }
  });

  const handleeSubmit = (e: React.SubmitEvent) => {
    e.preventDefault()

    if (!input.trim() || mutation.isPending) return

    const question = input

    setMessages(prev => [
      ...prev,
      { id: crypto.randomUUID(), 
        role: 'user', 
        content: question 
      },
    ])

    setInput('')
  
    mutation.mutate(question)
  };

  return (
    <div className='flex justify-center'>
      <div className='flex flex-col w-full h-full gap-3 px-8 py-5'>
        {/* Load chat here */}
        {messages.map((msg) => (
          <div 
            key={msg.id} 
            className={`flex flex-col min-w-fit max-w-3xl border-2 px-2 py-2 bg-white border-black shadow-[3px_3px_0_#14171A]
                ${msg.role == 'user' ? 'self-end' : 'self-start'}
              `}
            >
            <p>{msg.content}</p>
          </div>
        ))}

        {mutation.isPending && (
          <div>
            <p>Hämtar svar...</p>
          </div>
        )}

        {mutation.isError && (
          <div>
            <strong>Ett fel uppstod! Försök igen senare.</strong>
          </div>
        )}
      </div>

      <form 
        className='px-8 py-5 border-t2 border-[#14171A] fixed bottom-0'
        onSubmit={handleeSubmit}
      >
        <div className='max-w-2xl mx-auto flex gap-3'>
          <input 
            id='chat-input' 
            className='flex-1 border-2 border-[#14171A] w-2xl py-2 px-2 outline-none focus:shadow-[3px_3px_0_#14171A] transition-shadow bg-white' 
            type='text' 
            placeholder='Hur har Moderaterna ställt...'
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button 
            type='submit'
            className='cursor-pointer px-5 py-2 font-bold border-2 border-[#14171A] transition-transform hover:-translate-y-0.5'
            style={{background: '#FC7A1E', color: 'black'}}
          >
            Skicka
          </button>
        </div>
      </form>
    </div>
  );
};
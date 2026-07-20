import { createFileRoute } from '@tanstack/react-router';
import { useState } from 'react';

export const Route = createFileRoute('/chat')({
  component: ChatComponent,
});

function ChatComponent() {
  const [input, setInput] = useState('')

  return (
    <div className='flex justify-center'>
      <div className=''>
        {/* Load chat here */}
      </div>

      <form 
        className='px-8 py-5 border-t2 border-[#14171A] fixed bottom-0'
        onSubmit={(e) => { e.preventDefault(); setInput('') }}
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


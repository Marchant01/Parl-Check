import { useState } from 'react';
import { Navbar } from './components/Navbar';
import './App.css'

function App() {
  return (
    <div className='flex flex-col bg-gradient-to-b from-[#69A297] to-[#50808E] min-h-screen'>
      <Navbar />
    </div>
  );
}

export default App
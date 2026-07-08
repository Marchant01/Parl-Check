import { useState } from 'react';

export const Navbar = () => {
    return (
        <nav>
            <div className='flex items-center justify-between px-8 '>
                <span className='text-3xl font-bold'>
                    Parl-Check
                </span>
                <a href='#chat' className='font-bold'>
                    Chat
                </a>
                <a href='#data' className='font-bold'>
                    Data
                </a>
                <a href='#info' className='font-bold'>
                    Info
                </a>
            </div>
        </nav>
    )
}
import { useState } from 'react';

export const Navbar = () => {
    return (
        <nav>
            <div className='flex flex-row justify-between font-mono p-4 bg-white text-black items-center'>
                <span onClick={() => window.location.href = '/'} className='text-7xl font-bold'>
                    Parl-Check
                </span>
                <div className='flex flex-row gap-20 font-bold text-4xl'>
                    <a href='/chat' className='hover:text-[#ff9a00]'>
                        Chat
                    </a>
                    <a href='/data' className='hover:text-[#ff9a00]'>
                        Data
                    </a>
                    <a href='/info' className='hover:text-[#ff9a00]'>
                        Info
                    </a>
                </div>
            </div>
        </nav>
    )
}
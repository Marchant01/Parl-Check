import { Outlet, createRootRoute } from '@tanstack/react-router'
import { Navbar } from '../components/Navbar'

export const Route = createRootRoute({
  component: RootComponent,
})

function RootComponent() {
  return (
    <div className="flex flex-col bg-gradient-to-b from-[#69A297] to-[#50808E] min-h-screen">
      <Navbar />
      <Outlet />
    </div>
  )
}
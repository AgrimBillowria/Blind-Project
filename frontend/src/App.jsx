import { useState } from 'react'
import Launcher from './components/Launcher'
import VideoStream from './components/VideoStream'

export default function App() {
  const [session, setSession] = useState(null)

  return session
    ? <VideoStream config={session} onStop={() => setSession(null)} />
    : <Launcher onStart={setSession} />
}

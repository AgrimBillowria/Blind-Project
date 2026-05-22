import { useState, useEffect, useRef, useCallback } from 'react'
import styles from './VideoStream.module.css'

const WS_URL = 'ws://localhost:8000/ws'
const VOICE_COOLDOWN_MS = 3000  // match Python config VOICE_COOLDOWN

const PRIORITY_COLOR = {
  DANGER:  '#ef4444',
  WARNING: '#f97316',
  STAIR:   '#d946ef',
  CLEAR:   '#10b981',
}

const PRIORITY_LABEL = {
  DANGER:  'DANGER',
  WARNING: 'CAUTION',
  STAIR:   'STAIRS',
  CLEAR:   'CLEAR',
}

export default function VideoStream({ config, onStop }) {
  const [frame, setFrame]           = useState(null)
  const [navMessage, setNavMessage] = useState('Connecting…')
  const [priority, setPriority]     = useState('CLEAR')
  const [objectCount, setObjectCount] = useState(0)
  const [fps, setFps]               = useState(0)
  const [status, setStatus]         = useState('connecting') // connecting | streaming | error | done
  const [errorMsg, setErrorMsg]     = useState('')

  const wsRef           = useRef(null)
  const lastSpokenRef   = useRef({})   // message → timestamp for TTS cooldown
  const frameCountRef   = useRef(0)

  // ── Web Speech API TTS ──
  const speak = useCallback((message, prio) => {
    if (!('speechSynthesis' in window)) return

    const now = Date.now()
    const last = lastSpokenRef.current[message] || 0
    if (now - last < VOICE_COOLDOWN_MS) return

    lastSpokenRef.current[message] = now
    window.speechSynthesis.cancel()
    const utt = new SpeechSynthesisUtterance(message)
    utt.rate   = 1.1
    utt.volume = 1.0
    window.speechSynthesis.speak(utt)
  }, [])

  // ── WebSocket connection ──
  useEffect(() => {
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify(config))
      setStatus('connecting')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'error') {
        setStatus('error')
        setErrorMsg(data.message)
        return
      }

      if (data.type === 'done') {
        setStatus('done')
        return
      }

      if (data.type === 'frame') {
        frameCountRef.current += 1
        setFrame(data.frame)
        setNavMessage(data.nav_message)
        setPriority(data.priority)
        setObjectCount(data.object_count ?? 0)
        setFps(data.fps ?? 0)
        if (status !== 'streaming') setStatus('streaming')

        // Priority-based TTS frequency (mirrors main.py logic)
        const fc = frameCountRef.current
        if (data.priority === 'DANGER' || data.priority === 'STAIR') {
          speak(data.nav_message, data.priority)
        } else if (data.priority === 'WARNING' && fc % 10 === 0) {
          speak(data.nav_message, data.priority)
        } else if (data.priority === 'CLEAR' && fc % 30 === 0) {
          speak(data.nav_message, data.priority)
        }
      }
    }

    ws.onerror = () => {
      setStatus('error')
      setErrorMsg('WebSocket error. Is the server running on port 8000?')
    }

    ws.onclose = () => {
      if (status !== 'error') setStatus('done')
    }

    return () => {
      window.speechSynthesis.cancel()
      ws.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config])

  const handleStop = () => {
    window.speechSynthesis.cancel()
    wsRef.current?.close()
    onStop()
  }

  const accentColor = PRIORITY_COLOR[priority] || PRIORITY_COLOR.CLEAR

  return (
    <div className={styles.page}>

      {/* Top bar */}
      <div className={styles.topBar}>
        <button className={styles.backBtn} onClick={handleStop}>
          <BackIcon /> Back
        </button>
        <span className={styles.appName}>AI Navigation Assistant</span>
        <div className={styles.stats}>
          <StatBadge label="FPS" value={fps} />
          <StatBadge label="Objects" value={objectCount} />
        </div>
      </div>

      {/* Video area */}
      <div className={styles.videoWrap}>
        {status === 'connecting' && !frame && (
          <div className={styles.overlay}>
            <Spinner />
            <p>Opening {config.mode === 'camera' ? 'camera…' : 'video…'}</p>
          </div>
        )}

        {status === 'error' && (
          <div className={styles.overlay}>
            <ErrorIcon />
            <p className={styles.errorText}>{errorMsg}</p>
            <button className={styles.retryBtn} onClick={handleStop}>
              Go Back
            </button>
          </div>
        )}

        {frame && (
          <img
            className={styles.feed}
            src={`data:image/jpeg;base64,${frame}`}
            alt="Navigation feed"
          />
        )}
      </div>

      {/* Navigation status bar */}
      <div
        className={styles.navBar}
        style={{ '--accent-color': accentColor }}
      >
        <span
          className={styles.priorityBadge}
          style={{ background: accentColor }}
        >
          {PRIORITY_LABEL[priority] || priority}
        </span>
        <span className={styles.navMessage}>{navMessage}</span>
        <VoiceIcon />
      </div>

    </div>
  )
}

// ── Sub-components ──

function StatBadge({ label, value }) {
  return (
    <div className={styles.statBadge}>
      <span className={styles.statVal}>{value}</span>
      <span className={styles.statLabel}>{label}</span>
    </div>
  )
}

// ── SVG icons ──
function BackIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="15 18 9 12 15 6"/>
    </svg>
  )
}

function VoiceIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{flexShrink:0, opacity:0.6}}>
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
      <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
    </svg>
  )
}

function Spinner() {
  return (
    <svg className={styles.spinner} width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg>
  )
}

function ErrorIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="1.5">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
  )
}

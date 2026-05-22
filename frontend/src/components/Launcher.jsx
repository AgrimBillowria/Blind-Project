import { useState, useEffect, useRef } from 'react'
import styles from './Launcher.module.css'

const API = 'http://localhost:8000'

export default function Launcher({ onStart }) {
  const [mode, setMode] = useState('camera')
  const [cameras, setCameras] = useState([])
  const [cameraIdx, setCameraIdx] = useState(0)
  const [camsLoading, setCamsLoading] = useState(true)

  const [videoFile, setVideoFile] = useState(null)
  const [uploadState, setUploadState] = useState('idle') // idle | uploading | done | error
  const [videoId, setVideoId] = useState(null)
  const [uploadError, setUploadError] = useState('')

  const fileInputRef = useRef()

  // Fetch available cameras on mount
  useEffect(() => {
    fetch(`${API}/cameras`)
      .then(r => r.json())
      .then(data => {
        setCameras(data.cameras || [])
        if (data.cameras?.length > 0) setCameraIdx(data.cameras[0].index)
      })
      .catch(() => setCameras([]))
      .finally(() => setCamsLoading(false))
  }, [])

  // Re-fetch cameras when switching back to camera mode
  const handleModeChange = (m) => {
    setMode(m)
    if (m === 'camera') {
      setCamsLoading(true)
      fetch(`${API}/cameras`)
        .then(r => r.json())
        .then(data => {
          setCameras(data.cameras || [])
          if (data.cameras?.length > 0) setCameraIdx(data.cameras[0].index)
        })
        .catch(() => setCameras([]))
        .finally(() => setCamsLoading(false))
    }
  }

  const handleFileSelect = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setVideoFile(file)
    setVideoId(null)
    setUploadState('uploading')
    setUploadError('')

    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API}/upload-video`, { method: 'POST', body: form })
      const data = await res.json()
      if (data.error) {
        setUploadState('error')
        setUploadError(data.error)
      } else {
        setVideoId(data.video_id)
        setUploadState('done')
      }
    } catch {
      setUploadState('error')
      setUploadError('Upload failed. Is the server running?')
    }
  }

  const canStart =
    mode === 'camera'
      ? cameras.length > 0
      : uploadState === 'done' && videoId

  const handleStart = () => {
    if (!canStart) return
    if (mode === 'camera') {
      onStart({ mode: 'camera', camera_index: cameraIdx })
    } else {
      onStart({ mode: 'video', video_id: videoId })
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.card}>

        {/* Header */}
        <div className={styles.header}>
          <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
            <circle cx="12" cy="12" r="3"/>
          </svg>
          <div>
            <h1 className={styles.title}>AI Navigation Assistant</h1>
            <p className={styles.subtitle}>For Visually Impaired People</p>
          </div>
        </div>

        {/* Mode tabs */}
        <div className={styles.tabs}>
          <button
            className={`${styles.tab} ${mode === 'camera' ? styles.tabActive : ''}`}
            onClick={() => handleModeChange('camera')}
          >
            <CameraIcon /> Camera
          </button>
          <button
            className={`${styles.tab} ${mode === 'video' ? styles.tabActive : ''}`}
            onClick={() => handleModeChange('video')}
          >
            <VideoIcon /> Video
          </button>
        </div>

        {/* Camera panel */}
        {mode === 'camera' && (
          <div className={styles.panel}>
            <label className={styles.label}>Select Camera</label>
            {camsLoading ? (
              <p className={styles.hint}>Detecting cameras…</p>
            ) : cameras.length === 0 ? (
              <p className={styles.errorText}>No cameras found. Check connections and try again.</p>
            ) : (
              <select
                className={styles.select}
                value={cameraIdx}
                onChange={e => setCameraIdx(Number(e.target.value))}
              >
                {cameras.map(c => (
                  <option key={c.index} value={c.index}>{c.label}</option>
                ))}
              </select>
            )}
          </div>
        )}

        {/* Video panel */}
        {mode === 'video' && (
          <div className={styles.panel}>
            <label className={styles.label}>Select Video File</label>
            <div
              className={`${styles.dropzone} ${videoFile ? styles.dropzoneHasFile : ''}`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.wmv"
                onChange={handleFileSelect}
              />
              {uploadState === 'idle' && (
                <>
                  <UploadIcon />
                  <span>Click to browse or drop a video</span>
                  <span className={styles.hint}>MP4, AVI, MOV, MKV, WMV</span>
                </>
              )}
              {uploadState === 'uploading' && (
                <>
                  <Spinner />
                  <span>Uploading {videoFile?.name}…</span>
                </>
              )}
              {uploadState === 'done' && (
                <>
                  <CheckIcon />
                  <span className={styles.fileName}>{videoFile?.name}</span>
                  <span className={styles.hint}>Click to change</span>
                </>
              )}
              {uploadState === 'error' && (
                <>
                  <ErrorIcon />
                  <span className={styles.errorText}>{uploadError}</span>
                  <span className={styles.hint}>Click to retry</span>
                </>
              )}
            </div>
          </div>
        )}

        {/* Start button */}
        <button
          className={styles.startBtn}
          disabled={!canStart}
          onClick={handleStart}
        >
          Start
        </button>

      </div>
    </div>
  )
}

// ── Inline SVG icons ──
function CameraIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
      <circle cx="12" cy="13" r="4"/>
    </svg>
  )
}

function VideoIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="23 7 16 12 23 17 23 7"/>
      <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
      <polyline points="22 4 12 14.01 9 11.01"/>
    </svg>
  )
}

function ErrorIcon() {
  return (
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ef4444" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
  )
}

function Spinner() {
  return (
    <svg className={styles.spinner} width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2">
      <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
    </svg>
  )
}

import { useState, useRef } from 'react'
import { categoryColor } from '../colors.js'

const API = ''

const CATEGORY_ICON = {
  'Tropical Depression':             '🌱',
  'Tropical Storm':                  '🌧️',
  'Severe Cyclonic Storm':           '🌀',
  'Very Severe Cyclonic Storm':      '🌀',
  'Extremely Severe Cyclonic Storm': '🌀',
  'Super Cyclonic Storm':            '🚨',
}

export default function Predict() {
  const [file,        setFile]        = useState(null)
  const [preview,     setPreview]     = useState(null)
  const [result,      setResult]      = useState(null)
  const [loading,     setLoading]     = useState(false)
  const [error,       setError]       = useState(null)
  const [mockFrames,  setMockFrames]  = useState(null)
  const [mockResults, setMockResults] = useState({})
  const [dragging,    setDragging]    = useState(false)
  const inputRef = useRef()

  function handleFile(f) {
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  async function runPredict() {
    if (!file) return
    setLoading(true)
    setError(null)
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch(`${API}/api/predict`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text())
      setResult(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadMockFrames() {
    const res = await fetch(`${API}/api/mock-frames`)
    const data = await res.json()
    setMockFrames(data.frames)
  }

  async function predictMock(filename) {
    const res  = await fetch(`${API}/api/mock-frames/${filename}/predict`)
    const data = await res.json()
    setMockResults(prev => ({ ...prev, [filename]: data }))
  }

  return (
    <div style={{ maxWidth: 860, margin: '0 auto' }}>
      <h2 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 4, color: 'var(--text)' }}>
        AI Intensity Prediction
      </h2>
      <p style={{ color: 'var(--muted)', fontSize: '0.825rem', marginBottom: 22, lineHeight: 1.6 }}>
        Upload a satellite image of a cyclone — the ResNet-18 model estimates wind speed and intensity category.
      </p>

      <div className="grid-2" style={{ marginBottom: 18 }}>
        {/* ── Upload zone ── */}
        <div
          className="card"
          onDrop={onDrop}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onClick={() => inputRef.current?.click()}
          style={{
            border: `2px dashed ${dragging ? 'var(--accent)' : 'var(--border)'}`,
            cursor: 'pointer',
            textAlign: 'center',
            transition: 'border-color .2s, background .2s',
            background: dragging ? 'var(--accent-glow)' : 'var(--surface)',
            minHeight: 200,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={e => e.target.files[0] && handleFile(e.target.files[0])}
          />
          {preview ? (
            <img
              src={preview}
              alt="preview"
              style={{ maxWidth: '100%', maxHeight: 220, borderRadius: 6, objectFit: 'contain' }}
            />
          ) : (
            <div style={{ color: 'var(--muted)', padding: '32px 20px' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: 10, opacity: 0.7 }}>🛰️</div>
              <div style={{ fontSize: '0.85rem' }}>Drag &amp; drop or click to upload</div>
              <div style={{ fontSize: '0.72rem', marginTop: 6, color: 'var(--muted)' }}>
                PNG · JPG · JPEG
              </div>
            </div>
          )}
        </div>

        {/* ── Result panel ── */}
        <div
          className="card"
          style={{
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: 200,
            borderTop: result ? `2px solid ${categoryColor(result.category)}` : '1px solid var(--border)',
            transition: 'border-color .3s',
          }}
        >
          {result ? (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2.8rem', marginBottom: 8 }}>
                {CATEGORY_ICON[result.category] || '🌀'}
              </div>
              <div style={{ fontSize: '2.6rem', fontWeight: 800, color: categoryColor(result.category), lineHeight: 1 }}>
                {result.wind_speed_knots}
                <span style={{ fontSize: '0.95rem', color: 'var(--muted)', marginLeft: 6, fontWeight: 400 }}>
                  knots
                </span>
              </div>
              <span
                className="badge"
                style={{ background: categoryColor(result.category), color: '#000', marginTop: 12, fontSize: '0.78rem' }}
              >
                {result.category}
              </span>
              <div style={{ marginTop: 14, color: 'var(--muted)', fontSize: '0.72rem' }}>
                {result.filename}
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--muted)', textAlign: 'center' }}>
              <div style={{ fontSize: '1.8rem', marginBottom: 8, opacity: 0.4 }}>📡</div>
              <div style={{ fontSize: '0.82rem' }}>Result will appear here</div>
            </div>
          )}
        </div>
      </div>

      {error && <div className="error-msg">⚠ {error}</div>}

      <button
        className="btn-primary"
        onClick={runPredict}
        disabled={!file || loading}
        style={{ marginBottom: 32 }}
      >
        {loading ? '⏳  Running model…' : '🚀  Run Prediction'}
      </button>

      {/* ── Mock frames ── */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div className="section-heading" style={{ margin: 0 }}>Bundled Test Frames</div>
          <button className="btn-ghost" onClick={loadMockFrames}>Load Frames</button>
        </div>

        {mockFrames === null && (
          <p style={{ color: 'var(--muted)', fontSize: '0.825rem' }}>
            Click "Load Frames" to run predictions on the included mock satellite images.
          </p>
        )}
        {mockFrames?.length === 0 && (
          <p style={{ color: 'var(--muted)', fontSize: '0.825rem' }}>
            No frames found in mock_cyclone_frames/.
          </p>
        )}

        {mockFrames && mockFrames.length > 0 && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(148px, 1fr))',
            gap: 10,
          }}>
            {mockFrames.map(f => {
              const res = mockResults[f.filename]
              return (
                <div
                  key={f.filename}
                  style={{
                    background: 'var(--surface2)',
                    border: `1px solid ${res ? categoryColor(res.category) : 'var(--border)'}`,
                    borderRadius: 8,
                    padding: '12px 10px',
                    textAlign: 'center',
                    transition: 'border-color .2s',
                  }}
                >
                  <div style={{ fontSize: '1.6rem', marginBottom: 6 }}>🖼️</div>
                  <div style={{
                    fontSize: '0.72rem',
                    color: 'var(--muted)',
                    marginBottom: 8,
                    wordBreak: 'break-all',
                  }}>
                    {f.filename}
                  </div>
                  {res ? (
                    <>
                      <div style={{ fontWeight: 700, color: categoryColor(res.category), fontSize: '1.1rem' }}>
                        {res.wind_speed_knots} kts
                      </div>
                      <span
                        className="badge"
                        style={{ background: categoryColor(res.category), color: '#000', marginTop: 6, fontSize: '0.6rem' }}
                      >
                        {res.category}
                      </span>
                    </>
                  ) : (
                    <button
                      className="btn-ghost"
                      style={{ width: '100%', marginTop: 2 }}
                      onClick={() => predictMock(f.filename)}
                    >
                      Predict
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

import { useState, useRef } from 'react'

const API = ''

const CATEGORY_ICONS = {
  'Tropical Depression': '🌱',
  'Tropical Storm': '🌧️',
  'Severe Cyclonic Storm': '🌀',
  'Very Severe Cyclonic Storm': '🌀',
  'Extremely Severe Cyclonic Storm': '🌀',
  'Super Cyclonic Storm': '🚨',
}

export default function Predict() {
  const [file, setFile]         = useState(null)
  const [preview, setPreview]   = useState(null)
  const [result, setResult]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [mockFrames, setMockFrames] = useState(null)
  const [mockResults, setMockResults] = useState({})
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
      const data = await res.json()
      setResult(data)
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
    const res = await fetch(`${API}/api/mock-frames/${filename}/predict`)
    const data = await res.json()
    setMockResults(prev => ({ ...prev, [filename]: data }))
  }

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <h2 style={{ fontSize: '1.1rem', marginBottom: 4 }}>AI Intensity Prediction</h2>
      <p style={{ color: '#9ca3af', fontSize: '0.875rem', marginBottom: 24 }}>
        Upload a satellite image of a cyclone and the ResNet-18 model will estimate wind speed and intensity category.
      </p>

      <div className="grid-2" style={{ marginBottom: 24 }}>
        {/* Upload */}
        <div
          className="card"
          onDrop={onDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => inputRef.current?.click()}
          style={{ border: '2px dashed #374151', cursor: 'pointer', textAlign: 'center', transition: 'border-color .2s' }}
          onMouseEnter={e => e.currentTarget.style.borderColor = '#3b82f6'}
          onMouseLeave={e => e.currentTarget.style.borderColor = '#374151'}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={e => e.target.files[0] && handleFile(e.target.files[0])}
          />
          {preview ? (
            <img src={preview} alt="preview" style={{ maxWidth: '100%', maxHeight: 220, borderRadius: 8, objectFit: 'contain' }} />
          ) : (
            <div style={{ padding: '40px 20px', color: '#9ca3af' }}>
              <div style={{ fontSize: '3rem', marginBottom: 12 }}>🛰️</div>
              <div>Drag & drop or click to upload a satellite image</div>
              <div style={{ fontSize: '0.75rem', marginTop: 8 }}>PNG, JPG, JPEG supported</div>
            </div>
          )}
        </div>

        {/* Result */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
          {result ? (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: 8 }}>
                {CATEGORY_ICONS[result.category] || '🌀'}
              </div>
              <div style={{ fontSize: '2.5rem', fontWeight: 700, color: result.color }}>
                {result.wind_speed_knots} <span style={{ fontSize: '1rem', color: '#9ca3af' }}>knots</span>
              </div>
              <span className="badge" style={{ background: result.color, marginTop: 8, fontSize: '0.85rem' }}>
                {result.category}
              </span>
              <div style={{ marginTop: 16, color: '#9ca3af', fontSize: '0.8rem' }}>{result.filename}</div>
            </div>
          ) : (
            <div style={{ color: '#6b7280', textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', marginBottom: 8 }}>📡</div>
              <div>Prediction result will appear here</div>
            </div>
          )}
        </div>
      </div>

      {error && <div className="error-msg" style={{ marginBottom: 16 }}>⚠ {error}</div>}

      <button
        onClick={runPredict}
        disabled={!file || loading}
        style={{
          width: '100%', padding: '14px', borderRadius: 10,
          background: file && !loading ? '#2563eb' : '#374151',
          border: 'none', color: '#fff', fontWeight: 600,
          fontSize: '1rem', cursor: file ? 'pointer' : 'not-allowed',
          marginBottom: 32, transition: 'background .2s',
        }}
      >
        {loading ? '⏳ Running model…' : '🚀 Run Prediction'}
      </button>

      {/* Mock Frames Section */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ fontSize: '0.95rem' }}>Mock Satellite Frames</h3>
          <button
            onClick={loadMockFrames}
            style={{ padding: '6px 14px', borderRadius: 6, border: '1px solid #374151', background: 'transparent', color: '#9ca3af', cursor: 'pointer', fontSize: '0.8rem' }}
          >
            Load Frames
          </button>
        </div>
        {mockFrames === null && (
          <p style={{ color: '#6b7280', fontSize: '0.875rem' }}>Click "Load Frames" to see bundled test images.</p>
        )}
        {mockFrames?.length === 0 && (
          <p style={{ color: '#6b7280', fontSize: '0.875rem' }}>No mock frames found in mock_cyclone_frames/ directory.</p>
        )}
        {mockFrames && mockFrames.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px,1fr))', gap: 12 }}>
            {mockFrames.map(f => {
              const res = mockResults[f.filename]
              return (
                <div key={f.filename} style={{ background: '#0a0f1e', border: '1px solid #374151', borderRadius: 8, padding: 12, textAlign: 'center' }}>
                  <div style={{ fontSize: '2rem', marginBottom: 8 }}>🖼️</div>
                  <div style={{ fontSize: '0.78rem', color: '#9ca3af', marginBottom: 8, wordBreak: 'break-all' }}>{f.filename}</div>
                  {res ? (
                    <>
                      <div style={{ fontWeight: 700, color: res.color }}>{res.wind_speed_knots} kts</div>
                      <span className="badge" style={{ background: res.color, marginTop: 4, fontSize: '0.65rem' }}>{res.category}</span>
                    </>
                  ) : (
                    <button
                      onClick={() => predictMock(f.filename)}
                      style={{ padding: '5px 12px', borderRadius: 6, border: '1px solid #3b82f6', background: 'transparent', color: '#60a5fa', cursor: 'pointer', fontSize: '0.75rem' }}
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

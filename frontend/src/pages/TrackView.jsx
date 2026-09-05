import { useState, useEffect } from 'react'
import {
  MapContainer, TileLayer, Polyline, CircleMarker,
  Tooltip as MapTooltip,
} from 'react-leaflet'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { categoryColor } from '../colors.js'

const API = ''

function markerRadius(wind) {
  if (!wind)      return 5
  if (wind < 34)  return 5
  if (wind < 64)  return 7
  if (wind < 96)  return 9
  return 12
}

const ChartTooltipStyle = {
  background: 'var(--surface2)',
  border: '1px solid var(--border)',
  borderRadius: 8,
  fontSize: '0.78rem',
}

export default function TrackView() {
  const [cyclones,     setCyclones]     = useState([])
  const [selected,     setSelected]     = useState(null)
  const [track,        setTrack]        = useState(null)
  const [loading,      setLoading]      = useState(true)
  const [trackLoading, setTrackLoading] = useState(false)
  const [error,        setError]        = useState(null)

  useEffect(() => {
    fetch(`${API}/api/cyclones`)
      .then(r => r.json())
      .then(d => setCyclones(d.cyclones))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  function loadTrack(cyclone) {
    setSelected(cyclone)
    setTrackLoading(true)
    fetch(`${API}/api/cyclones/${cyclone.sid}/track`)
      .then(r => r.json())
      .then(d => setTrack(d))
      .catch(e => setError(e.message))
      .finally(() => setTrackLoading(false))
  }

  if (loading) return <div className="loading">Loading storm list…</div>
  if (error)   return <div className="error-msg">⚠ {error}</div>

  const trackPoints  = track?.track ?? []
  const validPoints  = trackPoints.filter(p => p.lat && p.lon)
  const midIdx       = Math.floor(validPoints.length / 2)
  const center       = validPoints.length ? [validPoints[midIdx].lat, validPoints[midIdx].lon] : [15, 80]

  const windChart = trackPoints
    .filter(p => p.wind_knots)
    .map(p => ({ time: p.time?.slice(5, 16), wind: p.wind_knots }))

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '260px 1fr',
      gap: 14,
      height: 'calc(100vh - 108px)',
      minHeight: 520,
    }}>
      {/* ── Sidebar ── */}
      <div className="card" style={{ overflowY: 'auto', padding: '12px 8px' }}>
        <div className="section-heading" style={{ paddingLeft: 8 }}>Storms</div>
        {cyclones.map(c => {
          const isActive = selected?.sid === c.sid
          return (
            <button
              key={c.sid}
              onClick={() => loadTrack(c)}
              style={{
                display: 'block',
                width: '100%',
                textAlign: 'left',
                padding: '9px 10px',
                background: isActive ? 'var(--accent-glow)' : 'transparent',
                border: `1px solid ${isActive ? 'var(--accent-dim)' : 'transparent'}`,
                borderRadius: 7,
                cursor: 'pointer',
                marginBottom: 3,
                color: isActive ? 'var(--accent)' : 'var(--text)',
                transition: 'all .12s',
              }}
              onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = 'var(--surface2)' }}
              onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
            >
              <div style={{ fontWeight: 600, fontSize: '0.83rem' }}>{c.name}</div>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: 4,
              }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>
                  {c.start_time?.slice(0, 10)}
                </span>
                <span
                  className="badge"
                  style={{ background: categoryColor(c.category), color: '#000', fontSize: '0.62rem' }}
                >
                  {c.max_wind_knots ?? '?'} kts
                </span>
              </div>
            </button>
          )
        })}
      </div>

      {/* ── Map + chart column ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14, minWidth: 0 }}>
        {/* Map */}
        <div className="card" style={{ flex: 1, padding: 0, overflow: 'hidden' }}>
          {trackLoading ? (
            <div className="loading">Loading track…</div>
          ) : (
            <MapContainer
              key={selected?.sid || 'default'}
              center={center}
              zoom={validPoints.length ? 5 : 4}
              style={{ height: '100%', minHeight: 320 }}
              scrollWheelZoom
            >
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png"
                attribution='&copy; <a href="https://carto.com/">CARTO</a>'
              />
              {/* Labels layer on top */}
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png"
                pane="overlayPane"
              />

              {validPoints.length > 1 && (
                <Polyline
                  positions={validPoints.map(p => [p.lat, p.lon])}
                  color="#0ea5e9"
                  weight={1.5}
                  opacity={0.5}
                  dashArray="4 3"
                />
              )}

              {validPoints.map((p, i) => (
                <CircleMarker
                  key={i}
                  center={[p.lat, p.lon]}
                  radius={markerRadius(p.wind_knots)}
                  fillColor={categoryColor(p.category)}
                  color="rgba(0,0,0,0.5)"
                  weight={1}
                  fillOpacity={0.9}
                >
                  <MapTooltip>
                    <div style={{ fontSize: '0.78rem', lineHeight: 1.5 }}>
                      <strong>{p.time?.slice(0, 16)}</strong><br />
                      {p.wind_knots ?? '—'} kts &nbsp;·&nbsp; {p.category}<br />
                      {p.lat.toFixed(2)}°N, {p.lon.toFixed(2)}°E
                    </div>
                  </MapTooltip>
                </CircleMarker>
              ))}

              {!selected && (
                <div style={{
                  position: 'absolute', top: '50%', left: '50%',
                  transform: 'translate(-50%,-50%)',
                  background: 'rgba(7,9,15,.75)',
                  border: '1px solid var(--border)',
                  padding: '10px 18px', borderRadius: 8,
                  color: 'var(--muted)', zIndex: 999, pointerEvents: 'none',
                  fontSize: '0.85rem',
                }}>
                  ← Select a storm to view its track
                </div>
              )}
            </MapContainer>
          )}
        </div>

        {/* Wind chart */}
        {windChart.length > 0 && (
          <div className="card">
            <div className="section-heading">
              Wind Speed — {track?.name}
              {track?.max_wind_knots && (
                <span style={{ color: 'var(--s4)', marginLeft: 8, fontWeight: 700 }}>
                  peak {track.max_wind_knots} kts
                </span>
              )}
            </div>
            <ResponsiveContainer width="100%" height={110}>
              <LineChart data={windChart} margin={{ top: 4, right: 8, left: -22, bottom: 0 }}>
                <XAxis
                  dataKey="time"
                  tick={{ fill: 'var(--muted)', fontSize: 9 }}
                  interval="preserveStartEnd"
                />
                <YAxis tick={{ fill: 'var(--muted)', fontSize: 9 }} />
                <Tooltip
                  contentStyle={ChartTooltipStyle}
                  formatter={v => [`${v} knots`, 'Wind']}
                />
                <ReferenceLine
                  y={64}
                  stroke="var(--s3)"
                  strokeDasharray="3 2"
                  label={{ value: 'Severe', fill: 'var(--s3)', fontSize: 9 }}
                />
                <Line
                  type="monotone"
                  dataKey="wind"
                  stroke="var(--accent)"
                  dot={false}
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

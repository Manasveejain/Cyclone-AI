import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Polyline, CircleMarker, Tooltip as MapTooltip } from 'react-leaflet'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const API = ''

// Map intensity color to marker size
function markerSize(wind) {
  if (!wind) return 5
  if (wind < 34)  return 5
  if (wind < 64)  return 7
  if (wind < 96)  return 9
  return 11
}

export default function TrackView() {
  const [cyclones, setCyclones] = useState([])
  const [selected, setSelected] = useState(null)
  const [track, setTrack]       = useState(null)
  const [loading, setLoading]   = useState(true)
  const [trackLoading, setTrackLoading] = useState(false)
  const [error, setError]       = useState(null)

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

  const trackPoints = track?.track ?? []
  const validPoints = trackPoints.filter(p => p.lat && p.lon)
  const center = validPoints.length
    ? [validPoints[Math.floor(validPoints.length / 2)].lat, validPoints[Math.floor(validPoints.length / 2)].lon]
    : [15, 80]

  const windChartData = trackPoints
    .filter(p => p.wind_knots)
    .map(p => ({ time: p.time?.slice(5, 16), wind: p.wind_knots }))

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 16, height: 'calc(100vh - 140px)', minHeight: 500 }}>
      {/* Storm list sidebar */}
      <div className="card" style={{ overflowY: 'auto', padding: 12 }}>
        <h3 style={{ fontSize: '0.85rem', color: '#9ca3af', marginBottom: 12, textTransform: 'uppercase' }}>Select Storm</h3>
        {cyclones.map(c => (
          <button
            key={c.sid}
            onClick={() => loadTrack(c)}
            style={{
              width: '100%', textAlign: 'left', padding: '10px 12px',
              background: selected?.sid === c.sid ? '#1d4ed8' : 'transparent',
              border: '1px solid ' + (selected?.sid === c.sid ? '#3b82f6' : '#1f2937'),
              borderRadius: 8, cursor: 'pointer', marginBottom: 6,
              color: selected?.sid === c.sid ? '#fff' : '#f9fafb',
              transition: 'all .15s',
            }}
          >
            <div style={{ fontWeight: 600, fontSize: '0.875rem' }}>{c.name}</div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
              <span style={{ fontSize: '0.72rem', color: selected?.sid === c.sid ? '#bfdbfe' : '#9ca3af' }}>
                {c.start_time?.slice(0, 10)}
              </span>
              <span className="badge" style={{ background: c.color }}>{c.max_wind_knots ?? '?'} kts</span>
            </div>
          </button>
        ))}
      </div>

      {/* Map + chart */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* Map */}
        <div className="card" style={{ flex: 1, padding: 0, overflow: 'hidden', borderRadius: 12 }}>
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
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                attribution='&copy; <a href="https://carto.com/">CARTO</a>'
              />
              {validPoints.length > 1 && (
                <Polyline
                  positions={validPoints.map(p => [p.lat, p.lon])}
                  color="#3b82f6"
                  weight={2}
                  opacity={0.7}
                />
              )}
              {validPoints.map((p, i) => (
                <CircleMarker
                  key={i}
                  center={[p.lat, p.lon]}
                  radius={markerSize(p.wind_knots)}
                  fillColor={p.color}
                  color="#fff"
                  weight={1}
                  fillOpacity={0.85}
                >
                  <MapTooltip>
                    <div style={{ fontSize: '0.8rem' }}>
                      <strong>{p.time?.slice(0, 16)}</strong><br />
                      Wind: {p.wind_knots ?? '—'} kts<br />
                      {p.category}<br />
                      {p.lat.toFixed(1)}°N, {p.lon.toFixed(1)}°E
                    </div>
                  </MapTooltip>
                </CircleMarker>
              ))}
              {!selected && (
                <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%,-50%)', background: 'rgba(0,0,0,.6)', padding: '12px 20px', borderRadius: 8, color: '#9ca3af', zIndex: 999, pointerEvents: 'none' }}>
                  ← Select a storm to view its track
                </div>
              )}
            </MapContainer>
          )}
        </div>

        {/* Wind speed chart */}
        {windChartData.length > 0 && (
          <div className="card">
            <h3 style={{ fontSize: '0.85rem', color: '#9ca3af', marginBottom: 12 }}>
              Wind Speed Over Time — {track?.name}
            </h3>
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={windChartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <XAxis dataKey="time" tick={{ fill: '#9ca3af', fontSize: 10 }} interval="preserveStartEnd" />
                <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                  formatter={v => [`${v} knots`, 'Wind']}
                />
                <ReferenceLine y={64} stroke="#f87171" strokeDasharray="4 2" label={{ value: 'Severe', fill: '#f87171', fontSize: 10 }} />
                <Line type="monotone" dataKey="wind" stroke="#06b6d4" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

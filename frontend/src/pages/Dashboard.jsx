import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API = ''  // proxied via vite

export default function Dashboard() {
  const [stats, setStats]     = useState(null)
  const [cyclones, setCyclones] = useState([])
  const [loading, setLoading]  = useState(true)
  const [error, setError]      = useState(null)

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/stats`).then(r => r.json()),
      fetch(`${API}/api/cyclones`).then(r => r.json()),
    ])
      .then(([s, c]) => { setStats(s); setCyclones(c.cyclones) })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading">Loading cyclone data…</div>
  if (error)   return <div className="error-msg">⚠ {error} — make sure the backend is running on port 8000.</div>

  const distData = Object.entries(stats.intensity_distribution || {}).map(([k, v]) => ({ name: k, value: v }))
  const COLORS = ['#4ade80','#facc15','#fb923c','#f87171','#c084fc','#e11d48']

  // Wind distribution per storm for bar chart
  const barData = cyclones
    .filter(c => c.max_wind_knots)
    .sort((a, b) => b.max_wind_knots - a.max_wind_knots)
    .slice(0, 15)
    .map(c => ({ name: c.name === 'UNNAMED' ? c.sid.slice(0, 10) : c.name, wind: c.max_wind_knots }))

  return (
    <div>
      {/* Stat cards */}
      <div className="grid-4">
        <div className="card">
          <div className="card-title">Total Storms</div>
          <div className="card-value" style={{ color: '#06b6d4' }}>{stats.total_storms}</div>
        </div>
        <div className="card">
          <div className="card-title">Named Storms</div>
          <div className="card-value" style={{ color: '#818cf8' }}>{stats.named_storms}</div>
        </div>
        <div className="card">
          <div className="card-title">Peak Wind Speed</div>
          <div className="card-value" style={{ color: '#f87171' }}>{stats.max_recorded_wind_knots} <span style={{ fontSize: '1rem', color: '#9ca3af' }}>kts</span></div>
        </div>
        <div className="card">
          <div className="card-title">Basin</div>
          <div style={{ fontSize: '1rem', fontWeight: 600, marginTop: 8, color: '#34d399' }}>{stats.basin}</div>
        </div>
      </div>

      <div className="grid-2">
        {/* Bar chart */}
        <div className="card">
          <h3 style={{ marginBottom: 16, fontSize: '0.95rem' }}>Top 15 Storms by Max Wind Speed (knots)</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} margin={{ top: 4, right: 8, left: -10, bottom: 60 }}>
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} angle={-40} textAnchor="end" interval={0} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#f9fafb' }}
                formatter={v => [`${v} knots`, 'Max Wind']}
              />
              <Bar dataKey="wind" fill="#3b82f6" radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie chart */}
        <div className="card">
          <h3 style={{ marginBottom: 16, fontSize: '0.95rem' }}>Storm Intensity Distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={distData} dataKey="value" nameKey="name" cx="50%" cy="45%" outerRadius={90} label={({ name, percent }) => `${(percent * 100).toFixed(0)}%`} labelLine={false}>
                {distData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Legend formatter={v => <span style={{ color: '#9ca3af', fontSize: 12 }}>{v}</span>} />
              <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Storm table */}
      <div className="card" style={{ marginTop: 20 }}>
        <h3 style={{ marginBottom: 16, fontSize: '0.95rem' }}>All Storms</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #374151' }}>
                {['Name', 'SID', 'Max Wind (kts)', 'Category', 'Start', 'Records'].map(h => (
                  <th key={h} style={{ padding: '8px 12px', textAlign: 'left', color: '#9ca3af', fontWeight: 500 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cyclones.map(c => (
                <tr key={c.sid} style={{ borderBottom: '1px solid #1f2937' }}>
                  <td style={{ padding: '8px 12px', fontWeight: 600 }}>{c.name}</td>
                  <td style={{ padding: '8px 12px', color: '#9ca3af', fontSize: '0.78rem' }}>{c.sid}</td>
                  <td style={{ padding: '8px 12px' }}>{c.max_wind_knots ?? '—'}</td>
                  <td style={{ padding: '8px 12px' }}>
                    <span className="badge" style={{ background: c.color }}>{c.category}</span>
                  </td>
                  <td style={{ padding: '8px 12px', color: '#9ca3af' }}>{c.start_time?.slice(0, 10)}</td>
                  <td style={{ padding: '8px 12px', color: '#9ca3af' }}>{c.record_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

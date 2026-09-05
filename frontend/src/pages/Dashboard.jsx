import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import { categoryColor, INTENSITY_COLORS_ORDERED } from '../colors.js'

const API = ''

const TT = {
  background: 'var(--surface2)',
  border: '1px solid var(--border)',
  borderRadius: 8,
  fontSize: '0.8rem',
}

export default function Dashboard() {
  const [stats,    setStats]    = useState(null)
  const [cyclones, setCyclones] = useState([])
  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)

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

  // Pie chart: sort by intensity order so colors line up with INTENSITY_COLORS_ORDERED
  const distData = Object.entries(stats.intensity_distribution || {})
    .map(([name, value]) => ({ name, value, color: categoryColor(name) }))
    .sort((a, b) => {
      const ORDER = [
        'Tropical Depression', 'Tropical Storm', 'Severe Cyclonic Storm',
        'Very Severe Cyclonic Storm', 'Extremely Severe Cyclonic Storm',
        'Super Cyclonic Storm', 'Unclassified',
      ]
      return ORDER.indexOf(a.name) - ORDER.indexOf(b.name)
    })

  const barData = cyclones
    .filter(c => c.max_wind_knots)
    .sort((a, b) => b.max_wind_knots - a.max_wind_knots)
    .slice(0, 15)
    .map(c => ({
      name:  c.name === 'UNNAMED' ? c.sid.slice(0, 10) : c.name,
      wind:  c.max_wind_knots,
      color: categoryColor(c.category),
    }))

  return (
    <div>
      {/* ── Stat cards ── */}
      <div className="grid-4">
        <StatCard title="Total Storms"  value={stats.total_storms}                color="var(--accent)" />
        <StatCard title="Named Storms"  value={stats.named_storms}                color="#86efac" />
        <StatCard title="Peak Wind"     value={stats.max_recorded_wind_knots}     color="#f87171" unit="kts" />
        <div className="card">
          <div className="card-title">Basin</div>
          <div style={{ fontSize: '0.88rem', fontWeight: 600, marginTop: 10, color: 'var(--muted2)' }}>
            {stats.basin}
          </div>
        </div>
      </div>

      {/* ── Charts ── */}
      <div className="grid-2" style={{ marginBottom: 20 }}>
        {/* Bar */}
        <div className="card">
          <div className="section-heading">Top 15 by Max Wind Speed</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData} margin={{ top: 4, right: 8, left: -14, bottom: 56 }}>
              <XAxis
                dataKey="name"
                tick={{ fill: 'var(--muted)', fontSize: 10 }}
                angle={-38}
                textAnchor="end"
                interval={0}
              />
              <YAxis tick={{ fill: 'var(--muted)', fontSize: 10 }} />
              <Tooltip
                contentStyle={TT}
                labelStyle={{ color: 'var(--text)', fontWeight: 600 }}
                formatter={v => [`${v} knots`, 'Max Wind']}
              />
              {/* Each bar gets its own intensity color */}
              <Bar dataKey="wind" radius={[4, 4, 0, 0]}>
                {barData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie */}
        <div className="card">
          <div className="section-heading">Intensity Distribution</div>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={distData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="46%"
                outerRadius={88}
                label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                labelLine={false}
              >
                {distData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Legend
                formatter={(value, entry) => (
                  <span style={{ color: entry.color, fontSize: 11 }}>{value}</span>
                )}
              />
              <Tooltip contentStyle={TT} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Storm table ── */}
      <div className="card">
        <div className="section-heading">All Storms</div>
        <div style={{ overflowX: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                {['Name', 'SID', 'Max Wind (kts)', 'Category', 'Start', 'Records'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cyclones.map(c => {
                const color = categoryColor(c.category)
                return (
                  <tr key={c.sid}>
                    <td style={{ fontWeight: 600 }}>{c.name}</td>
                    <td style={{ color: 'var(--muted)', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                      {c.sid}
                    </td>
                    <td>{c.max_wind_knots ?? '—'}</td>
                    <td>
                      <span className="badge" style={{ background: color, color: '#000' }}>
                        {c.category}
                      </span>
                    </td>
                    <td style={{ color: 'var(--muted)' }}>{c.start_time?.slice(0, 10)}</td>
                    <td style={{ color: 'var(--muted)' }}>{c.record_count}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StatCard({ title, value, color, unit }) {
  return (
    <div className="card" style={{ borderTop: `2px solid ${color}` }}>
      <div className="card-title">{title}</div>
      <div className="card-value" style={{ color }}>
        {value}
        {unit && <span style={{ fontSize: '0.9rem', color: 'var(--muted)', marginLeft: 6 }}>{unit}</span>}
      </div>
    </div>
  )
}

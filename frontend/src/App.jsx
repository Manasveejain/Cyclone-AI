import { useState, useEffect } from 'react'
import Dashboard from './pages/Dashboard.jsx'
import TrackView from './pages/TrackView.jsx'
import Predict from './pages/Predict.jsx'
import './App.css'

const TABS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'tracks',    label: 'Cyclone Tracks' },
  { id: 'predict',   label: 'AI Predict' },
]

export default function App() {
  const [tab, setTab] = useState('dashboard')

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">🌀</span>
          <div>
            <h1>Cyclone AI</h1>
            <p>North Indian Ocean — IBTrACS + Deep Learning</p>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map(t => (
            <button
              key={t.id}
              className={`tab-btn ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="app-main">
        {tab === 'dashboard' && <Dashboard />}
        {tab === 'tracks'    && <TrackView />}
        {tab === 'predict'   && <Predict />}
      </main>
    </div>
  )
}

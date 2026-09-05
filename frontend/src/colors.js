/**
 * Single source of truth for intensity colors.
 * Matches the category names returned by the backend classify_intensity().
 * Used by Dashboard (pie, table), TrackView (map markers, chart), and Predict.
 */
export const INTENSITY = {
  'Tropical Depression':             { color: '#22d3ee', order: 0 },
  'Tropical Storm':                  { color: '#86efac', order: 1 },
  'Severe Cyclonic Storm':           { color: '#fde047', order: 2 },
  'Very Severe Cyclonic Storm':      { color: '#fb923c', order: 3 },
  'Extremely Severe Cyclonic Storm': { color: '#f87171', order: 4 },
  'Super Cyclonic Storm':            { color: '#e879f9', order: 5 },
  'Unclassified':                    { color: '#6b7a99', order: 6 },
}

/** Returns the hex color for a category name (falls back to grey). */
export function categoryColor(name) {
  return INTENSITY[name]?.color ?? '#6b7a99'
}

/** Ordered array of hex colors — used for Recharts Pie cells. */
export const INTENSITY_COLORS_ORDERED = Object.values(INTENSITY)
  .sort((a, b) => a.order - b.order)
  .map(v => v.color)

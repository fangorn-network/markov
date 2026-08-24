/**
 * vec.ts — minimal Float32Array vector math
 * All operations are pure (no mutation) unless suffixed with `_`.
 * d = 384 (all-MiniLM-L6-v2, L2 space)
 */
export type Vec = Float32Array

export const zeros = (d: number): Vec => new Float32Array(d)
export const clone = (a: Vec): Vec => new Float32Array(a)

export const add = (a: Vec, b: Vec): Vec => {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]
  return out
}

export const sub = (a: Vec, b: Vec): Vec => {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i]
  return out
}

export const scale = (a: Vec, s: number): Vec => {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] * s
  return out
}

export const dot = (a: Vec, b: Vec): number => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

export const norm = (a: Vec): number => Math.sqrt(dot(a, a))

export const normalize = (a: Vec): Vec => {
  const n = norm(a)
  return n < 1e-10 ? clone(a) : scale(a, 1 / n)
}

export const l2dist = (a: Vec, b: Vec): number => norm(sub(a, b))

/**
 * Weighted mean of a set of vectors.
 * weights need not sum to 1 — normalized internally.
 */
export const weightedMean = (vecs: Vec[], weights: number[]): Vec => {
  if (vecs.length === 0) throw new Error('weightedMean: empty input')
  const d = vecs[0].length
  const out = new Float32Array(d)
  let wsum = 0
  for (let i = 0; i < vecs.length; i++) {
    const w = weights[i]
    wsum += w
    for (let j = 0; j < d; j++) out[j] += vecs[i][j] * w
  }
  for (let j = 0; j < d; j++) out[j] /= wsum
  return out
}

/**
 * Unweighted centroid of a set of vectors.
 * Equivalent to weightedMean with uniform weights.
 */
export const centroid = (vecs: Vec[]): Vec => {
  if (vecs.length === 0) throw new Error('centroid: empty input')
  const d = vecs[0].length
  const out = new Float32Array(d)
  for (const v of vecs) for (let j = 0; j < d; j++) out[j] += v[j]
  for (let j = 0; j < d; j++) out[j] /= vecs.length
  return out
}

/**
 * Project b onto the subspace orthogonal to a (unconditional).
 * Removes the full component of b along a, regardless of sign.
 *   b_perp = b - (b · â) â
 */
export const projectOut = (b: Vec, a: Vec): Vec => {
  const ahat = normalize(a)
  const proj = dot(b, ahat)
  return sub(b, scale(ahat, proj))
}

/**
 * Sign-aware velocity deflection (Gram-Schmidt, conditional).
 *
 * Removes the component of b pointing *toward* a only when that component
 * exists (b · â > 0).  If b already points away from a (b · â ≤ 0),
 * b is returned unchanged — removing it would dampen escape-directed velocity.
 *
 * Used in onSkip in place of projectOut.
 */
export const deflect = (b: Vec, a: Vec): Vec => {
  const an = norm(a)
  if (an < 1e-10) return clone(b)
  const ahat = scale(a, 1 / an)
  const proj = dot(b, ahat)
  if (proj <= 0) return clone(b)          // already pointing away — preserve
  return sub(b, scale(ahat, proj))        // remove toward-skip component only
}

/**
 * Convert a plain number[] to a Float32Array Vec.
 * Used when rehydrating embeddings from JSON or Chroma responses.
 */
export const fromArray = (a: number[]): Vec => new Float32Array(a)

/**
 * Convert a Vec to a plain number[] for JSON serialisation.
 */
export const toArray = (a: Vec): number[] => Array.from(a)
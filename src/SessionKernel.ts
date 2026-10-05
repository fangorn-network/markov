/**
 * sessionKernel.ts
 *
 * Corrected Markov kernel over the all-MiniLM-L6-v2 embedding space (d=384, L2).
 *
 * ── State ────────────────────────────────────────────────────────────────────
 *
 *   μ ∈ ℝ^d              position  (EMA of play embeddings)
 *   v ∈ ℝ^d              velocity  (EMA of displacement — direction of travel)
 *   σ ∈ ℝ^+              spread    (query Gaussian width)
 *   S ∈ (ℝ^d)^{≤W}       skip buffer
 *   c ∈ ℝ^d | null       skip centroid  ─┐ parameterize regional repulsion
 *   r ∈ ℝ^+              skip radius    ─┘ recomputed on every onSkip
 *   η ∈ [0,1]            entropy
 *   taste                leaky accumulators over {genres, moods, themes, contexts}
 *   artists              signed EMA affinity: +1 = played, −1 = skipped
 *   durationPref         EMA of played duration (ms)
 *   neg                  non-decaying dislike accumulator per artistId
 *   blacklist            permanent suppression set
 *   muted                session-scoped fatigue set (not persisted)
 *
 * ── Skip disambiguation ───────────────────────────────────────────────────────
 *
 *   onSkip() checks artists[a] at skip time to distinguish two signals:
 *
 *   Fatigue  (artists[a] > fatigue_threshold):
 *     The user has a positive play history with this artist and is skipping
 *     because they've heard them too much — not because they dislike them.
 *     → add to muted (session-scoped, not persisted)
 *     → neg unchanged (no path to permanent blacklist)
 *     → artist EMA unchanged (preference signal preserved)
 *     → geometric effects still fire (track won't resurface this session)
 *
 *   Genuine dislike  (artists[a] ≤ fatigue_threshold):
 *     Neutral or negative prior history — skip is likely real aversion.
 *     → neg[a] += delta_skip (path to permanent blacklist at theta_B)
 *     → artist EMA updated toward −1
 *     → full categorical + geometric effects
 *
 * ── Call chain ───────────────────────────────────────────────────────────────
 *
 *   const q        = queryVector(state)        // send to Chroma
 *   const weighted = reweight(hits, state)     // blacklist + muted filter + Gibbs
 *   const sampled  = sampleHit(weighted)       // draw one candidate
 *   state          = onPlay(state, track)      // or onSkip / onJump
 */

import { D, DEFAULTS } from './constants.ts'
import type {
  ChromaHit,
  KernelParams,
  KernelState,
  KernelStateJSON,
  WeightedHit,
  TrackFeatures,
  TasteWeights,
} from './types.ts'
import type { Vec } from './Vec.ts'
import {
  zeros,
  clone,
  add,
  sub,
  scale,
  norm,
  centroid,
  weightedMean,
  deflect,
  l2dist,
  fromArray,
  toArray,
} from './Vec.ts'

// ─────────────────────────────────────────────────────────────────────────────
// Taste helpers
// ─────────────────────────────────────────────────────────────────────────────

const emptyTaste = (): TasteWeights => ({
  genres: {}, moods: {}, themes: {}, contexts: {},
})

const decayTaste = (taste: TasteWeights, decay: number): TasteWeights => {
  const k  = 1 - decay
  const dk = (rec: Record<string, number>): Record<string, number> =>
    Object.fromEntries(Object.entries(rec).map(([tag, w]) => [tag, w * k]))
  return {
    genres:   dk(taste.genres),
    moods:    dk(taste.moods),
    themes:   dk(taste.themes),
    contexts: dk(taste.contexts),
  }
}

const accumulateTaste = (
  taste: TasteWeights,
  track: Pick<TrackFeatures, 'genres' | 'moods' | 'themes' | 'contexts'>,
  rate:  number,
): TasteWeights => {
  const acc = (rec: Record<string, number>, tags: string[]): Record<string, number> => {
    const out = { ...rec }
    for (const tag of tags) out[tag] = (out[tag] ?? 0) + rate
    return out
  }
  return {
    genres:   acc(taste.genres,   track.genres),
    moods:    acc(taste.moods,    track.moods),
    themes:   acc(taste.themes,   track.themes),
    contexts: acc(taste.contexts, track.contexts),
  }
}

const penalizeTaste = (
  taste:   TasteWeights,
  track:   Pick<TrackFeatures, 'genres' | 'moods' | 'themes' | 'contexts'>,
  penalty: number,
): TasteWeights => {
  const pen = (rec: Record<string, number>, tags: string[]): Record<string, number> => {
    const out = { ...rec }
    for (const tag of tags) out[tag] = Math.max((out[tag] ?? 0) - penalty, 0)
    return out
  }
  return {
    genres:   pen(taste.genres,   track.genres),
    moods:    pen(taste.moods,    track.moods),
    themes:   pen(taste.themes,   track.themes),
    contexts: pen(taste.contexts, track.contexts),
  }
}

const dimScore = (
  weights: Record<string, number>,
  tags:    string[],
  floor:   number,
): number => {
  if (tags.length === 0) return floor
  const total = Object.values(weights).reduce((s, w) => s + w, 0)
  if (total < 1e-10) return floor
  let score = 0
  for (const tag of tags) score += (weights[tag] ?? 0) / total
  return Math.max(score / tags.length, floor)
}

const taxonomyAffinity = (
  taste: TasteWeights,
  track: Pick<TrackFeatures, 'genres' | 'moods' | 'themes' | 'contexts'>,
  floor: number,
): number =>
  (
    dimScore(taste.genres,   track.genres,   floor) +
    dimScore(taste.moods,    track.moods,    floor) +
    dimScore(taste.themes,   track.themes,   floor) +
    dimScore(taste.contexts, track.contexts, floor)
  ) / 4

// ─────────────────────────────────────────────────────────────────────────────
// Affinity transforms
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Artist affinity via logistic transform.
 *
 *   f_art(a) = 2·σ(c·artists[a])     σ(x) = 1/(1+e^{−x})
 *
 *   artists[a] = 0  → f_art = 1,   log f_art = 0      (neutral)
 *   artists[a] → −∞ → f_art → 0,  log f_art → −∞     (suppressed)
 *   artists[a] → +∞ → f_art → 2,  log f_art → log 2  (mildly boosted)
 */
const artistAffinity = (
  artists:  Record<string, number>,
  artistId: string,
  c:        number,
): number => {
  const x = artists[artistId] ?? 0
  return 2 / (1 + Math.exp(-c * x))
}

const durationAffinity = (
  pref:       number | null,
  durationMs: number,
  sigmaMs:    number,
): number => {
  if (pref === null) return 1.0
  const delta = durationMs - pref
  return Math.exp(-(delta * delta) / (2 * sigmaMs * sigmaMs))
}

// ─────────────────────────────────────────────────────────────────────────────
// Skip region geometry
// ─────────────────────────────────────────────────────────────────────────────

const computeSkipRegion = (
  skips: Vec[],
): { skipCentroid: Vec | null; skipRadius: number } => {
  if (skips.length === 0) return { skipCentroid: null, skipRadius: 0 }
  const c = centroid(skips)
  let   r = 0
  for (const s of skips) {
    const d = l2dist(s, c)
    if (d > r) r = d
  }
  return { skipCentroid: c, skipRadius: r }
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

export const initFromSeeds = (
  tracks: TrackFeatures[],
  params: KernelParams = {},
): KernelState => {
  const { sigma_base = DEFAULTS.sigma_base } = params
  const d = params.d ?? tracks[0]?.embedding.length ?? DEFAULTS.d

  const harmonicWeights = tracks.map((_, i) => 1 / Math.log(i + 2))
  const mu = weightedMean(tracks.map(t => t.embedding), harmonicWeights)

  const seedRate = 1 / Math.max(tracks.length, 1)
  let   taste    = emptyTaste()
  const artists: Record<string, number> = {}

  for (const track of tracks) {
    taste = accumulateTaste(taste, track, seedRate)
    artists[track.artistId] = (artists[track.artistId] ?? 0) + seedRate
  }

  return {
    mu,
    v:            zeros(d),
    sigma:        sigma_base,
    skips:        [],
    skipCentroid: null,
    skipRadius:   0,
    t:            0,
    entropy:      0.2,
    taste,
    artists,
    durationPref: null,
    neg:          {},
    blacklist:    new Set(),
    muted:        new Set(),
  }
}

export const emptyKernel = (params: KernelParams = {}): KernelState => {
  const { sigma_base = DEFAULTS.sigma_base, d = DEFAULTS.d } = params
  return {
    mu:           zeros(d),
    v:            zeros(d),
    sigma:        sigma_base,
    skips:        [],
    skipCentroid: null,
    skipRadius:   0,
    t:            0,
    entropy:      0.2,
    taste:        emptyTaste(),
    artists:      {},
    durationPref: null,
    neg:          {},
    blacklist:    new Set(),
    muted:        new Set(),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Query
// ─────────────────────────────────────────────────────────────────────────────

export const queryVector = (state: KernelState, params: KernelParams = {}): Vec => {
  const { lambda_max = DEFAULTS.lambda_max } = params
  const vn = norm(state.v)
  if (vn < 1e-10) return clone(state.mu)
  const vhat  = scale(state.v, 1 / vn)
  const lambda = lambda_max * Math.tanh(vn)
  return add(state.mu, scale(vhat, lambda))
}

// ─────────────────────────────────────────────────────────────────────────────
// Reweighting — self-contained (blacklist + muted filter + Gibbs)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hard-filter hits against both the permanent blacklist and the session muted set.
 *
 * C̃_t = { e ∈ C_t : π(e) ∉ blacklist_t ∪ muted_t }
 *
 * Falls back to the original list if filtering empties the set.
 */
const applySuppressionFilter = (
  hits:      ChromaHit[],
  blacklist: Set<string>,
  muted:     Set<string>,
): ChromaHit[] => {
  const suppressed = blacklist.size + muted.size
  if (suppressed === 0) return hits

  const filtered = hits.filter(h => {
    const id = h.metadata?.artistId
    return id === undefined || (!blacklist.has(id) && !muted.has(id))
  })

  return filtered.length > 0 ? filtered : hits
}

export const reweight = (
  hits:   ChromaHit[],
  state:  KernelState,
  params: KernelParams = {},
): WeightedHit[] => {
  const {
    gamma_reg      = DEFAULTS.gamma_reg,
    temp_base      = DEFAULTS.temp_base,
    temp_max       = DEFAULTS.temp_max,
    tau_cat        = DEFAULTS.tau_cat,
    tau_art        = DEFAULTS.tau_art,
    tau_dur        = DEFAULTS.tau_dur,
    cat_floor      = DEFAULTS.cat_floor,
    art_logistic_c = DEFAULTS.art_logistic_c,
    dur_sigma_ms   = DEFAULTS.dur_sigma_ms,
  } = params

  // ── Step 1: hard suppression filter (blacklist ∪ muted) ──────────────────
  const candidates = applySuppressionFilter(hits, state.blacklist, state.muted)

  // ── Step 2: temperature ───────────────────────────────────────────────────
  const { sigma, entropy, skipCentroid, skipRadius } = state
  const temp = temp_base + entropy * (temp_max - temp_base)

  // ── Step 3: log-weights ───────────────────────────────────────────────────
  const logWeights = candidates.map(hit => {
    const e = fromArray(
      hit.embedding instanceof Float32Array
        ? Array.from(hit.embedding)
        : hit.embedding as number[]
    )

    let logW = -hit.distance / (2 * sigma * sigma)

    if (skipCentroid !== null && skipRadius > 1e-6) {
      const dc = l2dist(e, skipCentroid)
      logW -= gamma_reg * Math.exp(-(dc * dc) / (2 * skipRadius * skipRadius))
    }

    const meta = hit.metadata
    if (meta) {
      const catScore = taxonomyAffinity(state.taste, meta, cat_floor)
      logW += tau_cat * Math.log(Math.max(catScore, 1e-9))

      const artScore = artistAffinity(state.artists, meta.artistId, art_logistic_c)
      logW += tau_art * Math.log(Math.max(artScore, 1e-9))

      const durScore = durationAffinity(state.durationPref, meta.durationMs, dur_sigma_ms)
      logW += tau_dur * Math.log(Math.max(durScore, 1e-9))
    }

    return logW
  })

  // ── Step 4: temperature flattening + stable softmax ───────────────────────
  const tempered = logWeights.map(lw => lw / temp)
  const maxLW    = Math.max(...tempered)
  const raw      = tempered.map(lw => Math.exp(lw - maxLW))
  const total    = raw.reduce((s, w) => s + w, 0)

  if (total < 1e-12) {
    const u = 1 / candidates.length
    return candidates.map(h => ({ ...h, weight: u }))
  }

  return candidates.map((h, i) => ({ ...h, weight: raw[i] / total }))
}

export const sampleHit = (weighted: WeightedHit[]): WeightedHit => {
  let r = Math.random()
  for (const h of weighted) {
    r -= h.weight
    if (r <= 0) return h
  }
  return weighted[weighted.length - 1]
}

// ─────────────────────────────────────────────────────────────────────────────
// State transitions
// ─────────────────────────────────────────────────────────────────────────────

export const onPlay = (
  state:   KernelState,
  track:   TrackFeatures,
  muPrior: Vec | null = null,
  params:  KernelParams = {},
): KernelState => {
  const {
    alpha        = DEFAULTS.alpha,
    beta         = DEFAULTS.beta,
    sigma_base   = DEFAULTS.sigma_base,
    rho          = DEFAULTS.rho,
    prior_k      = DEFAULTS.prior_k,
    alpha_taste  = DEFAULTS.alpha_taste,
    taste_decay  = DEFAULTS.taste_decay,
    beta_artist  = DEFAULTS.beta_artist,
    rho_duration = DEFAULTS.rho_duration,
  } = params

  const e   = track.embedding
  const mu0 = state.mu
  const t1  = state.t + 1

  let mu1 = add(scale(e, alpha), scale(mu0, 1 - alpha))
  if (muPrior !== null) {
    const pull = prior_k / t1
    mu1 = add(mu1, scale(sub(muPrior, mu1), Math.min(pull, 1)))
  }
  const v1     = add(scale(state.v, beta), scale(sub(e, mu0), 1 - beta))
  const sigma1 = rho * state.sigma + (1 - rho) * sigma_base

  const taste1   = accumulateTaste(decayTaste(state.taste, taste_decay), track, alpha_taste)
  const artistId = track.artistId
  const artists1 = {
    ...state.artists,
    [artistId]: beta_artist * (state.artists[artistId] ?? 0) + (1 - beta_artist),
  }
  const durationPref1 =
    state.durationPref === null
      ? track.durationMs
      : rho_duration * state.durationPref + (1 - rho_duration) * track.durationMs

  const entropy1 = Math.max(state.entropy * 0.95, 0.05)

  return {
    ...state,
    mu:           mu1,
    v:            v1,
    sigma:        sigma1,
    t:            t1,
    entropy:      entropy1,
    taste:        taste1,
    artists:      artists1,
    durationPref: durationPref1,
    // skips, skipCentroid, skipRadius, neg, blacklist, muted unchanged on play
  }
}

/**
 * κ_skip: update state on a skipped track.
 *
 * Before any geometric update, classifies the skip as fatigue or dislike:
 *
 *   Fatigue  (artists[a] > fatigue_threshold):
 *     - muted ← muted ∪ {a}          session-scoped, not persisted
 *     - neg unchanged                 no path to permanent blacklist
 *     - artist EMA unchanged          preference signal preserved
 *     - taste penalized lightly       the track's tags still got a skip signal
 *     - geometric effects fire        track region is still repelled this session
 *
 *   Genuine dislike  (artists[a] ≤ fatigue_threshold):
 *     - neg[a] += delta_skip          path to permanent blacklist
 *     - blacklist updated if needed
 *     - artist EMA updated toward −1
 *     - taste penalized               full categorical signal
 *     - geometric effects fire
 *
 * Geometric effects (velocity deflection, position repulsion, skip buffer,
 * spread expansion, entropy increase) fire in both cases.
 */
export const onSkip = (
  state:  KernelState,
  track:  TrackFeatures,
  params: KernelParams = {},
): KernelState => {
  const {
    sigma_max         = DEFAULTS.sigma_max,
    gamma_base        = DEFAULTS.gamma_base,
    skip_window       = DEFAULTS.skip_window,
    skip_taste_pen    = DEFAULTS.skip_taste_pen,
    beta_artist       = DEFAULTS.beta_artist,
    delta_skip        = DEFAULTS.delta_skip,
    theta_B           = DEFAULTS.theta_B,
    epsilon           = DEFAULTS.epsilon,
    fatigue_threshold = DEFAULTS.fatigue_threshold,
  } = params

  const e        = track.embedding
  const mu       = state.mu
  const entropy  = state.entropy
  const artistId = track.artistId

  // ── Skip classification ───────────────────────────────────────────────────
  const priorEma  = state.artists[artistId] ?? 0
  const isFatigue = priorEma > fatigue_threshold

  // ── Fiber suppression (only on genuine dislike) ───────────────────────────
  let neg1       = state.neg
  let blacklist1 = state.blacklist
  let artists1   = state.artists
  let muted1     = state.muted
  let taste1     = state.taste

  if (isFatigue) {
    // Session mute — transient, no blacklist progression
    muted1 = new Set(state.muted)
    muted1.add(artistId)
    // Still penalize taste lightly (the specific track's tags got a skip)
    taste1 = penalizeTaste(state.taste, track, skip_taste_pen)
    // artists EMA and neg are intentionally unchanged
  } else {
    // Genuine dislike — full signal
    neg1 = { ...state.neg, [artistId]: (state.neg[artistId] ?? 0) + delta_skip }
    blacklist1 = new Set(state.blacklist)
    if (neg1[artistId] > theta_B) blacklist1.add(artistId)

    artists1 = {
      ...state.artists,
      [artistId]: beta_artist * priorEma + (1 - beta_artist) * (-1),
    }
    taste1 = penalizeTaste(state.taste, track, skip_taste_pen)
  }

  // ── Geometric effects (fire in both cases) ────────────────────────────────
  const v1 = deflect(state.v, sub(e, mu))

  const away     = sub(mu, e)
  const awayNorm = norm(away)
  const gamma    = gamma_base * (1 + entropy)
  const mu1      = awayNorm >= epsilon
    ? add(mu, scale(away, gamma / awayNorm))
    : mu

  const skips1 = [...state.skips, clone(e)]
  if (skips1.length > skip_window) skips1.shift()
  const { skipCentroid: skipCentroid1, skipRadius: skipRadius1 } = computeSkipRegion(skips1)

  const sigma1   = Math.min(state.sigma * 1.1, sigma_max)
  const entropy1 = Math.min(entropy * 1.1 + 0.02, 1.0)

  return {
    ...state,
    mu:           mu1,
    v:            v1,
    sigma:        sigma1,
    skips:        skips1,
    skipCentroid: skipCentroid1,
    skipRadius:   skipRadius1,
    entropy:      entropy1,
    taste:        taste1,
    artists:      artists1,
    neg:          neg1,
    blacklist:    blacklist1,
    muted:        muted1,
    // t, durationPref unchanged on skip
  }
}

export const onJump = (state: KernelState, params: KernelParams = {}): KernelState => {
  const { sigma_base = DEFAULTS.sigma_base } = params
  return { ...state, v: zeros(state.mu.length), sigma: sigma_base }
}

export const resetKernel = (params: KernelParams = {}): KernelState =>
  emptyKernel(params)

// ─────────────────────────────────────────────────────────────────────────────
// Serialisation — muted intentionally excluded (session-only)
// ─────────────────────────────────────────────────────────────────────────────

export const serializeState = (state: KernelState): KernelStateJSON => ({
  mu:           toArray(state.mu),
  v:            toArray(state.v),
  sigma:        state.sigma,
  skips:        state.skips.map(toArray),
  skipCentroid: state.skipCentroid ? toArray(state.skipCentroid) : null,
  skipRadius:   state.skipRadius,
  t:            state.t,
  entropy:      state.entropy,
  taste:        state.taste,
  artists:      state.artists,
  durationPref: state.durationPref,
  neg:          state.neg,
  blacklist:    [...state.blacklist].sort(),
  // muted deliberately absent — always starts empty on next session
})

/** Rehydrated state always has muted = new Set() — session fatigue clears between sessions. */
export const deserializeState = (json: KernelStateJSON): KernelState => ({
  mu:           new Float32Array(json.mu),
  v:            new Float32Array(json.v),
  sigma:        json.sigma,
  skips:        json.skips.map(a => new Float32Array(a)),
  skipCentroid: json.skipCentroid ? new Float32Array(json.skipCentroid) : null,
  skipRadius:   json.skipRadius,
  t:            json.t,
  entropy:      json.entropy,
  taste:        json.taste,
  artists:      json.artists,
  durationPref: json.durationPref,
  neg:          json.neg,
  blacklist:    new Set(json.blacklist),
  muted:        new Set(),  // session fatigue always starts fresh
})

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

export interface KernelSnapshot {
  speed:         number
  lookahead:     number
  spread:        number
  nSkips:        number
  skipRadius:    number
  timestep:      number
  entropy:       number
  topGenres:     [string, number][]
  topMoods:      [string, number][]
  topThemes:     [string, number][]
  topArtists:    [string, number][]
  durationPref:  string | null
  blacklisted:   string[]
  muted:         string[]           // session-muted artists
  nearThreshold: [string, number][] // approaching theta_B
}

export const describe = (
  state:  KernelState,
  params: KernelParams = {},
): KernelSnapshot => {
  const { lambda_max = DEFAULTS.lambda_max, theta_B = DEFAULTS.theta_B } = params

  const vn     = norm(state.v)
  const lambda = lambda_max * Math.tanh(vn)

  const topN = (rec: Record<string, number>, n = 3): [string, number][] =>
    Object.entries(rec)
      .sort(([, a], [, b]) => b - a)
      .slice(0, n)
      .map(([k, v]) => [k, Math.round(v * 1000) / 1000])

  const nearThreshold: [string, number][] = Object.entries(state.neg)
    .filter(([a, v]) => v >= theta_B * 0.5 && !state.blacklist.has(a))
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([k, v]) => [k, Math.round(v * 100) / 100])

  return {
    speed:         Math.round(vn * 1000) / 1000,
    lookahead:     Math.round(lambda * 1000) / 1000,
    spread:        Math.round(state.sigma * 1000) / 1000,
    nSkips:        state.skips.length,
    skipRadius:    Math.round(state.skipRadius * 1000) / 1000,
    timestep:      state.t,
    entropy:       Math.round(state.entropy * 1000) / 1000,
    topGenres:     topN(state.taste.genres),
    topMoods:      topN(state.taste.moods),
    topThemes:     topN(state.taste.themes),
    topArtists:    topN(state.artists),
    durationPref:  state.durationPref !== null
      ? `${Math.round(state.durationPref / 1000)}s`
      : null,
    blacklisted:   [...state.blacklist].sort(),
    muted:         [...state.muted].sort(),
    nearThreshold,
  }
}
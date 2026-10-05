/**
 * SessionKernel.ts
 *
 * A Markov kernel over an embedding space: a person's point of view, moved by
 * what they like and what they pass on. Nothing here knows what an item is.
 *
 * ── State ────────────────────────────────────────────────────────────────────
 *
 *   μ ∈ ℝ^d              position  (recency-weighted mean of liked embeddings)
 *   v ∈ ℝ^d              velocity  (EMA of displacement — direction of travel)
 *   σ ∈ ℝ^+              spread    (query Gaussian width)
 *   P ∈ (ℝ^d)^{≤W}       pass buffer
 *   c ∈ ℝ^d | null       pass centroid  ─┐ parameterize regional repulsion
 *   r ∈ ℝ^+              pass radius    ─┘ recomputed on every onPass
 *   η ∈ [0,1]            entropy
 *   facets               leaky accumulators over the caller's tag channels
 *   groups               signed EMA affinity: +1 = liked, −1 = passed
 *   scalarPref           EMA of liked scalars (log space)
 *   neg                  non-decaying dislike accumulator per group
 *   blacklist            permanent suppression set
 *   muted                session-scoped fatigue set (not persisted)
 *
 * An item is an embedding plus, optionally, a group, facets and a scalar. Each
 * optional field adds one signal when present and is ignored when absent, so a
 * caller with nothing but vectors gets the geometric kernel alone.
 *
 * ── Position ─────────────────────────────────────────────────────────────────
 *
 *   μ is a bias-corrected EMA: the t-th like moves it by
 *     w_t = α / (1 − (1−α)^t)
 *   of the way to the item. The first like (w=1) IS the position; after that
 *   the weights settle to α. A plain EMA started at the origin would leave μ
 *   short for the first dozen likes, and its velocity would mostly measure the
 *   walk out from the origin rather than the walk between items.
 *
 * ── Pass disambiguation ───────────────────────────────────────────────────────
 *
 *   onPass() checks groups[g] at pass time to distinguish two signals:
 *
 *   Fatigue  (groups[g] > fatigue_threshold):
 *     The person has a positive history with this group and is passing
 *     because they've had enough of it for now — not because they dislike it.
 *     → add to muted (session-scoped, not persisted)
 *     → neg unchanged (no path to permanent blacklist)
 *     → group EMA unchanged (preference signal preserved)
 *     → geometric effects still fire (item won't resurface this session)
 *
 *   Genuine dislike  (groups[g] ≤ fatigue_threshold, or no group):
 *     Neutral or negative prior history — the pass is likely real aversion.
 *     → neg[g] += delta_pass (path to permanent blacklist at theta_B)
 *     → group EMA updated toward −1
 *     → full facet + geometric effects
 *
 * ── Call chain ───────────────────────────────────────────────────────────────
 *
 *   const q        = queryVector(state)        // send to the vector store
 *   const weighted = reweight(hits, state)     // blacklist + muted filter + Gibbs
 *   const sampled  = sampleHit(weighted)       // draw one candidate
 *   state          = onLike(state, item)       // or onPass / onJump
 */

import { DEFAULTS } from './constants.ts'
import type {
  FacetWeights,
  Facets,
  Hit,
  Item,
  KernelParams,
  KernelState,
  KernelStateJSON,
  WeightedHit,
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
// Facet helpers
// ─────────────────────────────────────────────────────────────────────────────

const decayFacets = (facets: FacetWeights, decay: number): FacetWeights => {
  const k = 1 - decay
  return Object.fromEntries(Object.entries(facets).map(([ch, rec]) =>
    [ch, Object.fromEntries(Object.entries(rec).map(([tag, w]) => [tag, w * k]))]))
}

/** Add `delta` to every tag the item carries (clamped at 0 when negative). */
const bumpFacets = (facets: FacetWeights, item: Facets | undefined, delta: number): FacetWeights => {
  if (!item) return facets
  const out: FacetWeights = { ...facets }
  for (const [ch, tags] of Object.entries(item)) {
    if (!tags.length) continue
    const rec = { ...(out[ch] ?? {}) }
    for (const tag of tags) rec[tag] = Math.max((rec[tag] ?? 0) + delta, 0)
    out[ch] = rec
  }
  return out
}

const channelScore = (weights: Record<string, number>, tags: string[], floor: number): number => {
  const total = Object.values(weights).reduce((s, w) => s + w, 0)
  if (total < 1e-10) return floor
  let score = 0
  for (const tag of tags) score += (weights[tag] ?? 0) / total
  return Math.max(score / tags.length, floor)
}

/** Mean channel score over the channels this item has tags in; null when it
 *  has none, so an untagged item gets no facet term at all. */
const facetAffinity = (facets: FacetWeights, item: Facets | undefined, floor: number): number | null => {
  if (!item) return null
  const scores = Object.entries(item)
    .filter(([, tags]) => tags.length)
    .map(([ch, tags]) => channelScore(facets[ch] ?? {}, tags, floor))
  return scores.length ? scores.reduce((s, x) => s + x, 0) / scores.length : null
}

// ─────────────────────────────────────────────────────────────────────────────
// Affinity transforms
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Group affinity via logistic transform.
 *
 *   f_group(g) = 2·σ(c·groups[g])     σ(x) = 1/(1+e^{−x})
 *
 *   groups[g] = 0  → f = 1,   log f = 0      (neutral)
 *   groups[g] → −∞ → f → 0,  log f → −∞     (suppressed)
 *   groups[g] → +∞ → f → 2,  log f → log 2  (mildly boosted)
 */
const groupAffinity = (groups: Record<string, number>, group: string, c: number): number =>
  2 / (1 + Math.exp(-c * (groups[group] ?? 0)))

/** Gaussian in log-ratio: unit-free, and symmetric in "twice" and "half". */
const scalarAffinity = (logPref: number, x: number, sigma: number): number => {
  const delta = Math.log(x) - logPref
  return Math.exp(-(delta * delta) / (2 * sigma * sigma))
}

const hasScalar = (x: number | undefined): x is number => typeof x === 'number' && x > 0 && Number.isFinite(x)

// ─────────────────────────────────────────────────────────────────────────────
// Pass region geometry
// ─────────────────────────────────────────────────────────────────────────────

const computePassRegion = (passes: Vec[]): { passCentroid: Vec | null; passRadius: number } => {
  if (passes.length === 0) return { passCentroid: null, passRadius: 0 }
  const c = centroid(passes)
  let   r = 0
  for (const p of passes) {
    const d = l2dist(p, c)
    if (d > r) r = d
  }
  return { passCentroid: c, passRadius: r }
}

// ─────────────────────────────────────────────────────────────────────────────
// Initialisation
// ─────────────────────────────────────────────────────────────────────────────

/** A kernel seeded from items the person already likes, most important FIRST
 *  (harmonic weights). Each seed counts as a like already taken, so the next
 *  live like moves μ as the (n+1)-th would rather than overwriting the seeds. */
export const initFromSeeds = (items: Item[], params: KernelParams = {}): KernelState => {
  const k0 = emptyKernel({ ...params, d: params.d ?? items[0]?.embedding.length ?? DEFAULTS.d })
  if (!items.length) return k0

  const harmonicWeights = items.map((_, i) => 1 / Math.log(i + 2))
  const mu = weightedMean(items.map(it => it.embedding), harmonicWeights)

  const seedRate = 1 / items.length
  let   facets: FacetWeights = {}
  const groups: Record<string, number> = {}
  for (const it of items) {
    facets = bumpFacets(facets, it.facets, seedRate)
    if (it.group !== undefined) groups[it.group] = (groups[it.group] ?? 0) + seedRate
  }
  return { ...k0, mu, t: items.length, facets, groups }
}

export const emptyKernel = (params: KernelParams = {}): KernelState => {
  const { sigma_base = DEFAULTS.sigma_base, d = DEFAULTS.d } = params
  return {
    mu:           zeros(d),
    v:            zeros(d),
    sigma:        sigma_base,
    passes:       [],
    passCentroid: null,
    passRadius:   0,
    t:            0,
    entropy:      0.2,
    facets:       {},
    groups:       {},
    scalarPref:   null,
    neg:          {},
    blacklist:    new Set(),
    muted:        new Set(),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Query
// ─────────────────────────────────────────────────────────────────────────────

/** μ pushed along v by λ_max·tanh(‖v‖): you search slightly ahead of where you
 *  are, further the faster you are moving. */
export const queryVector = (state: KernelState, params: KernelParams = {}): Vec => {
  const { lambda_max = DEFAULTS.lambda_max } = params
  const vn = norm(state.v)
  if (vn < 1e-10) return clone(state.mu)
  const vhat   = scale(state.v, 1 / vn)
  const lambda = lambda_max * Math.tanh(vn)
  return add(state.mu, scale(vhat, lambda))
}

// ─────────────────────────────────────────────────────────────────────────────
// Reweighting — self-contained (blacklist + muted filter + Gibbs)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Hard-filter hits against both the permanent blacklist and the session muted set.
 *
 * C̃_t = { e ∈ C_t : group(e) ∉ blacklist_t ∪ muted_t }
 *
 * Falls back to the original list if filtering empties the set.
 */
const applySuppressionFilter = (hits: Hit[], blacklist: Set<string>, muted: Set<string>): Hit[] => {
  if (blacklist.size + muted.size === 0) return hits
  const filtered = hits.filter(h => {
    const g = h.metadata?.group
    return g === undefined || (!blacklist.has(g) && !muted.has(g))
  })
  return filtered.length > 0 ? filtered : hits
}

export const reweight = (hits: Hit[], state: KernelState, params: KernelParams = {}): WeightedHit[] => {
  const {
    gamma_reg        = DEFAULTS.gamma_reg,
    temp_base        = DEFAULTS.temp_base,
    temp_max         = DEFAULTS.temp_max,
    tau_facet        = DEFAULTS.tau_facet,
    tau_group        = DEFAULTS.tau_group,
    tau_scalar       = DEFAULTS.tau_scalar,
    facet_floor      = DEFAULTS.facet_floor,
    group_logistic_c = DEFAULTS.group_logistic_c,
    scalar_sigma     = DEFAULTS.scalar_sigma,
  } = params

  // ── Step 1: hard suppression filter (blacklist ∪ muted) ──────────────────
  const candidates = applySuppressionFilter(hits, state.blacklist, state.muted)

  // ── Step 2: temperature ───────────────────────────────────────────────────
  const { sigma, entropy, passCentroid, passRadius } = state
  const temp = temp_base + entropy * (temp_max - temp_base)

  // ── Step 3: log-weights ───────────────────────────────────────────────────
  const logWeights = candidates.map(hit => {
    const e = hit.embedding instanceof Float32Array ? hit.embedding : fromArray(hit.embedding)

    let logW = -hit.distance / (2 * sigma * sigma)

    if (passCentroid !== null && passRadius > 1e-6) {
      const dc = l2dist(e, passCentroid)
      logW -= gamma_reg * Math.exp(-(dc * dc) / (2 * passRadius * passRadius))
    }

    const meta = hit.metadata
    if (meta) {
      const f = facetAffinity(state.facets, meta.facets, facet_floor)
      if (f !== null) logW += tau_facet * Math.log(Math.max(f, 1e-9))

      if (meta.group !== undefined)
        logW += tau_group * Math.log(Math.max(groupAffinity(state.groups, meta.group, group_logistic_c), 1e-9))

      if (state.scalarPref !== null && hasScalar(meta.scalar))
        logW += tau_scalar * Math.log(Math.max(scalarAffinity(state.scalarPref, meta.scalar, scalar_sigma), 1e-9))
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

/** κ_like: the person took this item. */
export const onLike = (
  state:   KernelState,
  item:    Item,
  muPrior: Vec | null = null,
  params:  KernelParams = {},
): KernelState => {
  const {
    alpha       = DEFAULTS.alpha,
    beta        = DEFAULTS.beta,
    sigma_base  = DEFAULTS.sigma_base,
    rho         = DEFAULTS.rho,
    prior_k     = DEFAULTS.prior_k,
    alpha_facet = DEFAULTS.alpha_facet,
    facet_decay = DEFAULTS.facet_decay,
    beta_group  = DEFAULTS.beta_group,
    rho_scalar  = DEFAULTS.rho_scalar,
  } = params

  const e   = item.embedding
  const mu0 = state.mu
  const t1  = state.t + 1

  // Bias-corrected step: 1 on the first like, settling to α.
  const w = alpha / (1 - Math.pow(1 - alpha, t1))
  let mu1 = add(mu0, scale(sub(e, mu0), w))
  if (muPrior !== null) {
    const pull = prior_k / t1
    mu1 = add(mu1, scale(sub(muPrior, mu1), Math.min(pull, 1)))
  }
  // One position has no direction: velocity starts at the second like.
  const v1     = state.t === 0 ? state.v : add(scale(state.v, beta), scale(sub(e, mu0), 1 - beta))
  const sigma1 = rho * state.sigma + (1 - rho) * sigma_base

  const facets1 = bumpFacets(decayFacets(state.facets, facet_decay), item.facets, alpha_facet)
  const groups1 = item.group === undefined ? state.groups : {
    ...state.groups,
    [item.group]: beta_group * (state.groups[item.group] ?? 0) + (1 - beta_group),
  }
  const scalarPref1 = !hasScalar(item.scalar)
    ? state.scalarPref
    : state.scalarPref === null
      ? Math.log(item.scalar)
      : rho_scalar * state.scalarPref + (1 - rho_scalar) * Math.log(item.scalar)

  return {
    ...state,
    mu:         mu1,
    v:          v1,
    sigma:      sigma1,
    t:          t1,
    entropy:    Math.max(state.entropy * 0.95, 0.05),
    facets:     facets1,
    groups:     groups1,
    scalarPref: scalarPref1,
    // passes, passCentroid, passRadius, neg, blacklist, muted unchanged on a like
  }
}

/**
 * κ_pass: the person was shown this item and passed on it.
 *
 * Before any geometric update, classifies the pass as fatigue or dislike:
 *
 *   Fatigue  (groups[g] > fatigue_threshold):
 *     - muted ← muted ∪ {g}          session-scoped, not persisted
 *     - neg unchanged                 no path to permanent blacklist
 *     - group EMA unchanged           preference signal preserved
 *     - facets penalized              the item's tags still got a pass
 *     - geometric effects fire        item region is still repelled this session
 *
 *   Genuine dislike  (groups[g] ≤ fatigue_threshold):
 *     - neg[g] += delta_pass          path to permanent blacklist
 *     - blacklist updated if needed
 *     - group EMA updated toward −1
 *     - facets penalized
 *     - geometric effects fire
 *
 * An item with no group skips the classification: facets and geometry only.
 * Geometric effects (velocity deflection, position repulsion, pass buffer,
 * spread expansion, entropy increase) fire in every case.
 */
export const onPass = (state: KernelState, item: Item, params: KernelParams = {}): KernelState => {
  const {
    sigma_max         = DEFAULTS.sigma_max,
    gamma_base        = DEFAULTS.gamma_base,
    pass_window       = DEFAULTS.pass_window,
    pass_facet_pen    = DEFAULTS.pass_facet_pen,
    beta_group        = DEFAULTS.beta_group,
    delta_pass        = DEFAULTS.delta_pass,
    theta_B           = DEFAULTS.theta_B,
    epsilon           = DEFAULTS.epsilon,
    fatigue_threshold = DEFAULTS.fatigue_threshold,
  } = params

  const e       = item.embedding
  const mu      = state.mu
  const entropy = state.entropy
  const g       = item.group

  let neg1       = state.neg
  let blacklist1 = state.blacklist
  let groups1    = state.groups
  let muted1     = state.muted

  if (g !== undefined) {
    const prior = state.groups[g] ?? 0
    if (prior > fatigue_threshold) {
      // Session mute — transient, no blacklist progression
      muted1 = new Set(state.muted)
      muted1.add(g)
    } else {
      // Genuine dislike — full signal
      neg1 = { ...state.neg, [g]: (state.neg[g] ?? 0) + delta_pass }
      blacklist1 = new Set(state.blacklist)
      if (neg1[g] > theta_B) blacklist1.add(g)
      groups1 = { ...state.groups, [g]: beta_group * prior + (1 - beta_group) * (-1) }
    }
  }
  const facets1 = bumpFacets(state.facets, item.facets, -pass_facet_pen)

  // ── Geometric effects (fire in every case) ────────────────────────────────
  const v1 = deflect(state.v, sub(e, mu))

  const away     = sub(mu, e)
  const awayNorm = norm(away)
  const gamma    = gamma_base * (1 + entropy)
  const mu1      = state.t > 0 && awayNorm >= epsilon
    ? add(mu, scale(away, gamma / awayNorm))
    : mu

  const passes1 = [...state.passes, clone(e)]
  if (passes1.length > pass_window) passes1.shift()
  const { passCentroid, passRadius } = computePassRegion(passes1)

  return {
    ...state,
    mu:           mu1,
    v:            v1,
    sigma:        Math.min(state.sigma * 1.1, sigma_max),
    passes:       passes1,
    passCentroid,
    passRadius,
    entropy:      Math.min(entropy * 1.1 + 0.02, 1.0),
    facets:       facets1,
    groups:       groups1,
    neg:          neg1,
    blacklist:    blacklist1,
    muted:        muted1,
    // t, scalarPref unchanged on a pass
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
  passes:       state.passes.map(toArray),
  passCentroid: state.passCentroid ? toArray(state.passCentroid) : null,
  passRadius:   state.passRadius,
  t:            state.t,
  entropy:      state.entropy,
  facets:       state.facets,
  groups:       state.groups,
  scalarPref:   state.scalarPref,
  neg:          state.neg,
  blacklist:    [...state.blacklist].sort(),
  // muted deliberately absent — always starts empty on next session
})

/** Rehydrated state always has muted = new Set() — session fatigue clears
 *  between sessions. Also reads state saved by 0.0.x (skips/artists/taste/
 *  durationPref); a saved duration preference is dropped, since it was in ms
 *  and the scalar preference is a log. */
export const deserializeState = (json: KernelStateJSON): KernelState => {
  const old = json as Partial<KernelStateJSON> & Record<string, any>
  const passes = old.passes ?? old.skips ?? []
  const centroidArr = old.passCentroid ?? old.skipCentroid ?? null
  return {
    mu:           new Float32Array(json.mu),
    v:            new Float32Array(json.v),
    sigma:        json.sigma,
    passes:       passes.map((a: number[]) => new Float32Array(a)),
    passCentroid: centroidArr ? new Float32Array(centroidArr) : null,
    passRadius:   old.passRadius ?? old.skipRadius ?? 0,
    t:            json.t,
    entropy:      json.entropy,
    facets:       old.facets ?? old.taste ?? {},
    groups:       old.groups ?? old.artists ?? {},
    scalarPref:   old.scalarPref ?? null,
    neg:          json.neg ?? {},
    blacklist:    new Set(json.blacklist ?? []),
    muted:        new Set(),  // session fatigue always starts fresh
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Diagnostics
// ─────────────────────────────────────────────────────────────────────────────

export interface KernelSnapshot {
  speed:         number
  lookahead:     number
  spread:        number
  nPasses:       number
  passRadius:    number
  timestep:      number
  entropy:       number
  topFacets:     Record<string, [string, number][]>
  topGroups:     [string, number][]
  scalarPref:    number | null
  blacklisted:   string[]
  muted:         string[]           // session-muted groups
  nearThreshold: [string, number][] // approaching theta_B
}

export const describe = (state: KernelState, params: KernelParams = {}): KernelSnapshot => {
  const { lambda_max = DEFAULTS.lambda_max, theta_B = DEFAULTS.theta_B } = params

  const vn     = norm(state.v)
  const lambda = lambda_max * Math.tanh(vn)
  const r3     = (x: number) => Math.round(x * 1000) / 1000

  const topN = (rec: Record<string, number>, n = 3): [string, number][] =>
    Object.entries(rec)
      .sort(([, a], [, b]) => b - a)
      .slice(0, n)
      .map(([k, v]) => [k, r3(v)])

  const nearThreshold: [string, number][] = Object.entries(state.neg)
    .filter(([g, v]) => v >= theta_B * 0.5 && !state.blacklist.has(g))
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([k, v]) => [k, Math.round(v * 100) / 100])

  return {
    speed:         r3(vn),
    lookahead:     r3(lambda),
    spread:        r3(state.sigma),
    nPasses:       state.passes.length,
    passRadius:    r3(state.passRadius),
    timestep:      state.t,
    entropy:       r3(state.entropy),
    topFacets:     Object.fromEntries(Object.entries(state.facets).map(([ch, rec]) => [ch, topN(rec)])),
    topGroups:     topN(state.groups),
    scalarPref:    state.scalarPref === null ? null : r3(Math.exp(state.scalarPref)),
    blacklisted:   [...state.blacklist].sort(),
    muted:         [...state.muted].sort(),
    nearThreshold,
  }
}

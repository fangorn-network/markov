import type { Vec } from './Vec.ts'

// ── Params ────────────────────────────────────────────────────────────────────

export interface KernelParams {
  /** Dimension of the embedding space. Must match the catalogue's
   *  `embedding.dim`; a kernel is only meaningful inside one space. */
  d?: number
  // ── geometric ──────────────────────────────────────────────────────────────
  /** Position EMA rate α ∈ (0,1). */
  alpha?: number
  /** Velocity EMA rate β ∈ (0,1). */
  beta?: number
  /** Lookahead saturation bound λ_max.
   *  Must exceed the radius of clusters you want the kernel to escape from.
   *  Calibrate against your catalogue's point cloud. */
  lambda_max?: number
  /** Baseline spread σ_base. Spread contracts toward this on likes. */
  sigma_base?: number
  /** Maximum spread σ_max. Spread cannot exceed this on passes. */
  sigma_max?: number
  /** Spread EMA rate ρ ∈ (0,1). */
  rho?: number
  /** Numerical stability floor ε. */
  epsilon?: number
  /** Prior pull decay constant k. Pull = k/t decays with timestep. */
  prior_k?: number
  /** Pass buffer capacity W. Older passes are evicted FIFO. */
  pass_window?: number

  // ── repulsion ──────────────────────────────────────────────────────────────
  /**
   * Base position repulsion magnitude γ_base.
   * Actual magnitude = γ_base · (1 + η) — entropy-coupled, decoupled from σ.
   */
  gamma_base?: number
  /**
   * Amplitude of the Gaussian regional repulsion field in the log-weight.
   *   − γ_reg · exp(−‖e − c‖² / (2r²))
   * Field is centred at the pass centroid with width = pass radius.
   */
  gamma_reg?: number

  // ── temperature ────────────────────────────────────────────────────────────
  /** Minimum Gibbs temperature (low entropy / exploitation). */
  temp_base?: number
  /** Maximum Gibbs temperature (high entropy / exploration). */
  temp_max?: number

  // ── facet accumulators ─────────────────────────────────────────────────────
  /** Accumulation rate for facet tags on a like. */
  alpha_facet?: number
  /** Per-step decay rate for all facet weights. */
  facet_decay?: number
  /** Per-tag penalty applied to facet accumulators on a pass. */
  pass_facet_pen?: number

  // ── group affinity ─────────────────────────────────────────────────────────
  /** Group EMA smoothing factor β_group ∈ (0,1). Higher = slower to forget. */
  beta_group?: number
  /**
   * Logistic steepness c for the group affinity transform.
   *   f_group = 2σ(c · groups[g])
   * Equals 1 at groups[g]=0 (neutral). Unbounded below as groups[g]→−∞.
   */
  group_logistic_c?: number

  // ── scalar preference ──────────────────────────────────────────────────────
  /** Scalar-preference EMA smoothing factor ρ_scalar ∈ (0,1). */
  rho_scalar?: number
  /** Width of the soft scalar preference, in log-ratio units: an item at
   *  e^σ times (or 1/e^σ of) the preferred value scores exp(−½). Unit-free, so
   *  it means the same for a duration, a price or a page count. */
  scalar_sigma?: number

  // ── log-weight coefficients ────────────────────────────────────────────────
  /** Facet affinity coefficient τ_facet. */
  tau_facet?: number
  /** Group affinity coefficient τ_group. */
  tau_group?: number
  /** Scalar affinity coefficient τ_scalar. */
  tau_scalar?: number
  /** Floor for per-channel facet scores. Prevents total suppression of
   *  unexplored tag space early in a session. */
  facet_floor?: number

  // ── persistent suppression ────────────────────────────────────────────────
  /** Negative accumulator increment per genuine-dislike pass. neg[g] += delta_pass. */
  delta_pass?: number
  /**
   * Suppression threshold θ_B.
   * A group enters the blacklist permanently when neg[g] > theta_B.
   */
  theta_B?: number

  // ── session fatigue ────────────────────────────────────────────────────────
  /**
   * Group EMA value above which a pass is read as session fatigue rather than
   * genuine dislike.
   *
   * When groups[g] > fatigue_threshold at pass time:
   *   → group is added to the session-scoped `muted` set
   *   → neg[g] is NOT incremented (no path to blacklist)
   *   → group EMA is NOT updated (still liked)
   *   → geometric repulsion still fires (item won't resurface this session)
   *
   * When groups[g] ≤ fatigue_threshold at pass time:
   *   → treated as genuine dislike
   *   → neg[g] incremented as normal
   *
   * With beta_group=0.7, groups[g] exceeds 0.3 after roughly one like, and
   * 0.5 after two. Set higher to require more likes before a pass is treated
   * as fatigue rather than dislike.
   */
  fatigue_threshold?: number
}

// ── Items ─────────────────────────────────────────────────────────────────────

/** Facet tags by channel, e.g. `{ genre: ['ambient'], mood: ['calm'] }`. The
 *  channel names are the caller's; the kernel only keeps them apart. */
export type Facets = Record<string, string[]>

/** Learned facet weights by channel, then by tag. */
export type FacetWeights = Record<string, Record<string, number>>

/**
 * Something a person can like or pass on. Only the embedding is required; every
 * other field adds one signal when present and is ignored when absent.
 */
export interface Item {
  embedding: Vec
  /** Who or what the item belongs to: an artist, a developer, a publisher, a
   *  feed. Group affinity, fatigue and permanent suppression act on it. */
  group?: string
  /** Tags by channel. */
  facets?: Facets
  /** A positive quantity the person develops a preferred value of: a duration,
   *  a price, a length. Compared as a ratio, so its unit does not matter. */
  scalar?: number
}

/** The non-geometric part of an item, as carried on a search hit. */
export type ItemMeta = Omit<Item, 'embedding'>

export interface Hit {
  id:        string
  embedding: number[] | Float32Array
  distance:  number
  metadata?: ItemMeta
}

export interface WeightedHit extends Hit {
  weight: number
}

// ── Kernel state ──────────────────────────────────────────────────────────────

export interface KernelState {
  // geometric
  /** Position: a recency-weighted mean of liked embeddings. Bias-corrected, so
   *  it never starts at the origin: the first like IS the position. */
  mu:           Vec
  /** Velocity: an EMA of displacement between consecutive positions. Zero
   *  until the second like, since one position has no direction. */
  v:            Vec
  sigma:        number
  passes:       Vec[]
  /** Centroid of the pass buffer. Null when buffer is empty. */
  passCentroid: Vec | null
  /** Radius of the pass buffer: max distance from centroid to any pass. */
  passRadius:   number
  /** Number of likes so far. */
  t:            number
  entropy:      number
  // categorical
  facets:       FacetWeights
  groups:       Record<string, number>   // signed EMA: +1 = liked, −1 = passed
  scalarPref:   number | null
  // persistent suppression (never decays, survives sessions)
  neg:          Record<string, number>
  blacklist:    Set<string>
  /**
   * Session-scoped mute set.
   *
   * Groups land here when they are passed on while their EMA is positive (the
   * person likes them but is fatigued). They are excluded from candidates for
   * the rest of the session identically to the blacklist, but:
   *   - neg is NOT incremented → no path to permanent blacklist
   *   - group EMA is NOT updated → preference signal preserved
   *   - muted is NOT persisted → clears on every cold start
   */
  muted:        Set<string>
}

/**
 * JSON-serialisable form of KernelState.
 * Float32Array → number[], Set<string> → string[].
 * Note: `muted` is intentionally absent — it is session-scoped and always
 * starts empty on deserialisation.
 */
export interface KernelStateJSON {
  mu:           number[]
  v:            number[]
  sigma:        number
  passes:       number[][]
  passCentroid: number[] | null
  passRadius:   number
  t:            number
  entropy:      number
  facets:       FacetWeights
  groups:       Record<string, number>
  scalarPref:   number | null
  neg:          Record<string, number>
  blacklist:    string[]
  // muted deliberately omitted — session-only
}

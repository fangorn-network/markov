import type { Vec } from './Vec.ts'

// ── Params ────────────────────────────────────────────────────────────────────

export interface KernelParams {
  // ── geometric ──────────────────────────────────────────────────────────────
  /** Position EMA rate α ∈ (0,1). */
  alpha?: number
  /** Velocity EMA rate β ∈ (0,1). */
  beta?: number
  /** Lookahead saturation bound λ_max.
   *  Must exceed the radius of clusters you want the kernel to escape from.
   *  Calibrate against your ChromaDB point cloud. */
  lambda_max?: number
  /** Baseline spread σ_base. Spread contracts toward this on plays. */
  sigma_base?: number
  /** Maximum spread σ_max. Spread cannot exceed this on skips. */
  sigma_max?: number
  /** Spread EMA rate ρ ∈ (0,1). */
  rho?: number
  /** Numerical stability floor ε. */
  epsilon?: number
  /** Prior pull decay constant k. Pull = k/t decays with timestep. */
  prior_k?: number
  /** Skip buffer capacity W. Older skips are evicted FIFO. */
  skip_window?: number

  // ── repulsion ──────────────────────────────────────────────────────────────
  /**
   * Base position repulsion magnitude γ_base.
   * Actual magnitude = γ_base · (1 + η) — entropy-coupled, decoupled from σ.
   */
  gamma_base?: number
  /**
   * Amplitude of the Gaussian regional repulsion field in the log-weight.
   *   − γ_reg · exp(−‖e − c‖² / (2r²))
   * Field is centred at the skip cluster centroid with width = skip radius.
   */
  gamma_reg?: number

  // ── temperature ────────────────────────────────────────────────────────────
  /** Minimum Gibbs temperature (low entropy / exploitation). */
  temp_base?: number
  /** Maximum Gibbs temperature (high entropy / exploration). */
  temp_max?: number

  // ── taste accumulators ─────────────────────────────────────────────────────
  /** Accumulation rate for taxonomy tags on play. */
  alpha_taste?: number
  /** Per-step decay rate for all taste weights. */
  taste_decay?: number
  /** Per-tag penalty applied to taxonomy accumulators on skip. */
  skip_taste_pen?: number

  // ── artist affinity ────────────────────────────────────────────────────────
  /** Artist EMA smoothing factor β_art ∈ (0,1). Higher = slower to forget. */
  beta_artist?: number
  /**
   * Logistic steepness c for the artist affinity transform.
   *   f_art = 2σ(c · artists[a])
   * Equals 1 at artists[a]=0 (neutral). Unbounded below as artists[a]→−∞.
   */
  art_logistic_c?: number

  // ── duration preference ────────────────────────────────────────────────────
  /** Duration EMA smoothing factor ρ_dur ∈ (0,1). */
  rho_duration?: number
  /** Standard deviation (ms) for the soft duration preference Gaussian. */
  dur_sigma_ms?: number

  // ── log-weight coefficients ────────────────────────────────────────────────
  /** Taxonomy affinity coefficient τ_cat. */
  tau_cat?: number
  /** Artist affinity coefficient τ_art. */
  tau_art?: number
  /** Duration affinity coefficient τ_dur. */
  tau_dur?: number
  /** Floor for taxonomy dimension scores. Prevents total suppression of
   *  unexplored tag space early in a session. */
  cat_floor?: number

  // ── persistent suppression ────────────────────────────────────────────────
  /** Negative accumulator increment per genuine-dislike skip. neg[a] += delta_skip. */
  delta_skip?: number
  /**
   * Suppression threshold θ_B.
   * Artist enters the blacklist permanently when neg[a] > theta_B.
   */
  theta_B?: number

  // ── session fatigue ────────────────────────────────────────────────────────
  /**
   * Artist EMA value above which a skip is interpreted as session fatigue
   * rather than genuine dislike.
   *
   * When artists[a] > fatigue_threshold at skip time:
   *   → artist is added to the session-scoped `muted` set
   *   → neg[a] is NOT incremented (no path to blacklist)
   *   → artist EMA is NOT updated (still liked)
   *   → geometric repulsion still fires (track won't resurface this session)
   *
   * When artists[a] ≤ fatigue_threshold at skip time:
   *   → treated as genuine dislike
   *   → neg[a] incremented as normal
   *
   * With beta_artist=0.7, artists[a] exceeds 0.3 after roughly one play,
   * and exceeds 0.5 after two plays. Set higher to require more plays before
   * a skip is treated as fatigue rather than dislike.
   */
  fatigue_threshold?: number
}

// ── Track data ────────────────────────────────────────────────────────────────

export type FieldValue = string | number | boolean | null

export interface TasteWeights {
  genres:   Record<string, number>
  moods:    Record<string, number>
  themes:   Record<string, number>
  contexts: Record<string, number>
}

export interface TrackFeatures {
  embedding:  Vec
  artistId:   string
  genres:     string[]
  moods:      string[]
  themes:     string[]
  contexts:   string[]
  durationMs: number
}

export interface ChromaHit {
  id:        string
  embedding: number[] | Float32Array
  distance:  number
  metadata?: {
    artistId:   string
    genres:     string[]
    moods:      string[]
    themes:     string[]
    contexts:   string[]
    durationMs: number
  }
}

export interface WeightedHit extends ChromaHit {
  weight: number
}

// ── Kernel state ──────────────────────────────────────────────────────────────

export interface KernelState {
  // geometric
  mu:           Vec
  v:            Vec
  sigma:        number
  skips:        Vec[]
  /** Centroid of the skip buffer. Null when buffer is empty. */
  skipCentroid: Vec | null
  /** Radius of the skip buffer: max distance from centroid to any skip. */
  skipRadius:   number
  t:            number
  entropy:      number
  // categorical
  taste:        TasteWeights
  artists:      Record<string, number>   // signed EMA: +1 = played, −1 = skipped
  durationPref: number | null
  // persistent suppression (never decays, survives sessions)
  neg:          Record<string, number>
  blacklist:    Set<string>
  /**
   * Session-scoped mute set.
   *
   * Artists land here when they are skipped while their EMA is positive
   * (the user likes them but is fatigued). They are excluded from candidates
   * for the rest of the session identically to the blacklist, but:
   *   - neg is NOT incremented → no path to permanent blacklist
   *   - artist EMA is NOT updated → preference signal preserved
   *   - muted is NOT persisted to disk → clears on every cold start
   *
   * The user can also clear it explicitly via clearMuted() / unmuteArtist().
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
  skips:        number[][]
  skipCentroid: number[] | null
  skipRadius:   number
  t:            number
  entropy:      number
  taste:        TasteWeights
  artists:      Record<string, number>
  durationPref: number | null
  neg:          Record<string, number>
  blacklist:    string[]
  // muted deliberately omitted — session-only
}
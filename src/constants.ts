/**
 * constants.ts
 *
 * Embedding dimension and default kernel parameters.
 *
 * Calibration notes
 * ─────────────────
 * lambda_max        Must exceed the radius of the largest cluster you want the
 *                   kernel to escape. Measure empirically against your ChromaDB
 *                   point cloud.
 *
 * sigma_max         Caps spread at 2× sigma_base by default.
 *
 * gamma_base        Base repulsion step. Actual magnitude = gamma_base × (1+η).
 *                   Decoupled from σ so repulsion does not weaken as spread
 *                   contracts on play.
 *
 * theta_B           Suppression threshold. With delta_skip=1, an artist is
 *                   permanently blacklisted after 3 genuine-dislike skips.
 *
 * art_logistic_c    Logistic steepness. c=3 gives f_art ≈ 0.05 at artists=−1
 *                   (strong suppression) and f_art ≈ 1.95 at artists=+1.
 *
 * fatigue_threshold Artist EMA value above which a skip is interpreted as
 *                   session fatigue rather than genuine dislike. With
 *                   beta_artist=0.7, artists[a] ≈ 0.30 after 1 play and
 *                   ≈ 0.51 after 2 plays. The default of 0.3 means a single
 *                   prior play is enough to activate fatigue protection.
 *                   Raise to require more listening history before protecting
 *                   an artist from the neg accumulator.
 */

import type { KernelParams } from './types.ts'

/** Canonical embedding space: nomic-embed-text-v1.5.
 *  Matryoshka, so 512/256/128 truncations stay usable for centroids and
 *  coarse scans. Override per-kernel with `params.d` when mounting a
 *  catalogue that declares a different `embedding.dim`. */
export const D = 768

export const DEFAULTS: Required<KernelParams> = {
  d:                 D,
  // geometric
  alpha:             0.10,
  beta:              0.30,
  lambda_max:        0.40,
  sigma_base:        0.50,
  sigma_max:         1.00,
  rho:               0.80,
  epsilon:           1e-6,
  prior_k:           1.00,
  skip_window:       20,

  // repulsion
  gamma_base:        0.15,
  gamma_reg:         2.00,

  // temperature
  temp_base:         1.00,
  temp_max:          3.00,

  // taste
  alpha_taste:       0.30,
  taste_decay:       0.05,
  skip_taste_pen:    0.20,

  // artist
  beta_artist:       0.70,
  art_logistic_c:    3.00,

  // duration
  rho_duration:      0.80,
  dur_sigma_ms:      60_000,

  // log-weight coefficients
  tau_cat:           1.00,
  tau_art:           2.00,
  tau_dur:           0.50,
  cat_floor:         0.01,

  // persistent suppression
  delta_skip:        1.00,
  theta_B:           3.00,

  // session fatigue
  fatigue_threshold: 0.30,
}
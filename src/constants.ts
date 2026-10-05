/**
 * constants.ts
 *
 * Embedding dimension and default kernel parameters.
 *
 * Calibration notes
 * ─────────────────
 * lambda_max        Must exceed the radius of the largest cluster you want the
 *                   kernel to escape. Measure empirically against your
 *                   catalogue's point cloud.
 *
 * sigma_max         Caps spread at 2× sigma_base by default.
 *
 * gamma_base        Base repulsion step. Actual magnitude = gamma_base × (1+η).
 *                   Decoupled from σ so repulsion does not weaken as spread
 *                   contracts on likes.
 *
 * theta_B           Suppression threshold. With delta_pass=1, a group is
 *                   permanently blacklisted after 4 genuine-dislike passes
 *                   (neg must EXCEED theta_B).
 *
 * group_logistic_c  Logistic steepness. c=3 gives f_group ≈ 0.05 at groups=−1
 *                   (strong suppression) and f_group ≈ 1.95 at groups=+1.
 *
 * scalar_sigma      0.5 in log-ratio units: an item 1.65× (or 0.6×) the
 *                   preferred value scores exp(−½) on the scalar term.
 *
 * fatigue_threshold Group EMA value above which a pass is read as session
 *                   fatigue rather than genuine dislike. With beta_group=0.7,
 *                   groups[g] ≈ 0.30 after 1 like and ≈ 0.51 after 2. The
 *                   default of 0.3 means a single prior like is enough to
 *                   activate fatigue protection.
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
  pass_window:       20,

  // repulsion
  gamma_base:        0.15,
  gamma_reg:         2.00,

  // temperature
  temp_base:         1.00,
  temp_max:          3.00,

  // facets
  alpha_facet:       0.30,
  facet_decay:       0.05,
  pass_facet_pen:    0.20,

  // group
  beta_group:        0.70,
  group_logistic_c:  3.00,

  // scalar
  rho_scalar:        0.80,
  scalar_sigma:      0.50,

  // log-weight coefficients
  tau_facet:         1.00,
  tau_group:         2.00,
  tau_scalar:        0.50,
  facet_floor:       0.01,

  // persistent suppression
  delta_pass:        1.00,
  theta_B:           3.00,

  // session fatigue
  fatigue_threshold: 0.30,
}

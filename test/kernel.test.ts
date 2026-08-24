/**
 * One runnable check for the kernel, and one experiment.
 *
 * The check: playing moves the query toward what you played, skipping pushes
 * it away, and state survives a serialise round-trip.
 *
 * The experiment (`transfer`): a profile is built ONLY from items in domain A,
 * then used to rank items in domain B that the kernel has never seen. If the
 * right B items come out on top, taste is portable across domains — which is
 * the whole thesis, reduced to synthetic vectors.
 */
import assert from 'node:assert/strict'
import { D } from '../src/constants.ts'
import { emptyKernel, onPlay, onSkip, queryVector, reweight, serialiseState, deserialiseState } from '../src/SessionKernel.ts'
import type { TrackFeatures, ChromaHit } from '../src/types.ts'
import { zeros, dot, norm, scale } from '../src/Vec.ts'

// ── a synthetic conceptual space: two well-separated regions ────────────────
const unit = (axes: number[]): Float32Array => {
  const v = zeros(D)
  for (const a of axes) v[a] = 1
  return scale(v, 1 / norm(v))
}
/** i-th member of a region: mostly the region centre, slightly its own. */
const member = (region: number[], i: number): Float32Array => {
  const v = zeros(D)
  for (const a of region) v[a] = 0.9
  v[region[0] + 20 + i] = 0.6
  return scale(v, 1 / norm(v))
}

const REGION_A = [0, 1, 2, 3, 4]
const REGION_B = [200, 201, 202, 203, 204]
const centreA = unit(REGION_A)
const centreB = unit(REGION_B)

const cos = (a: Float32Array, b: Float32Array) => dot(a, b) / (norm(a) * norm(b))

const track = (embedding: Float32Array, artistId: string, tags: string[]): TrackFeatures => ({
  embedding, artistId,
  genres: tags, moods: [], themes: [], contexts: [],
  durationMs: 200_000,
})

const hit = (id: string, embedding: Float32Array, q: Float32Array, artistId: string, tags: string[]): ChromaHit => ({
  id, embedding,
  distance: 1 - cos(embedding, q),          // cosine distance, as a vector DB would return
  metadata: { artistId, genres: tags, moods: [], themes: [], contexts: [], durationMs: 200_000 },
})

// ── 1. playing moves the query toward what was played ───────────────────────
let k = emptyKernel()
for (let i = 0; i < 4; i++) k = onPlay(k, track(member(REGION_A, i), `artist-a${i}`, ['ambient']))

const q = queryVector(k)
assert.ok(cos(q, centreA) > cos(q, centreB),
  'query vector must lean toward the region that was played')
assert.ok(cos(q, centreA) > 0.5,
  `query vector should sit near region A, got cos=${cos(q, centreA).toFixed(3)}`)

// ── 2. ranking prefers the played region ────────────────────────────────────
const candidates = [
  hit('b0', member(REGION_B, 0), q, 'artist-b0', ['metal']),
  hit('a9', member(REGION_A, 9), q, 'artist-a9', ['ambient']),
]
const ranked = [...reweight(candidates, k)].sort((x, y) => y.weight - x.weight)
assert.equal(ranked[0].id, 'a9', 'a candidate in the played region must outrank a distant one')

// ── 3. skipping pushes away ─────────────────────────────────────────────────
const beforeB = reweight(candidates, k).find(h => h.id === 'b0')!.weight
let k2 = k
for (let i = 0; i < 3; i++) k2 = onSkip(k2, track(member(REGION_B, i), `artist-b${i}`, ['metal']))
const afterB = reweight(candidates, k2).find(h => h.id === 'b0')!.weight
assert.ok(afterB < beforeB,
  `skipping a region must lower its weight (${beforeB.toFixed(4)} -> ${afterB.toFixed(4)})`)

// ── 4. state survives a round-trip ──────────────────────────────────────────
const back = deserialiseState(serialiseState(k2))
assert.deepEqual(Array.from(back.mu), Array.from(k2.mu), 'mu must survive serialisation')
assert.equal(back.sigma, k2.sigma)
assert.equal(back.t, k2.t)

// ── 5. THE EXPERIMENT: does the profile transfer to an unseen domain? ───────
// Domain B here is a different *catalogue* sharing the same embedding space —
// no artist, tag, or item overlap with anything the kernel has seen.
const foreign = [
  hit('doc-near', member(REGION_A, 30), queryVector(k), 'author-x', ['essay']),
  hit('doc-far',  member(REGION_B, 30), queryVector(k), 'author-y', ['essay']),
]
const foreignRanked = [...reweight(foreign, k)].sort((x, y) => y.weight - x.weight)
const margin = foreignRanked[0].weight / foreignRanked[1].weight
assert.equal(foreignRanked[0].id, 'doc-near',
  'a profile learned in one domain must rank an unseen domain by the same geometry')

console.log(`ok — 5 checks passed`)
console.log(`   query lean:      cos(A)=${cos(q, centreA).toFixed(3)}  cos(B)=${cos(q, centreB).toFixed(3)}`)
console.log(`   skip effect:     ${beforeB.toFixed(4)} -> ${afterB.toFixed(4)}`)
console.log(`   transfer margin: ${margin.toFixed(2)}x  (unseen domain, zero shared metadata)`)

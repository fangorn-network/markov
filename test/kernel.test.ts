/**
 * One runnable check for the kernel, and one experiment.
 *
 * The check: playing moves the query toward what you played, skipping pushes
 * it away, and state survives a serialize round-trip.
 *
 * The experiment (`transfer`): a profile is built ONLY from items in catalogue
 * A, then used to rank items in catalogue B that the kernel has never seen. If
 * the right B items come out on top, taste is portable between catalogues —
 * which is the whole thesis, reduced to synthetic vectors.
 *
 * Both run at 384 and at the canonical 768, because a kernel that silently
 * assumes one dimension is a kernel that cannot mount a second catalogue.
 */
import assert from 'node:assert/strict'
import { D } from '../src/constants.ts'
import { emptyKernel, onPlay, onSkip, queryVector, reweight, serializeState, deserializeState } from '../src/SessionKernel.ts'
import type { TrackFeatures, ChromaHit } from '../src/types.ts'
import { zeros, dot, norm, scale } from '../src/Vec.ts'

const cos = (a: Float32Array, b: Float32Array) => dot(a, b) / (norm(a) * norm(b))

const run = (d: number) => {
  // ── a synthetic conceptual space: two well-separated regions ──────────────
  const unit = (axes: number[]): Float32Array => {
    const v = zeros(d)
    for (const a of axes) v[a] = 1
    return scale(v, 1 / norm(v))
  }
  /** i-th member of a region: mostly the region centre, slightly its own. */
  const member = (region: number[], i: number): Float32Array => {
    const v = zeros(d)
    for (const a of region) v[a] = 0.9
    v[region[0] + 20 + i] = 0.6
    return scale(v, 1 / norm(v))
  }

  const A = [0, 1, 2, 3, 4]
  const B = [d / 2, d / 2 + 1, d / 2 + 2, d / 2 + 3, d / 2 + 4]
  const centreA = unit(A)
  const centreB = unit(B)

  const track = (embedding: Float32Array, artistId: string, tags: string[]): TrackFeatures => ({
    embedding, artistId, genres: tags, moods: [], themes: [], contexts: [], durationMs: 200_000,
  })
  const hit = (id: string, embedding: Float32Array, q: Float32Array, artistId: string, tags: string[]): ChromaHit => ({
    id, embedding,
    distance: 1 - cos(embedding, q),        // cosine distance, as a vector store would return
    metadata: { artistId, genres: tags, moods: [], themes: [], contexts: [], durationMs: 200_000 },
  })

  // ── 1. playing moves the query toward what was played ─────────────────────
  let k = emptyKernel({ d })
  assert.equal(k.mu.length, d, 'kernel must allocate in the space it was given')
  for (let i = 0; i < 4; i++) k = onPlay(k, track(member(A, i), `artist-a${i}`, ['ambient']))

  const q = queryVector(k)
  assert.ok(cos(q, centreA) > cos(q, centreB), 'query must lean toward what was played')
  assert.ok(cos(q, centreA) > 0.5, `query should sit near region A, got ${cos(q, centreA).toFixed(3)}`)

  // ── 2. ranking follows the lean ───────────────────────────────────────────
  const candidates = [
    hit('b0', member(B, 0), q, 'artist-b0', ['metal']),
    hit('a9', member(A, 9), q, 'artist-a9', ['ambient']),
  ]
  const ranked = [...reweight(candidates, k)].sort((x, y) => y.weight - x.weight)
  assert.equal(ranked[0].id, 'a9', 'a candidate in the played region must outrank a distant one')

  // ── 3. skipping pushes away ───────────────────────────────────────────────
  const beforeB = reweight(candidates, k).find(h => h.id === 'b0')!.weight
  let k2 = k
  for (let i = 0; i < 3; i++) k2 = onSkip(k2, track(member(B, i), `artist-b${i}`, ['metal']))
  const afterB = reweight(candidates, k2).find(h => h.id === 'b0')!.weight
  assert.ok(afterB < beforeB, `skipping a region must lower its weight (${beforeB.toFixed(4)} -> ${afterB.toFixed(4)})`)

  // ── 4. state survives a round-trip ────────────────────────────────────────
  const back = deserializeState(serializeState(k2))
  assert.deepEqual(Array.from(back.mu), Array.from(k2.mu), 'mu must survive serialisation')
  assert.equal(back.mu.length, d, 'dimension must survive serialisation')

  // ── 5. THE EXPERIMENT: does the profile transfer to an unseen catalogue? ──
  // A different catalogue in the same space — no artist, tag, or id overlap
  // with anything the kernel has ever been shown.
  const foreign = [
    hit('doc-near', member(A, 30), q, 'author-x', ['essay']),
    hit('doc-far',  member(B, 30), q, 'author-y', ['essay']),
  ]
  const fr = [...reweight(foreign, k)].sort((x, y) => y.weight - x.weight)
  assert.equal(fr[0].id, 'doc-near',
    'a profile learned in one catalogue must rank an unseen catalogue by the same geometry')

  return { lean: cos(q, centreA), skip: [beforeB, afterB] as const, margin: fr[0].weight / fr[1].weight }
}

for (const d of [384, D]) {
  const r = run(d)
  console.log(`ok  d=${d}${d === D ? ' (canonical)' : ''}` +
    `  lean=${r.lean.toFixed(3)}  skip=${r.skip[0].toFixed(4)}->${r.skip[1].toFixed(4)}` +
    `  transfer=${r.margin.toFixed(2)}x`)
}
console.log('5 checks passed in each space')

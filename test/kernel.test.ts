/**
 * One runnable check for the kernel, and one experiment.
 *
 * The check: liking moves the query toward what you liked, passing pushes
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
import { emptyKernel, onLike, onPass, queryVector, reweight, serializeState, deserializeState } from '../src/SessionKernel.ts'
import type { Item, Hit } from '../src/types.ts'
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

  const item = (embedding: Float32Array, group: string, tags: string[]): Item => ({
    embedding, group, facets: { kind: tags }, scalar: 200,
  })
  const hit = (id: string, embedding: Float32Array, q: Float32Array, group: string, tags: string[]): Hit => ({
    id, embedding,
    distance: 1 - cos(embedding, q),        // cosine distance, as a vector store would return
    metadata: { group, facets: { kind: tags }, scalar: 200 },
  })

  // ── 1. liking moves the query toward what was liked ─────────────────────
  let k = emptyKernel({ d })
  assert.equal(k.mu.length, d, 'kernel must allocate in the space it was given')
  for (let i = 0; i < 4; i++) k = onLike(k, item(member(A, i), `group-a${i}`, ['ambient']))

  const q = queryVector(k)
  assert.ok(cos(q, centreA) > cos(q, centreB), 'query must lean toward what was liked')
  assert.ok(cos(q, centreA) > 0.5, `query should sit near region A, got ${cos(q, centreA).toFixed(3)}`)

  // ── 2. ranking follows the lean ───────────────────────────────────────────
  const candidates = [
    hit('b0', member(B, 0), q, 'group-b0', ['metal']),
    hit('a9', member(A, 9), q, 'group-a9', ['ambient']),
  ]
  const ranked = [...reweight(candidates, k)].sort((x, y) => y.weight - x.weight)
  assert.equal(ranked[0].id, 'a9', 'a candidate in the liked region must outrank a distant one')

  // ── 3. passing pushes away ───────────────────────────────────────────────
  const beforeB = reweight(candidates, k).find(h => h.id === 'b0')!.weight
  let k2 = k
  for (let i = 0; i < 3; i++) k2 = onPass(k2, item(member(B, i), `group-b${i}`, ['metal']))
  const afterB = reweight(candidates, k2).find(h => h.id === 'b0')!.weight
  assert.ok(afterB < beforeB, `passing on a region must lower its weight (${beforeB.toFixed(4)} -> ${afterB.toFixed(4)})`)

  // ── 4. state survives a round-trip ────────────────────────────────────────
  const back = deserializeState(serializeState(k2))
  assert.deepEqual(Array.from(back.mu), Array.from(k2.mu), 'mu must survive serialisation')
  assert.equal(back.mu.length, d, 'dimension must survive serialisation')

  // ── 5. THE EXPERIMENT: does the profile transfer to an unseen catalogue? ──
  // A different catalogue in the same space — no group, tag, or id overlap
  // with anything the kernel has ever been shown.
  const foreign = [
    hit('doc-near', member(A, 30), q, 'author-x', ['essay']),
    hit('doc-far',  member(B, 30), q, 'author-y', ['essay']),
  ]
  const fr = [...reweight(foreign, k)].sort((x, y) => y.weight - x.weight)
  assert.equal(fr[0].id, 'doc-near',
    'a profile learned in one catalogue must rank an unseen catalogue by the same geometry')

  return { lean: cos(q, centreA), pass: [beforeB, afterB] as const, margin: fr[0].weight / fr[1].weight }
}

for (const d of [384, D]) {
  const r = run(d)
  console.log(`ok  d=${d}${d === D ? ' (canonical)' : ''}` +
    `  lean=${r.lean.toFixed(3)}  pass=${r.pass[0].toFixed(4)}->${r.pass[1].toFixed(4)}` +
    `  transfer=${r.margin.toFixed(2)}x`)
}
console.log('5 checks passed in each space')

// ── the general kernel: no music, no required metadata ──────────────────────
{
  const d = 8
  const e = (i: number) => { const v = zeros(d); v[i] = 1; return v }

  // The first like IS the position; one position has no direction.
  let k = onLike(emptyKernel({ d }), { embedding: e(0) })
  assert.deepEqual(Array.from(k.mu), Array.from(e(0)), 'the first like must be the position, not a tenth of it')
  assert.equal(norm(k.v), 0, 'one like has no heading')
  k = onLike(k, { embedding: e(1) })
  assert.ok(k.v[1] > 0 && k.v[0] < 0, 'the second like gives a heading from the first toward it')

  // A vectors-only item runs the geometric kernel alone: no NaN, no fake groups.
  const w = reweight([{ id: 'x', embedding: e(1), distance: 0.1 }, { id: 'y', embedding: e(2), distance: 0.9 }], k)
  assert.ok(w.every(h => Number.isFinite(h.weight)) && w[0].weight > w[1].weight, 'bare vectors must rank by geometry')
  assert.deepEqual(k.groups, {}, 'no group given, none invented')

  // Passing before any like records the pass but does not move a position that does not exist.
  const p0 = onPass(emptyKernel({ d }), { embedding: e(3) })
  assert.equal(norm(p0.mu), 0, 'nothing liked yet: mu stays put')
  assert.equal(p0.passes.length, 1)

  // Fatigue vs dislike, on whatever a group is.
  let g = onLike(emptyKernel({ d }), { embedding: e(0), group: 'studio' })
  g = onLike(g, { embedding: e(1), group: 'studio' })   // two likes: clear of the 0.3 threshold, not on it
  g = onPass(g, { embedding: e(0), group: 'studio' })
  assert.ok(g.muted.has('studio') && !g.neg.studio, 'passing on a liked group is fatigue, not dislike')
  g = onPass(g, { embedding: e(4), group: 'stranger' })
  assert.equal(g.neg.stranger, 1, 'passing on an unknown group is dislike')

  // The scalar is unit-free: the same items in seconds or milliseconds rank the same.
  const rank = (unit: number) => {
    let s = onLike(emptyKernel({ d }), { embedding: e(0), scalar: 200 * unit })
    s = onLike(s, { embedding: e(0), scalar: 220 * unit })
    return reweight([
      { id: 'near', embedding: e(5), distance: 0.5, metadata: { scalar: 210 * unit } },
      { id: 'far',  embedding: e(5), distance: 0.5, metadata: { scalar: 2000 * unit } },
    ], s).map(h => h.weight)
  }
  const [s1, ms] = [rank(1), rank(1000)]
  assert.ok(s1[0] > s1[1], 'nearer the preferred scalar ranks higher')
  assert.ok(Math.abs(s1[0] - ms[0]) < 1e-9, 'and it does not care about the unit')

  // Facet channels are the caller's.
  let f = onLike(emptyKernel({ d }), { embedding: e(0), facets: { signal: ['momentum'], sizing: ['vol_target'] } })
  const fw = reweight([
    { id: 'match', embedding: e(6), distance: 0.5, metadata: { facets: { signal: ['momentum'] } } },
    { id: 'other', embedding: e(6), distance: 0.5, metadata: { facets: { signal: ['carry'] } } },
  ], f)
  assert.ok(fw[0].weight > fw[1].weight, 'a liked tag in a caller-named channel must rank higher')

  // State saved by 0.0.x still loads.
  const legacy = { mu: [1, 0], v: [0, 0], sigma: 0.5, skips: [[0, 1]], skipCentroid: [0, 1], skipRadius: 0,
    t: 2, entropy: 0.2, taste: { genres: { ambient: 0.3 }, moods: {}, themes: {}, contexts: {} },
    artists: { a: 0.5 }, durationPref: 180000, neg: {}, blacklist: ['b'] }
  const back = deserializeState(legacy as any)
  assert.equal(back.passes.length, 1); assert.equal(back.groups.a, 0.5); assert.equal(back.facets.genres.ambient, 0.3)
  assert.equal(back.scalarPref, null, 'a duration in ms is not a log scalar; drop it rather than misread it')
  assert.ok(back.blacklist.has('b'))
  console.log('ok  general: first like is the position, bare vectors, fatigue vs dislike by group, unit-free scalar, caller facets, 0.0.x state loads')
}

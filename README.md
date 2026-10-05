# Markov

A **markov kernel** for locally storing and encoding user preference as they navigate a higher-dimensional embedding space. It provides a local, portable point-of-view in an embedding space.

It has zero host dependencies and operates agnostically of the underlying embedding model.

## The idea

Your "taste" and the content live in the same space, so we can think of preference as a *measure* within a *measure space*. The Markov kernel tracks a position, a direction of travel, and a spread as you interact with content in a given embedding space:

| | |
|---|---|
| `mu` | where you are - a recency-weighted mean of what you liked |
| `v` | where you're heading - EMA of displacement between likes |
| `sigma` | how wide to cast - grows on passes, shrinks on likes |
| `facets` | leaky accumulators over tags, in whatever channels you name |
| `groups` | signed EMA per group (artist, developer, publisher…): +1 liked, −1 passed |
| `scalarPref` | a preferred value of one positive quantity (a length, a price), unit-free |
| `neg` / `blacklist` | permanent suppression of a group |
| `muted` | session-scoped fatigue, never persisted |

`queryVector(state)` returns `mu` pushed along `v` — you search slightly ahead
of where you are, which is what makes it feel like motion rather than a filter.

`mu` is a bias-corrected EMA: the first like *is* the position, and later likes
settle to weight `alpha`. Velocity starts at the second like, since one position
has no direction.

Passes are disambiguated. Passing on a group you have a positive history with is
read as **fatigue** (mute for the session, preference preserved); passing on one
you don't is read as **dislike** (accumulates toward suppression). Conflating
those two is why most recommenders punish you for your own favourites.

Debt owed to Gärdenfors, *The Geometry of Meaning* — this is a conceptual space
with a walker in it.

## Use

```ts
import { emptyKernel, onLike, onPass, queryVector, reweight } from '@fangorn-network/markov'

let k = emptyKernel({ d: 768 })
k = onLike(k, { embedding })                        // only the embedding is required
k = onLike(k, { embedding, group: 'nolla-games', facets: { genre: ['roguelike'] }, scalar: 12 })
k = onPass(k, { embedding: other })
const hits = await db.search(queryVector(k))        // [{ id, embedding, distance, metadata?: { group, facets, scalar } }]
const ranked = reweight(hits, k).sort((a, b) => b.weight - a.weight)
```

Every optional field adds one signal when it is there and is ignored when it is
not, so a caller with nothing but vectors gets the geometric kernel alone.

`serializeState` / `deserializeState` round-trip it to JSON. It is a small file
on your disk. Nobody else has a copy.

## Check

```sh
npm test          # node test/kernel.test.ts — no deps, Node 22.18+
npm run build     # dist/ — the JS that ships (Node will not strip types under node_modules)
```

Five assertions, run in two spaces: liking leans the query, ranking follows the
lean, passing pushes away, state survives serialisation, and — the interesting
one — a profile built only in catalogue A correctly ranks an **unseen catalogue
B** sharing the space. Current margin on synthetic vectors: **4.1x**. Then the
general cases: bare vectors, fatigue vs dislike by group, a unit-free scalar,
caller-named facet channels, and state saved by 0.0.x still loading.

```
ok  d=384  lean=0.983  pass=0.0074->0.0061  transfer=4.14x
ok  d=768 (canonical)  lean=0.983  pass=0.0074->0.0061  transfer=4.14x
5 checks passed in each space
ok  general: first like is the position, bare vectors, fatigue vs dislike by group, unit-free scalar, caller facets, 0.0.x state loads
```

## The space

The canonical space is **`nomic-embed-text-v1.5`, 768 dims**, chosen because it
is Matryoshka: the same vectors truncate to 512/256/128 for centroids and coarse
scans, which is what lets a catalogue advertize its coverage in kilobytes. It is
Apache-2.0, has ONNX weights for in-browser use, and is English-first — that last
one is the known exposure.

A kernel is only meaningful inside one space. `params.d` sets the dimension and
must match the catalogue's declared `embedding.dim`; two catalogues on different
models are different spaces, and moving a profile between them needs a learned
projection that does not exist yet.

## Known limits

- No projection between spaces. A profile does not survive a change of model.
- The transfer result above is synthetic — clean, well-separated regions. It
  proves the mechanism, not that real cross-domain taste is legible.

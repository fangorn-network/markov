# Markov

A **markov kernel** for locally storing and encoding user preference as they navigate a higher-dimensional embedding space. It provides a local, portable point-of-view in an embedding space.

It has zero host dependencies and operates agnostically of the underlying embedding model.

## The idea

Your "taste" and the content live in the same space, so we can think of preference as a *measure* within a *measure space*. The Markov kernel tracks a position, a direction of travel, and a spread as you interact with content in a given embedding space:

| | |
|---|---|
| `mu` | where you are - EMA of what you played |
| `v` | where you're heading - EMA of displacement |
| `sigma` | how wide to cast - grows on skips, shrinks on plays |
| `taste` | leaky accumulators over tags |
| `artists` | signed EMA: +1 played, −1 skipped |
| `neg` / `blacklist` | permanent suppression |
| `muted` | session-scoped fatigue, never persisted |

`queryVector(state)` returns `mu` pushed along `v` — you search slightly ahead
of where you are, which is what makes it feel like motion rather than a filter.

Skips are disambiguated. Skipping an artist you have a positive history with is
read as **fatigue** (mute for the session, preference preserved); skipping one
you don't is read as **dislike** (accumulates toward suppression). Conflating
those two is why most recommenders punish you for your own favourites.

Debt owed to Gärdenfors, *The Geometry of Meaning* — this is a conceptual space
with a walker in it.

## Use

```ts
import { emptyKernel, onPlay, onSkip, queryVector, reweight } from '@fangorn/kernel'

let k = emptyKernel()
k = onPlay(k, features)              // TrackFeatures: embedding + tags + artist
const hits = await db.search(queryVector(k))
const ranked = reweight(hits, k).sort((a, b) => b.weight - a.weight)
```

`serializeState` / `deserializeState` round-trip it to JSON. It is a small file
on your disk. Nobody else has a copy.

## Check

```sh
npm test          # node test/kernel.test.ts — no deps, Node 22.18+
```

Five assertions, run in two spaces: playing leans the query, ranking follows the
lean, skipping pushes away, state survives serialisation, and — the interesting
one — a profile built only in catalogue A correctly ranks an **unseen catalogue
B** sharing the space. Current margin on synthetic vectors: **4.1x**.

```
ok  d=384            lean=0.983  skip=0.0204->0.0144  transfer=4.14x
ok  d=768 (canonical) lean=0.983  skip=0.0204->0.0144  transfer=4.14x
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
- Tag roles (`genres/moods/themes/contexts`) are music-shaped names for what are
  really four generic facet channels.

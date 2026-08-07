# Cyber Scout — the web app

The same check, in a browser. Paste a text message and it tells you whether it looks
like a scam. **The message never leaves the device** — the model runs in WebAssembly on
the phone or laptop that's reading it.

It exists because iOS has no free distribution path: TestFlight, ad-hoc and the App
Store all require the $99/yr Apple Developer Program. A web app has no gatekeeper, no
review, no expiry, and works on Android too. Since iOS was already the weaker build
(paste-only, no share extension), it gives up very little there.

```
index.html              the one screen
src/main.ts             wires onnxruntime-web into the shared Scanner
src/assets.ts           fetch + Cache API, with progress
src/ui.ts               rendering; every sentence comes from ../core/copy.ts
src/styles.css          theme carried over from the phone app
public/manifest.webmanifest, public/sw.js, public/icons/
scripts/prepare-assets.mjs   stages the exported models into public/
scripts/make_icons.py        draws the icons
test/pipeline.ts        the whole pipeline through WASM vs Python
```

## Running it

```bash
npm install
npm run dev            # stages assets, then serves on localhost
npm test               # verify before deploying
npm run build          # -> dist/
```

`prepare-assets` copies the exported models from `../app/assets/models`. It copies
rather than re-exports, so the phone and the web app provably run the same bytes. If it
complains the models are missing:

```bash
cd ../app-backend && python export_onnx.py --arm minilm_feat
```

## The pipeline is not written here

`src/` holds a screen and a download manager. The actual work — tokenizer, the 29
engineered features, the two heads — is `../core/`, byte-identical to what the phone
app runs. `core/model.ts` takes the ONNX runtime as an argument, which is the seam that
lets one copy of it serve `onnxruntime-react-native`, `onnxruntime-node` and
`onnxruntime-web`.

The user-facing wording is shared too, in `../core/copy.ts`. That's deliberate: the
phrasing is a product decision for a vulnerable audience, and two codebases drifting
into telling people different things about the same message is a real failure mode.

## Tests

```bash
npm test
```

160 messages through the real graphs under the real WASM runtime, compared against the
probability Python recorded. Bar is zero decision flips.

This is a different question from `app/test/pipeline.ts`, which covers the same code
under `onnxruntime-node`. WASM is a separate kernel implementation, and a backend where
an unsupported operator can degrade quietly rather than refuse to load. Last run:

```
160 messages, max|delta| 0.00327, decision flips 0, 38 ms/message
```

One thing it does **not** cover, and it bit: the test hands ORT the WASM bytes directly,
so it never exercises **URL resolution**. Both bugs found on first run in a real browser
were resolution bugs the test could not have seen — see "Do not touch the ORT runtime
paths" below. Browser-only concerns need a browser.

## What the user downloads

| | size | why |
|---|---|---|
| encoder | 86 MB | MiniLM, 22.7M params × 4 bytes |
| ORT WASM runtime | 12.9 MB (~3.5 MB gzipped) | the kernels |
| heads, vocab, JSON | 1.1 MB | |
| app itself | 87 KB | |

Once, then cached. **fp32, not int8** — int8 flipped a genuine bank security notice
from 0.032 to 0.943, because LightGBM's hard split boundaries amplify quantization
noise. fp16 would halve the encoder and is much safer than int8, but it is untested
here and ORT's WASM backend has patchy fp16 kernel coverage; see the note in
`../app-backend/export_onnx.py`.

## Deploying

The app is static. The **86 MB encoder is the constraint** — Cloudflare Pages caps a
file at 25 MB, so it usually cannot sit next to the app:

```bash
VITE_MODEL_BASE=https://huggingface.co/<user>/<repo>/resolve/main npm run build
```

Setting it changes what the build contains, not just where it points:

| | `dist/` | goes to |
|---|---|---|
| default | 100 MB | one host that accepts an 86 MB file |
| `VITE_MODEL_BASE` set | **13 MB** | any static host; models live at that URL |

The models are dropped from the build rather than shipped and ignored — otherwise the
deploy carries 87 MB nothing requests, and fails outright on a host with a file cap.

Hugging Face is free, built for model files and sends CORS headers. GitHub Pages can
host the 86 MB file directly (100 MB per-file limit) but it is an unhappy fit for a
git repo. Set `VITE_BASE=/repo-name/` for a GitHub Pages project site.

Whatever serves the WASM must send `Content-Type: application/wasm`, or the browser
falls back from streaming compilation and startup gets noticeably slower.

## Do not touch the ORT runtime paths

Both bugs found the first time this ran in a browser were about *locating* the ONNX
Runtime, not running it. Neither could fail in `npm test`, which hands ORT the bytes
directly. Two rules, and the reasoning, so they are not undone:

**Do not set `env.wasm.wasmPaths`.** It looks like the right API — it is what ORT's own
docs suggest for a known location — but onnxruntime-web `import()`s its Emscripten glue
as a *module*, not as a fetch. Point it at a copy in `public/` and Vite refuses to serve
it, because `public/` is copied verbatim and never goes through module resolution. The
dev server dies before anything loads. Vite already emits the binary from the
`new URL(..., import.meta.url)` inside ORT; let it.

**Keep `optimizeDeps.exclude: ['onnxruntime-web']`.** Vite pre-bundles dependencies into
`node_modules/.vite/deps/`, which rewrites `import.meta.url` — so ORT's relative path to
its binary resolves next to the pre-bundled copy, where no binary exists. The dev server
answers that 404 with `index.html`, and the browser reports:

```
CompileError: WebAssembly.instantiate(): expected magic word 00 61 73 6d,
found 3c 21 64 6f
```

`3c 21 64 6f` is `<!do`. **A WASM magic-word error is a 404 serving HTML**, not a corrupt
binary — worth knowing before spending an afternoon on the model file.

**Optional speed-up.** `main.ts` asks for 4 threads only when `crossOriginIsolated` is
true, which needs the host to send:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them it runs single-threaded, which is the default assumption — asking for
threads without the headers makes ORT fail rather than degrade.

## iOS, and the thing that will actually bite

Safari evicts script-writable storage after about **seven days of non-use** — unless
the site was added to the home screen. A tester who skips that step and comes back a
fortnight later re-downloads 86 MB.

So Add to Home Screen isn't a nicety here, and `src/ui.ts` shows an instruction card
explaining it on iOS Safari specifically. Expect to walk people through it. Also note
iOS Safari does not implement Web Share Target, so `share_target` in the manifest works
on Android only and the paste box is the primary control, not a fallback.

## What it does not catch

Same coverage hole as the phone app: `manifest.types_not_covered` lists four of the
thirteen scam types with no training rows — family emergency, charity, Medicare and
health, utility shutoff. "Hi Mum, this is my new number" scores 0.026. Every safe
verdict says so rather than reading as an all-clear. `../labeling/` is the fix.

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
scripts/prepare-assets.mjs   stages models and the ORT runtime into public/
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

`prepare-assets` copies the exported models from `../app/assets/models` and the ORT
WASM binary out of `node_modules`. It copies rather than re-exports, so the phone and
the web app provably run the same bytes. If it complains the models are missing:

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

Two things it does **not** cover, both needing a real browser: whether the WASM binary
resolves from `env.wasm.wasmPaths`, and whether the Cache API actually keeps the 86 MB
encoder between visits.

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

Hugging Face is free, built for model files and sends CORS headers. GitHub Pages can
host the 86 MB file directly (100 MB per-file limit) but it is an unhappy fit for a
git repo. Set `VITE_BASE=/repo-name/` for a GitHub Pages project site.

Whatever serves the WASM must send `Content-Type: application/wasm`, or the browser
falls back from streaming compilation and startup gets noticeably slower.

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

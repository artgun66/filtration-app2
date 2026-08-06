# Cyber Scout — the phone app

Paste a text message, or share one from your messages app, and it tells you whether it
looks like a scam. **Everything runs on the phone.** No message leaves the device,
which is why the app asks for no permissions at all.

```
App.tsx                 paste box, share handling, loading state
src/model.ts            the pipeline: tokenize -> encode -> features -> heads
src/features.ts         the 29 engineered features, ported from dataset/features.py
src/tokenizer.ts        WordPiece, ported
src/assets.ts           bundled models -> paths onnxruntime can open
src/ui/                 theme and the result card
assets/models/          written by ../app-backend/export_onnx.py
test/golden.ts          features + tokenizer vs Python
test/pipeline.ts        the whole pipeline through real ONNX vs Python
```

## Running it

Not Expo Go — `onnxruntime-react-native` is a native module, so this needs a dev
build. That means a JDK and the Android SDK (installing Android Studio gives both).

```bash
npm install
npm test                       # verify before building; runs on any machine

npx expo prebuild --platform android
npx expo run:android           # device or emulator
```

No Android SDK? Build the APK in the cloud instead:

```bash
npx eas build --platform android --profile preview
```

The `preview` profile emits an **APK**, which a phone can install directly. The
default production profile emits an AAB, which only the Play Store can take.

## Tests

```bash
npm test
```

Two of them, and both matter more than they look:

| test | what it proves |
|---|---|
| `test/golden.ts` | the TypeScript features and tokenizer match Python exactly — 10,720 feature values and 160 token sequences |
| `test/pipeline.ts` | the app's own code, through the real ONNX graphs, decides every message the same way Python does |

The 29 features exist three times over — Python, ONNX, TypeScript — and every way they
can disagree is silent. A drifted regex or a transposed row does not throw; it moves
the probability and the app keeps answering confidently. Places the two languages
actually differed during the port are documented at the top of `src/features.ts`.

`test/pipeline.ts` runs the same `src/model.ts` the phone runs, with
`onnxruntime-node` swapped in for `onnxruntime-react-native` — the runtime is injected
for exactly that reason. It catches integration bugs (wrong output index, off-by-one
in `buildRow`) on a laptop, before a device is involved.

## What it does not catch

`manifest.types_not_covered` lists four of the thirteen scam types with **no training
rows in the corpus**: family emergency, charity, Medicare and health, utility shutoff.

The classic "Hi Mum, this is my new number" message currently scores 0.026 — safe. The
result card says so on every safe verdict rather than reading as an all-clear, because
an app for older adults must not imply coverage it does not have. Closing this is
`../labeling/`'s job, not a tuning problem.

## Notes

- `metro.config.js` adds `onnx` and `txt` to `assetExts`. Without it Metro silently
  refuses to bundle the model and every `require` of it fails at runtime.
- The encoder is **fp32, not int8**, on purpose — int8 flipped a genuine bank security
  notice from 0.032 to 0.943. See the comment on `SHIPPED_ENCODER` in
  `../app-backend/export_onnx.py`.
- Cold start unpacks 86 MB out of the APK, which is why there is a loading state.
  After that a message takes ~15 ms on a laptop; a phone will be slower.
- iOS gets the paste box; the share extension is not built yet. Android gets both.

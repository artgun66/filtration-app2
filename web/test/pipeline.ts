/**
 * The web build's pipeline, through onnxruntime-web's WASM backend.
 *
 *   npm test        (from web/)
 *
 * app/test/pipeline.ts already proves core/ agrees with Python under onnxruntime-node.
 * This asks the separate question the web build introduces: does the *WASM* runtime run
 * these graphs the same way the native one does? Different kernel implementations, a
 * different threading model, and a backend where unsupported operators degrade quietly
 * rather than refusing to load -- none of which the native test can see.
 *
 * It runs under Node rather than a browser because the thing being tested is the
 * runtime and the graphs, not the DOM. Two browser-only concerns are therefore *not*
 * covered here and need a real device: whether the WASM binary resolves from
 * env.wasm.wasmPaths, and whether the Cache API keeps the 86 MB encoder across visits.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import * as ort from 'onnxruntime-web/wasm';

import { loadVocab } from '../../core/tokenizer.ts';
import { Scanner, type Manifest, type OrtLike } from '../../core/model.ts';
import type { FeatureConfig } from '../../core/features.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, '..');
const ASSETS = join(WEB, '..', 'app', 'assets', 'models');
const read = (f: string) => JSON.parse(readFileSync(join(ASSETS, f), 'utf-8'));

// In a browser ORT fetches this over HTTP from env.wasm.wasmPaths. Node has no fetch
// for file:// URLs, so hand it the bytes directly -- same binary either way.
ort.env.wasm.wasmBinary = readFileSync(
  join(WEB, 'node_modules', 'onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.wasm'),
).buffer as ArrayBuffer;
ort.env.wasm.numThreads = 1;
ort.env.logLevel = 'error';

const manifest: Manifest = read('manifest.json');
const cfg: FeatureConfig = read('feature_config.json');
const golden = read('golden_vectors.json');
const vocab = loadVocab(read('vocab.json') as string[]);

const t0 = Date.now();
const scanner = await Scanner.create({
  ort: ort as unknown as OrtLike,
  manifest,
  cfg,
  vocab,
  // The browser hands over bytes from the Cache API rather than a path, so the test
  // exercises that route too.
  resolve: (f) => new Uint8Array(readFileSync(join(ASSETS, f))),
});
console.log(`sessions ready in ${((Date.now() - t0) / 1000).toFixed(1)}s ` +
            `(${manifest.encoder_file}, ${manifest.feature_names.length} features, wasm)`);

let maxDelta = 0;
let flips = 0;
let sum = 0;
const started = Date.now();

for (const [i, c] of golden.cases.entries()) {
  const v = await scanner.scan(c.text);
  const d = Math.abs(v.prob - c.prob);
  maxDelta = Math.max(maxDelta, d);
  sum += d;
  if ((v.prob >= manifest.threshold) !== (c.prob >= manifest.threshold)) {
    flips++;
    if (flips <= 3) {
      console.error(`case ${i} flipped: python ${c.prob.toFixed(4)} ` +
        `wasm ${v.prob.toFixed(4)} (threshold ${manifest.threshold.toFixed(4)})\n` +
        `   "${c.text.slice(0, 70).replace(/\n/g, ' ')}"`);
    }
  }
}

const n = golden.cases.length;
const ms = (Date.now() - started) / n;
console.log(`\n${n} messages through the web pipeline`);
console.log(`  max|delta| ${maxDelta.toFixed(5)}   mean ${(sum / n).toFixed(7)}   ` +
            `decision flips ${flips}`);
console.log(`  ${ms.toFixed(0)} ms/message under WASM on this machine`);

if (flips === 0) {
  console.log('PASS -- onnxruntime-web decides every message the way Python does');
} else {
  console.log(`FAIL -- ${flips} of ${n} messages decided differently`);
  process.exit(1);
}

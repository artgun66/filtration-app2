/**
 * End-to-end: the app's own TypeScript, through the real ONNX graphs.
 *
 *   node --experimental-strip-types app/test/pipeline.ts
 *
 * golden.ts proves the features and tokens are right. export_onnx.py --check proves
 * the graphs are right -- but it feeds them *Python's* features. Neither covers the
 * seam between them, which is where an integration bug lives: a transposed row, the
 * wrong output index, an off-by-one in buildRow. This runs src/model.ts exactly as
 * the phone will, under onnxruntime-node, and compares against the probability Python
 * recorded for the same message.
 *
 * The bar is decision agreement, as everywhere else in this project. fp32 ONNX and
 * PyTorch differ in the last few decimals for reasons no product decision depends on.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

import { loadVocab } from '../src/tokenizer.ts';
import { Scanner, type Manifest, type OrtLike } from '../src/model.ts';
import type { FeatureConfig } from '../src/features.ts';

const require = createRequire(import.meta.url);
const ort = require('onnxruntime-node') as OrtLike;

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = join(HERE, '..', 'assets', 'models');
const read = (f: string) => JSON.parse(readFileSync(join(ASSETS, f), 'utf-8'));

const manifest: Manifest = read('manifest.json');
const cfg: FeatureConfig = read('feature_config.json');
const golden = read('golden_vectors.json');
const vocab = loadVocab(read('vocab.json') as string[]);

const t0 = Date.now();
const scanner = await Scanner.create({
  ort, manifest, cfg, vocab,
  resolve: (f) => join(ASSETS, f),
});
console.log(`sessions ready in ${((Date.now() - t0) / 1000).toFixed(1)}s ` +
            `(${manifest.encoder_file}, ${manifest.feature_names.length} features)`);

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
        `typescript ${v.prob.toFixed(4)} (threshold ${manifest.threshold.toFixed(4)})\n` +
        `   "${c.text.slice(0, 70).replace(/\n/g, ' ')}"`);
    }
  }
}

const n = golden.cases.length;
const ms = (Date.now() - started) / n;
console.log(`\n${n} messages through the full pipeline`);
console.log(`  max|delta| ${maxDelta.toFixed(5)}   mean ${(sum / n).toFixed(7)}   ` +
            `decision flips ${flips}`);
console.log(`  ${ms.toFixed(0)} ms/message on this machine (a phone will be slower)`);

if (flips === 0) {
  console.log('PASS -- the app pipeline agrees with Python on every message');
} else {
  console.log(`FAIL -- ${flips} of ${n} messages decided differently`);
  process.exit(1);
}

/**
 * Conformance test: the TypeScript features must equal the Python ones exactly.
 *
 *   node --experimental-strip-types app/test/golden.ts
 *
 * Every case comes from serving/app_assets/golden_vectors.json, written by
 * serving/export_onnx.py straight out of dataset/features.py. A mismatch here means
 * the app is computing a different input than the model was trained on -- which
 * produces wrong answers, not errors, so this is the only thing standing between a
 * regex subtlety and a silently broken app.
 *
 * Floats are compared at 1e-6, the precision the fixture is rounded to. Integer
 * features must match exactly.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { extractFeatures, keywordFlags, type FeatureConfig } from '../src/features.ts';
import { loadVocab, encode } from '../src/tokenizer.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ASSETS = join(HERE, '..', 'assets', 'models');

const read = (f: string) => JSON.parse(readFileSync(join(ASSETS, f), 'utf-8'));

const cfg: FeatureConfig = read('feature_config.json');
const golden = read('golden_vectors.json');
const manifest = read('manifest.json');
// vocab.json, not vocab.txt: this is the form the app loads, so it is the form the
// test should exercise.
const vocab = loadVocab(read('vocab.json') as string[]);

const TOL = 1e-6;
let failures = 0;
let checked = 0;
const byFeature = new Map<string, number>();

for (const [i, c] of golden.cases.entries()) {
  const got = extractFeatures(c.text, cfg);
  const kws = keywordFlags(c.text, cfg);

  for (const [name, want] of Object.entries({ ...c.features, ...c.keywords })) {
    const mine = name in got ? got[name] : kws[name];
    checked++;
    const ok = mine !== undefined && Math.abs(mine - (want as number)) <= TOL;
    if (!ok) {
      byFeature.set(name, (byFeature.get(name) ?? 0) + 1);
      if (failures < 10) {
        console.error(
          `case ${i} "${c.text.slice(0, 60).replace(/\n/g, ' ')}"\n` +
          `   ${name}: python ${want}  typescript ${mine}`);
      }
      failures++;
    }
  }
}

// --- tokenizer: the other half of the model's input
let tokMismatch = 0;
let tokChecked = 0;
for (const [i, c] of golden.cases.entries()) {
  if (!c.input_ids) continue;
  tokChecked++;
  const mine = encode(c.text, vocab, manifest.max_seq_len).inputIds;
  const same = mine.length === c.input_ids.length &&
    mine.every((v, k) => v === c.input_ids[k]);
  if (!same) {
    if (tokMismatch < 3) {
      const at = mine.findIndex((v, k) => v !== c.input_ids[k]);
      console.error(
        `case ${i} tokens differ at ${at} (len ${mine.length} vs ${c.input_ids.length})\n` +
        `   "${c.text.slice(0, 60).replace(/\n/g, ' ')}"\n` +
        `   python ${c.input_ids.slice(Math.max(0, at - 2), at + 3)}\n` +
        `   typescript ${mine.slice(Math.max(0, at - 2), at + 3)}`);
    }
    tokMismatch++;
  }
}

console.log(`\n${golden.cases.length} cases, ${checked} feature values checked`);
if (tokChecked) {
  console.log(`${tokChecked} token sequences checked, ${tokMismatch} mismatched`);
}
if (failures === 0 && tokMismatch === 0) {
  console.log('PASS -- TypeScript features and tokenizer match Python exactly');
} else if (failures === 0) {
  console.log(`FAIL -- features match but ${tokMismatch} token sequences differ`);
  process.exit(1);
} else {
  console.log(`FAIL -- ${failures} mismatches`);
  console.log('worst features: ' + [...byFeature.entries()]
    .sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([n, c]) => `${n} (${c})`).join(', '));
  process.exit(1);
}

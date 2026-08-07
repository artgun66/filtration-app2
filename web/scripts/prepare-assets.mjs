/**
 * Stage everything the browser has to fetch into public/.
 *
 * Two sources, neither of which belongs in this repo twice:
 *   - the exported models, written by ../../app-backend/export_onnx.py into
 *     ../app/assets/models. The 86 MB encoder is gitignored there and stays gitignored
 *     here; this copies rather than re-exports so the phone and the web app are
 *     provably running the same bytes.
 *   - onnxruntime-web's WASM runtime, which Vite will not emit on its own because ORT
 *     loads it by URL at runtime rather than importing it.
 *
 * Copies are skipped when the destination already matches by size, because the encoder
 * is 86 MB and every dev server start would otherwise pay for it.
 */
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, '..');
const MODELS_SRC = join(WEB, '..', 'app', 'assets', 'models');
const ORT_SRC = join(WEB, 'node_modules', 'onnxruntime-web', 'dist');

// golden_vectors.json and the tokenizer_*.json are test and provenance artifacts; the
// browser never needs them, and one of them is 338 KB.
const MODELS = [
  'manifest.json',
  'feature_config.json',
  'vocab.json',
  'encoder_fp32.onnx',
  'scam_head.onnx',
  'type_head.onnx',
];

// The single-threaded SIMD build. The threaded path needs SharedArrayBuffer, which
// needs COOP/COEP headers, which static hosts generally will not set -- see README.
const ORT = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];

function stage(src, dstDir, names, label) {
  mkdirSync(dstDir, { recursive: true });
  let copied = 0;
  let bytes = 0;
  for (const name of names) {
    const from = join(src, name);
    const to = join(dstDir, name);
    if (!existsSync(from)) {
      throw new Error(
        `missing ${from}\n` +
        (label === 'models'
          ? '  run: cd ../app-backend && python export_onnx.py --arm minilm_feat'
          : '  run: npm install'));
    }
    const size = statSync(from).size;
    bytes += size;
    if (existsSync(to) && statSync(to).size === size) continue;
    copyFileSync(from, to);
    copied += 1;
  }
  const mb = (bytes / 1024 / 1024).toFixed(1);
  console.log(`${label}: ${names.length} files, ${mb} MB (${copied} copied, ` +
              `${names.length - copied} already current)`);
}

stage(MODELS_SRC, join(WEB, 'public', 'models'), MODELS, 'models');
stage(ORT_SRC, join(WEB, 'public', 'ort'), ORT, 'ort');

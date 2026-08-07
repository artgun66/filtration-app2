/**
 * Stage the exported models into public/.
 *
 * They are written by ../../app-backend/export_onnx.py into ../app/assets/models. The
 * 86 MB encoder is gitignored there and stays gitignored here; this copies rather than
 * re-exports, so the phone and the web app provably run the same bytes.
 *
 * Copies are skipped when the destination already matches by size, because the encoder
 * is 86 MB and every dev server start would otherwise pay for it.
 *
 * The ONNX Runtime WASM binary is deliberately NOT staged here. Vite emits it from the
 * `new URL(..., import.meta.url)` inside onnxruntime-web; a copy in public/ paired with
 * env.wasm.wasmPaths breaks the dev server, because ORT imports its Emscripten glue as
 * a module and Vite will not serve public/ as source.
 */
import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, '..');
const MODELS_SRC = join(WEB, '..', 'app', 'assets', 'models');

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
        '  run: cd ../app-backend && python export_onnx.py --arm minilm_feat');
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

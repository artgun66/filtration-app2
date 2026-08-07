/**
 * Bootstrap: wire onnxruntime-web into the shared Scanner and drive one screen.
 *
 * The pipeline itself is ../../core -- byte-identical to what the phone app runs. The
 * only thing this file contributes is the browser's answers to "where do the models
 * come from" and "what does the runtime look like", which is exactly the seam
 * core/model.ts leaves open.
 */
import * as ort from 'onnxruntime-web/wasm';

import { Scanner, type OrtLike, type Verdict } from '../../core/model.ts';
import { loadVocab } from '../../core/tokenizer.ts';
import { loadBundle, alreadyCached } from './assets.ts';
import { resultCard, progressPanel, errorPanel, installHint } from './ui.ts';
import './styles.css';

// env.wasm.wasmPaths is deliberately NOT set. onnxruntime-web reaches its runtime two
// ways: it `import()`s the Emscripten glue as a module, and locates the binary through
// `new URL(..., import.meta.url)`. Vite rewrites both and emits the .wasm itself, so
// pointing wasmPaths at a copy in public/ actively breaks the dev server -- Vite
// refuses to serve public/ as source, and the import fails before anything loads.
// Let the bundler resolve it.

// Multi-threading needs SharedArrayBuffer, which needs COOP/COEP response headers,
// which most static hosts do not send. crossOriginIsolated tells us whether we
// actually got them; asking for threads without it makes ORT fail rather than fall
// back. See README for the headers that unlock this.
ort.env.wasm.numThreads = globalThis.crossOriginIsolated ? 4 : 1;

const app = document.getElementById('app')!;
const form = document.getElementById('check') as HTMLFormElement;
const input = document.getElementById('message') as HTMLTextAreaElement;
const button = document.getElementById('submit') as HTMLButtonElement;

function show(node: HTMLElement | null): void {
  app.replaceChildren(...(node ? [node] : []));
}

async function boot(): Promise<Scanner> {
  const cached = await alreadyCached();
  const panel = progressPanel();
  show(panel.node);
  if (!cached) {
    button.disabled = true;
    input.disabled = true;
  }

  const bundle = await loadBundle(panel.update);
  const scanner = await Scanner.create({
    ort: ort as unknown as OrtLike,
    resolve: bundle.resolve,
    manifest: bundle.manifest,
    cfg: bundle.cfg,
    vocab: loadVocab(bundle.vocabTokens),
  });

  button.disabled = false;
  input.disabled = false;
  show(installHint());
  return scanner;
}

const ready = boot().catch((err) => {
  console.error(err);
  show(errorPanel(err));
  throw err;
});

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const text = input.value.trim();
  if (!text) return;

  button.disabled = true;
  button.textContent = 'Checking…';
  try {
    const scanner = await ready;
    const verdict: Verdict = await scanner.scan(text);
    show(resultCard(verdict));
  } catch (err) {
    console.error(err);
    show(errorPanel(err));
  } finally {
    button.disabled = false;
    button.textContent = 'Check this message';
  }
});

// Shared into the app from another site or the OS share sheet, where supported. iOS
// Safari does not implement Web Share Target, so on iPhone this never fires and the
// paste box is the only route -- which is why the paste box is the primary control
// rather than a fallback.
const shared = new URLSearchParams(location.search).get('text');
if (shared) {
  input.value = shared;
  form.requestSubmit();
}

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  addEventListener('load', () => {
    navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`)
      .catch((err) => console.warn('service worker registration failed', err));
  });
}

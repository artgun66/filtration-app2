/**
 * Fetching the models, and keeping them.
 *
 * The phone bundles 87 MB inside the APK. A browser cannot, so the same bytes arrive
 * over the network on first run and live in the Cache API afterwards. Two consequences
 * shape this file:
 *
 *   - The first load is long enough to need a progress bar, so downloads are streamed
 *     and report bytes as they arrive rather than resolving in one silent lump.
 *   - The cache is the whole user experience. Safari evicts script-writable storage
 *     after seven days of non-use *unless* the site was added to the home screen,
 *     which is why the UI pushes that so hard. `cachedBytes` is what turns a repeat
 *     visit into an instant one.
 *
 * The model host is configurable because 86 MB exceeds what several static hosts will
 * serve -- Cloudflare Pages caps a file at 25 MB -- so production points VITE_MODEL_BASE
 * at Hugging Face while the app itself is served from anywhere.
 */
import type { Manifest, ModelSource } from '../../core/model.ts';
import type { FeatureConfig } from '../../core/features.ts';

const BASE: string = (import.meta.env.VITE_MODEL_BASE ?? '/models').replace(/\/$/, '');
const CACHE = 'cyber-scout-models-v1';

export type Progress = (file: string, received: number, total: number) => void;

function url(file: string): string {
  return `${BASE}/${file}`;
}

/** Bytes for `file`, from the cache when it is there and from the network when not. */
async function cachedBytes(file: string, onProgress?: Progress): Promise<Uint8Array> {
  const target = url(file);
  // caches is absent on plain http:// origins other than localhost. Everything still
  // works, it just re-downloads every visit -- which is a reason to serve over HTTPS,
  // not a reason to fail.
  const cache = 'caches' in globalThis ? await caches.open(CACHE) : null;

  const hit = await cache?.match(target);
  if (hit) {
    const buf = new Uint8Array(await hit.arrayBuffer());
    onProgress?.(file, buf.length, buf.length);
    return buf;
  }

  const res = await fetch(target);
  if (!res.ok) throw new Error(`${target} -> ${res.status} ${res.statusText}`);

  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body?.getReader();
  if (!reader) {                       // no streaming support: fall back to one lump
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(file, buf.length, buf.length);
    await cache?.put(target, new Response(buf));
    return buf;
  }

  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(file, received, total);
  }

  const bytes = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) {
    bytes.set(c, at);
    at += c.length;
  }
  await cache?.put(target, new Response(bytes));
  return bytes;
}

async function json<T>(file: string, onProgress?: Progress): Promise<T> {
  const bytes = await cachedBytes(file, onProgress);
  return JSON.parse(new TextDecoder().decode(bytes)) as T;
}

export type Bundle = {
  manifest: Manifest;
  cfg: FeatureConfig;
  vocabTokens: string[];
  resolve: (file: string) => Promise<ModelSource>;
};

/**
 * Everything Scanner.create needs. The graphs are handed over as bytes rather than
 * URLs so onnxruntime reads them out of the Cache API instead of going back to the
 * network -- which is the difference between an instant second visit and another 86 MB.
 */
export async function loadBundle(onProgress?: Progress): Promise<Bundle> {
  const manifest = await json<Manifest>('manifest.json', onProgress);
  const cfg = await json<FeatureConfig>('feature_config.json', onProgress);
  const vocabTokens = await json<string[]>('vocab.json', onProgress);
  return {
    manifest,
    cfg,
    vocabTokens,
    resolve: (file) => cachedBytes(file, onProgress),
  };
}

/** Files the loading screen knows about ahead of time, so it can show real progress. */
export function plannedFiles(manifest: Manifest | null): string[] {
  const graphs = manifest?.encoder ? [manifest.encoder_file] : [];
  return [...graphs, 'scam_head.onnx', 'type_head.onnx'];
}

/** Roughly how much is left to fetch, for the "this is a big download" warning. */
export async function alreadyCached(): Promise<boolean> {
  if (!('caches' in globalThis)) return false;
  const cache = await caches.open(CACHE);
  return Boolean(await cache.match(url('encoder_fp32.onnx')));
}

/**
 * Fetching the models, and keeping them.
 *
 * The phone bundles 87 MB inside the APK. A browser cannot, so the same bytes arrive
 * over the network on first run and are stored locally afterwards. Two consequences
 * shape this file:
 *
 *   - The first load is long enough to need a progress bar, so downloads are streamed
 *     and report bytes as they arrive rather than resolving in one silent lump.
 *   - The store is the whole user experience. Safari evicts script-writable storage
 *     after seven days of non-use *unless* the site was added to the home screen,
 *     which is why the UI pushes that so hard. `cachedBytes` is what turns a repeat
 *     visit into an instant one.
 *
 * IndexedDB, not the Cache API. The Cache API is restricted to secure contexts, so on
 * a plain-http origin -- which is exactly what testing on a phone over the LAN looks
 * like -- it is simply absent and every visit re-downloads 87 MB. IndexedDB carries no
 * such restriction, works identically once deployed over HTTPS, and means the caching
 * path is the same one in testing and in production rather than only existing in the
 * environment that is hardest to check.
 *
 * The model host is configurable because 86 MB exceeds what several static hosts will
 * serve -- Cloudflare Pages caps a file at 25 MB -- so production points VITE_MODEL_BASE
 * at Hugging Face while the app itself is served from anywhere.
 */
import type { Manifest, ModelSource } from '../../core/model.ts';
import type { FeatureConfig } from '../../core/features.ts';

const BASE: string = (import.meta.env.VITE_MODEL_BASE ?? '/models').replace(/\/$/, '');
const DB_NAME = 'cyber-scout';
const STORE = 'models';

export type Progress = (file: string, received: number, total: number) => void;

function url(file: string): string {
  return `${BASE}/${file}`;
}

/** Resolves to null rather than throwing: private browsing can refuse to open a DB. */
function openDb(): Promise<IDBDatabase | null> {
  if (!('indexedDB' in globalThis)) return Promise.resolve(null);
  return new Promise((resolve) => {
    let req: IDBOpenDBRequest;
    try {
      req = indexedDB.open(DB_NAME, 1);
    } catch {
      resolve(null);
      return;
    }
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains(STORE)) req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
}

function idbGet(db: IDBDatabase, key: string): Promise<Blob | undefined> {
  return new Promise((resolve) => {
    const req = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result as Blob | undefined);
    req.onerror = () => resolve(undefined);
  });
}

/**
 * Stored as a Blob, not an ArrayBuffer: the browser can keep a Blob on disk instead of
 * holding 86 MB in the heap through the write, which matters on a phone.
 */
function idbPut(db: IDBDatabase, key: string, value: Blob): Promise<void> {
  return new Promise((resolve) => {
    let tx: IDBTransaction;
    try {
      tx = db.transaction(STORE, 'readwrite');
    } catch {
      resolve();
      return;
    }
    tx.objectStore(STORE).put(value, key);
    // A quota failure must not break the app -- it just means a slower next visit.
    tx.oncomplete = () => resolve();
    tx.onerror = () => resolve();
    tx.onabort = () => resolve();
  });
}

/** Bytes for `file`, from the store when it is there and from the network when not. */
async function cachedBytes(file: string, onProgress?: Progress): Promise<Uint8Array> {
  const target = url(file);
  const db = await openDb();

  if (db) {
    const hit = await idbGet(db, target);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(file, buf.length, buf.length);
      return buf;
    }
  }

  const res = await fetch(target);
  if (!res.ok) throw new Error(`${target} -> ${res.status} ${res.statusText}`);

  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body?.getReader();
  if (!reader) {                       // no streaming support: fall back to one lump
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(file, buf.length, buf.length);
    if (db) await idbPut(db, target, new Blob([buf]));
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
  if (db) await idbPut(db, target, new Blob([bytes]));
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

export type StorageState =
  | 'stored'        // the encoder is already here; opening should be quick
  | 'empty'         // storage works, nothing in it yet -- expect the big download
  | 'unavailable';  // no IndexedDB at all, so every visit will re-download

/**
 * Whether the big download has already happened.
 *
 * Distinguishes "nothing stored yet" from "storage does not work", because they look
 * identical to a user -- a progress bar on every visit -- and have completely different
 * causes. Private browsing is the usual reason for `unavailable`.
 */
export async function storageState(): Promise<StorageState> {
  const db = await openDb();
  if (!db) return 'unavailable';
  return (await idbGet(db, url('encoder_fp32.onnx'))) ? 'stored' : 'empty';
}

/**
 * Offline support for the shell. The models are deliberately not its problem.
 *
 * src/assets.ts already stores the graphs in its own Cache API bucket, keyed by URL and
 * streamed with a progress bar. If this worker also cached /models/ the browser would
 * hold two copies of an 86 MB file, which on iOS is a fast route to being evicted for
 * using too much space. So model requests are passed straight through and the worker
 * only looks after the things that make the page open at all.
 *
 * Runtime caching rather than a precache list, because Vite fingerprints its output and
 * a hand-maintained list of hashed filenames goes stale silently.
 */
const SHELL = 'cyber-scout-shell-v1';

// Stable paths, safe to name ahead of time. Everything else is cached as it is asked for.
const CORE = ['./', './index.html', './manifest.webmanifest'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL)
      // Individually, so one missing icon cannot fail the whole install.
      .then((cache) => Promise.allSettled(CORE.map((url) => cache.add(url))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k.startsWith('cyber-scout-shell-') && k !== SHELL)
            .map((k) => caches.delete(k)),
      ))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== location.origin) return;
  // Owned by src/assets.ts. Caching them here too would double the storage.
  if (url.pathname.includes('/models/')) return;

  // A navigation must resolve even offline, and index.html is the whole app.
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((res) => {
          const copy = res.clone();
          caches.open(SHELL).then((c) => c.put('./index.html', copy));
          return res;
        })
        .catch(() => caches.match('./index.html').then((r) => r ?? Response.error())),
    );
    return;
  }

  // Cache-first for everything else: Vite's output is content-hashed, and the ORT wasm
  // is 13 MB that should never be fetched twice.
  event.respondWith(
    caches.match(request).then((hit) => hit ?? fetch(request).then((res) => {
      if (res.ok && res.type === 'basic') {
        const copy = res.clone();
        caches.open(SHELL).then((c) => c.put(request, copy));
      }
      return res;
    })),
  );
});

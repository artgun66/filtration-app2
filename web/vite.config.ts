import { rm } from 'node:fs/promises';
import { resolve } from 'node:path';

import type { Plugin } from 'vite';
import { defineConfig } from 'vite';

// Deployed under a subpath (GitHub Pages project sites) as often as at a domain root,
// and every asset URL in main.ts is built from BASE_URL so both work.
const base = process.env.VITE_BASE ?? '/';

// The ONNX Runtime WASM binary is emitted by Vite itself, from the
// `new URL(..., import.meta.url)` inside onnxruntime-web. Do not also stage a copy in
// public/ and point env.wasm.wasmPaths at it: ORT `import()`s its Emscripten glue as a
// module, and Vite will not serve public/ as source, so the dev server dies on load.

/**
 * Keep the models out of dist/ when they are served from somewhere else.
 *
 * scripts/prepare-assets.mjs already skips *staging* them under VITE_MODEL_BASE, but
 * a public/models left over from a previous dev run is still copied verbatim by Vite.
 * That silently puts 87 MB of never-requested files into the deploy -- and fails the
 * upload outright on a host with a per-file cap, which is the whole reason the models
 * are hosted elsewhere.
 */
function dropUnusedModels(): Plugin {
  return {
    name: 'drop-unused-models',
    apply: 'build',
    async closeBundle() {
      if (!process.env.VITE_MODEL_BASE) return;
      await rm(resolve(__dirname, 'dist', 'models'), { recursive: true, force: true });
      this.info?.('removed dist/models -- served from VITE_MODEL_BASE');
    },
  };
}

export default defineConfig({
  base,
  plugins: [dropUnusedModels()],
  optimizeDeps: {
    // Dev only, and load-bearing. Vite normally pre-bundles dependencies into
    // node_modules/.vite/deps/, which rewrites import.meta.url -- so ORT's
    // `new URL('./ort-wasm-simd-threaded.wasm', import.meta.url)` resolves next to the
    // pre-bundled copy, where the binary does not exist. The dev server then answers
    // the 404 with index.html, and WebAssembly.instantiate reports a bad magic word
    // ("<!do..."), which reads like a corrupt model rather than a missing file.
    // Excluding it leaves the package where its own relative paths still work.
    exclude: ['onnxruntime-web'],
  },
  server: {
    // Vite only serves files under the project root; core/ is a sibling.
    fs: { allow: ['..'] },
    // Vite rejects Host headers it does not recognise, which is what stops a public
    // tunnel dead with a 403 -- the protection is against DNS rebinding and is right
    // to be on by default. A leading dot matches subdomains, so this permits a
    // throwaway Cloudflare quick tunnel without opening the dev server to any host.
    allowedHosts: ['.trycloudflare.com', '.ngrok-free.app', '.ngrok.io'],
  },
  build: {
    // The 86 MB encoder lives in public/ and is copied verbatim, so warning about it
    // on every build is noise. The JS bundle is what this threshold should watch.
    chunkSizeWarningLimit: 1024,
  },
  worker: { format: 'es' },
});

import type { Plugin } from 'vite';
import { defineConfig } from 'vite';

// Deployed under a subpath (GitHub Pages project sites) as often as at a domain root,
// and every asset URL in main.ts is built from BASE_URL so both work.
const base = process.env.VITE_BASE ?? '/';

/**
 * Drop Vite's copy of the ONNX Runtime WASM binary.
 *
 * onnxruntime-web locates its binary two ways: a `new URL(..., import.meta.url)` that
 * Vite rewrites and emits under assets/ with a content hash, and `env.wasm.wasmPaths`,
 * the documented override. main.ts sets wasmPaths to `${BASE_URL}ort/`, which
 * scripts/prepare-assets.mjs fills from node_modules, so the hashed emission is never
 * fetched -- it is just 13 MB of dead weight in every deploy.
 *
 * If wasmPaths is ever removed from main.ts, remove this plugin in the same commit, or
 * the runtime will 404 looking for a file the build deleted.
 */
function dropDuplicateOrtWasm(): Plugin {
  return {
    name: 'drop-duplicate-ort-wasm',
    apply: 'build',
    generateBundle(_options, bundle) {
      for (const name of Object.keys(bundle)) {
        if (/ort-wasm.*\.wasm$/.test(name)) delete bundle[name];
      }
    },
  };
}

export default defineConfig({
  base,
  plugins: [dropDuplicateOrtWasm()],
  server: {
    // Vite only serves files under the project root; core/ is a sibling.
    fs: { allow: ['..'] },
  },
  build: {
    // The 86 MB encoder lives in public/ and is copied verbatim, so warning about it
    // on every build is noise. The JS bundle is what this threshold should watch.
    chunkSizeWarningLimit: 1024,
  },
  worker: { format: 'es' },
});

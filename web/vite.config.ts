import { defineConfig } from 'vite';

// Deployed under a subpath (GitHub Pages project sites) as often as at a domain root,
// and every asset URL in main.ts is built from BASE_URL so both work.
const base = process.env.VITE_BASE ?? '/';

// The ONNX Runtime WASM binary is emitted by Vite itself, from the
// `new URL(..., import.meta.url)` inside onnxruntime-web. Do not also stage a copy in
// public/ and point env.wasm.wasmPaths at it: ORT `import()`s its Emscripten glue as a
// module, and Vite will not serve public/ as source, so the dev server dies on load.

export default defineConfig({
  base,
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
  },
  build: {
    // The 86 MB encoder lives in public/ and is copied verbatim, so warning about it
    // on every build is noise. The JS bundle is what this threshold should watch.
    chunkSizeWarningLimit: 1024,
  },
  worker: { format: 'es' },
});

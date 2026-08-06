/**
 * Bundled model assets, resolved to something the ONNX runtime can open.
 *
 * The JSON loads by require() -- Metro inlines it, so it is available synchronously.
 * The .onnx graphs cannot be: on Android they are packed inside the APK and have to be
 * unpacked to a real path before onnxruntime can memory-map them. That is what
 * downloadAsync does, and it is the reason startup needs a loading state.
 */
import { Asset } from 'expo-asset';

import type { Manifest } from './model.ts';
import type { FeatureConfig } from './features.ts';

export const manifest = require('../assets/models/manifest.json') as Manifest;
export const featureConfig = require('../assets/models/feature_config.json') as FeatureConfig;
export const vocabTokens = require('../assets/models/vocab.json') as string[];

// require() needs a literal path at build time, so the graphs are listed rather than
// looked up by name. manifest.encoder_file selects between the two encoders.
const MODULES: Record<string, number> = {
  'encoder_fp32.onnx': require('../assets/models/encoder_fp32.onnx'),
  'scam_head.onnx': require('../assets/models/scam_head.onnx'),
  'type_head.onnx': require('../assets/models/type_head.onnx'),
};

/** Unpack a bundled graph and return a path onnxruntime will accept. */
export async function resolveModel(file: string): Promise<string> {
  const mod = MODULES[file];
  if (mod === undefined) {
    throw new Error(`${file} is not bundled -- add it to MODULES in src/assets.ts`);
  }
  const asset = Asset.fromModule(mod);
  await asset.downloadAsync();
  const uri = asset.localUri ?? asset.uri;
  // onnxruntime-react-native wants a filesystem path, not a file:// URL.
  return uri.startsWith('file://') ? uri.slice('file://'.length) : uri;
}

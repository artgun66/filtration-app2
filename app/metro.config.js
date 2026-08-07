// Metro does not bundle unknown binary extensions, so without this the encoder and
// the vocabulary are simply absent at runtime and every require() of them fails.
const path = require('path');
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('onnx', 'txt');

// The pipeline lives in ../core so the web app runs the identical code. Metro only
// watches the project root by default and treats anything outside it as missing, so
// the folder has to be declared -- and node_modules pinned to this project, since
// ../core has none of its own.
const core = path.resolve(__dirname, '..', 'core');
config.watchFolders = [core];
config.resolver.nodeModulesPaths = [path.resolve(__dirname, 'node_modules')];

module.exports = config;

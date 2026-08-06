// Metro does not bundle unknown binary extensions, so without this the encoder and
// the vocabulary are simply absent at runtime and every require() of them fails.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push('onnx', 'txt');

module.exports = config;

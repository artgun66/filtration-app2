/**
 * The on-device pipeline: raw message in, verdict out.
 *
 * Mirrors app-backend/predict.py step for step. The runtime is injected rather than
 * imported, which is what lets one copy of this file serve three hosts:
 * onnxruntime-react-native on a phone, onnxruntime-web in a browser, and
 * onnxruntime-node in app/test/pipeline.ts -- so the tests exercise the real thing
 * instead of a stand-in, and the phone and the web app cannot drift apart.
 *
 * Nothing here may import React, React Native, expo-* or the DOM. Loading the models
 * is the platform's job (app/src/assets.ts, web/src/assets.ts); this file only ever
 * receives a `resolve` function.
 */
import { extractFeatures, keywordFlags, buildRow, type FeatureConfig } from './features.ts';
import { encode, type Vocab } from './tokenizer.ts';

export type Manifest = {
  arm: string;
  encoder: string | null;
  encoder_file: string;
  threshold: number;
  max_seq_len: number;
  feature_names: string[];
  engineered: string[];
  keywords: string[];
  feature_labels: Record<string, string>;
  type_classes: string[] | null;
  type_min_conf: number;
  types_not_covered: string[];
};

export type Verdict = {
  scam: boolean;
  prob: number;
  /** The scam type, or null when the head is unsure or the message looks safe. */
  type: string | null;
  typeProb: number | null;
  /** Plain-English descriptions of the engineered features that fired. */
  signals: string[];
};

/**
 * The bits of an ONNX runtime this needs -- satisfied by all three ORT packages.
 *
 * `create` takes bytes as well as a path because the three hosts supply the model
 * differently: react-native and node hand over a filesystem path, while the browser
 * has already pulled the graph into a Cache API entry and has an ArrayBuffer.
 */
export type ModelSource = string | Uint8Array | ArrayBuffer;

export interface OrtLike {
  InferenceSession: {
    create(model: ModelSource): Promise<OrtSession>;
  };
  Tensor: new (type: string, data: never, dims: number[]) => unknown;
}
export interface OrtSession {
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: ArrayLike<number> }>>;
}

/**
 * Which features to surface. Counts (`money_n`) are model inputs but say nothing a
 * person can act on; the binary ones name a concrete thing about the message.
 */
function firedSignals(feats: Record<string, number>, labels: Record<string, string>): string[] {
  return Object.keys(feats)
    .filter((k) => k in labels && feats[k] > 0 &&
      (k.endsWith('_hit') || k.startsWith('has_') ||
       k === 'suspicious_tld' || k === 'ip_as_domain' || k === 'brand_domain_mismatch'))
    .map((k) => labels[k]);
}

export class Scanner {
  // Written out rather than as constructor parameter properties: Node's
  // --experimental-strip-types cannot parse those, and test/pipeline.ts runs this
  // file directly.
  private ort: OrtLike;
  private encoder: OrtSession | null;
  private scamHead: OrtSession;
  private typeHead: OrtSession | null;
  private vocab: Vocab;
  private cfg: FeatureConfig;
  manifest: Manifest;

  private constructor(
    ort: OrtLike,
    encoder: OrtSession | null,
    scamHead: OrtSession,
    typeHead: OrtSession | null,
    vocab: Vocab,
    cfg: FeatureConfig,
    manifest: Manifest,
  ) {
    this.ort = ort;
    this.encoder = encoder;
    this.scamHead = scamHead;
    this.typeHead = typeHead;
    this.vocab = vocab;
    this.cfg = cfg;
    this.manifest = manifest;
  }

  /**
   * Build once and reuse -- session creation unpacks and memory-maps 86 MB, which is
   * the app's whole cold start. Doing it per message would be unusable.
   */
  static async create(opts: {
    ort: OrtLike;
    resolve: (file: string) => Promise<ModelSource> | ModelSource;
    manifest: Manifest;
    cfg: FeatureConfig;
    vocab: Vocab;
  }): Promise<Scanner> {
    const { ort, resolve, manifest, cfg, vocab } = opts;
    const open = async (f: string) => ort.InferenceSession.create(await resolve(f));
    const encoder = manifest.encoder ? await open(manifest.encoder_file) : null;
    const scamHead = await open('scam_head.onnx');
    const typeHead = manifest.type_classes ? await open('type_head.onnx') : null;
    return new Scanner(ort, encoder, scamHead, typeHead, vocab, cfg, manifest);
  }

  private async embed(text: string): Promise<Float32Array | null> {
    if (!this.encoder) return null;
    const { inputIds, attentionMask } = encode(text, this.vocab, this.manifest.max_seq_len);
    const dims = [1, inputIds.length];
    const mk = (a: number[]) =>
      new this.ort.Tensor('int64', BigInt64Array.from(a, BigInt) as never, dims);
    const out = await this.encoder.run({
      input_ids: mk(inputIds),
      attention_mask: mk(attentionMask),
    });
    return Float32Array.from(out.embedding.data);
  }

  /** Probability of the positive class from a head with zipmap disabled. */
  private static positive(probs: ArrayLike<number>): number {
    // [P(ham), P(scam)] for the binary head; a single value would be P(scam) already.
    return probs.length >= 2 ? probs[1] : probs[0];
  }

  async scan(text: string): Promise<Verdict> {
    const feats = extractFeatures(text, this.cfg);
    const kws = keywordFlags(text, this.cfg);
    const embedding = await this.embed(text);
    const row = buildRow(this.manifest.feature_names, embedding, feats, kws);

    const input = new this.ort.Tensor('float32', row as never,
                                      [1, this.manifest.feature_names.length]);
    const out = await this.scamHead.run({ input });
    const prob = Scanner.positive(out.probabilities.data);
    const scam = prob >= this.manifest.threshold;

    let type: string | null = null;
    let typeProb: number | null = null;
    // Only name a type for something already judged suspicious: the head never saw a
    // safe message, so its answer on one is meaningless.
    if (scam && this.typeHead && this.manifest.type_classes) {
      const t = await this.typeHead.run({ input });
      const p = t.probabilities.data;
      let best = 0;
      for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
      typeProb = p[best];
      // Below the floor, say nothing. Four of the thirteen types have no training
      // rows at all, so a confident-looking guess would often be one of those.
      type = typeProb >= this.manifest.type_min_conf
        ? this.manifest.type_classes[best] : null;
    }

    return { scam, prob, type, typeProb, signals: firedSignals(feats, this.manifest.feature_labels) };
  }
}

/**
 * The 29 engineered features, ported from dataset/features.py.
 *
 * This file is the app's single largest correctness risk. Nothing here throws when it
 * is wrong -- a feature that drifts by one just moves the probability, and the app
 * keeps answering confidently. serving/app_assets/golden_vectors.json exists to catch
 * exactly that; run `node --experimental-strip-types test/golden.ts` after any edit.
 *
 * Places Python and JavaScript disagree, all of which bit during the port:
 *   - `len(s)` counts code points; `s.length` counts UTF-16 units, so one emoji is 2.
 *     Everything below iterates [...s] and uses codePoints().
 *   - Python's `\w` covers Unicode letters and digits; JavaScript's is ASCII-only.
 *     Word matching uses \p{L}\p{N}_ with the /u flag instead.
 *   - `str.isupper()` is true only when there is at least one cased character, so
 *     "123" is not all-caps. Uppercase comparison alone would say it is.
 *
 * Word lists are NOT duplicated here -- they load from feature_config.json, which
 * export_onnx.py writes from dataset/features.yaml.
 */

export type FeatureConfig = {
  known_tlds: string[];
  suspicious_tlds: string[];
  shorteners: string[];
  mismatch_brands: string[];
  families: Record<string, string[]>;
  keywords: string[];
};

const URL_RE =
  /(https?:\/\/|www\.|\b[a-z0-9-]+\.(com|net|org|info|xyz|ru|cn|co|io|me|us|biz|top|link|click|vip|shop|site|online|icu|cc|tk|ml|ga|gq|buzz)\b)/i;
const URL_FULL_RE = /(https?:\/\/|www\.)?([a-z0-9][a-z0-9.-]*\.[a-z]{2,})(\/\S*)?/gi;
const IP_DOMAIN_RE = /\b(?:\d{1,3}\.){3}\d{1,3}\b/;
const LEET_RE = /\b[a-z]+[0-9]+[a-z]+\b/i;
const WORD_RE = /[\p{L}\p{N}_]+/gu;

const IS_LETTER = /\p{L}/u;      // Python str.isalpha
const IS_UPPER = /\p{Lu}/u;      // a cased character in upper case
const IS_DIGIT = /\p{Nd}/u;      // Python str.isdigit
const IS_ALNUM = /[\p{L}\p{N}]/u;
const IS_SPACE = /\s/u;

const codePoints = (s: string): string[] => Array.from(s);

/** Non-overlapping occurrences, matching Python's str.count. */
function countOccurrences(haystack: string, needle: string): number {
  if (!needle) return 0;
  return haystack.split(needle).length - 1;
}

/** Python str.isupper: all cased characters upper, and at least one cased. */
function isAllCaps(w: string): boolean {
  return w === w.toUpperCase() && IS_UPPER.test(w);
}

function registrable(host: string): string {
  const parts = host.split('.');
  return parts.length >= 2 ? parts.slice(-2).join('.') : host;
}

/** Hosts that look like real domains, mirroring features.py:_domains. */
function extractDomains(text: string, knownTlds: Set<string>): string[] {
  const out: string[] = [];
  URL_FULL_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = URL_FULL_RE.exec(text)) !== null) {
    if (m[0] === '') {
      URL_FULL_RE.lastIndex++;   // zero-width match: the prefix group is optional
      continue;
    }
    const prefix = m[1];
    const host = (m[2] || '').toLowerCase().replace(/\.+$/, '');
    if (!host.includes('.')) continue;
    if (/^\d+$/.test(host.replace(/\./g, ''))) continue;      // "3.50" is not a host
    const tld = host.split('.').pop() as string;
    if (!prefix && !knownTlds.has(tld)) continue;             // "tomorrow.We" is not
    out.push(host);
  }
  return out;
}

export function extractFeatures(text: string, cfg: FeatureConfig): Record<string, number> {
  const t = text || '';
  const low = t.toLowerCase();
  const flat = low.replace(/[^a-z0-9]/g, '');
  const chars = codePoints(t);
  const n = chars.length || 1;
  const words = t.match(WORD_RE) ?? [];

  const knownTlds = new Set(cfg.known_tlds);
  const suspiciousTlds = new Set(cfg.suspicious_tlds);
  const shorteners = new Set(cfg.shorteners);
  const domains = extractDomains(t, knownTlds);
  const tlds = new Set(domains.map((d) => d.split('.').pop() as string));

  const f: Record<string, number> = {};

  // --- links
  f.has_url = URL_RE.test(t) ? 1 : 0;
  f.n_urls = domains.length;
  f.has_shortener = domains.some((d) => shorteners.has(registrable(d))) ? 1 : 0;
  f.suspicious_tld = [...tlds].some((x) => suspiciousTlds.has(x)) ? 1 : 0;
  f.ip_as_domain = IP_DOMAIN_RE.test(t) && domains.length > 0 ? 1 : 0;

  const domainChars = domains.reduce((a, d) => a + codePoints(d).length, 0);
  const domainDigits = domains.reduce(
    (a, d) => a + codePoints(d).filter((c) => IS_DIGIT.test(c)).length, 0);
  f.url_digit_ratio = domainChars ? domainDigits / domainChars : 0.0;
  f.domain_len_max = domains.length
    ? Math.max(...domains.map((d) => codePoints(d).length)) : 0;

  const named = cfg.mismatch_brands.filter((b) => flat.includes(b));
  f.brand_domain_mismatch =
    named.length > 0 &&
    domains.length > 0 &&
    !named.some((b) => domains.some((d) => d.replace(/\./g, '').replace(/-/g, '').includes(b)))
      ? 1 : 0;

  // --- keyword families
  for (const [fam, terms] of Object.entries(cfg.families)) {
    const hits = terms.reduce((a, term) => a + countOccurrences(low, term), 0);
    f[`${fam}_n`] = hits;
    f[`${fam}_hit`] = hits > 0 ? 1 : 0;
  }

  // --- text shape
  const letters = chars.filter((c) => IS_LETTER.test(c)).length;
  f.length = chars.length;
  f.n_words = words.length;
  f.capital_ratio = chars.filter((c) => IS_UPPER.test(c)).length / (letters || 1);
  f.digit_ratio = chars.filter((c) => IS_DIGIT.test(c)).length / n;
  f.punct_ratio =
    chars.filter((c) => !IS_ALNUM.test(c) && !IS_SPACE.test(c)).length / n;
  f.n_exclaim = countOccurrences(t, '!');
  f.n_allcaps_words = words.filter(
    (w) => codePoints(w).length > 2 && isAllCaps(w)).length;
  f.has_leetspeak = LEET_RE.test(t) ? 1 : 0;
  f.avg_word_len = words.length
    ? words.reduce((a, w) => a + codePoints(w).length, 0) / words.length : 0.0;

  return f;
}

/** The 38 keyword flags, ported from dataset/keywords.py. */
export function keywordFlags(text: string, cfg: FeatureConfig): Record<string, number> {
  const low = (text || '').toLowerCase();
  const out: Record<string, number> = {};
  for (const kw of cfg.keywords) out[keywordColumn(kw)] = low.includes(kw) ? 1 : 0;
  return out;
}

/** dataset/keywords.py:column -- kw_ + the term slugged to [a-z0-9_]. */
export function keywordColumn(kw: string): string {
  return 'kw_' + kw.toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
}

/**
 * The model's input row, ordered by the manifest's feature_names.
 *
 * Assembled by name, never by concatenation -- the same rule serving/predict.py
 * follows. A column-order mismatch between training and the app does not raise; it
 * silently changes every answer.
 */
export function buildRow(
  featureNames: string[],
  embedding: Float32Array | number[] | null,
  feats: Record<string, number>,
  kws: Record<string, number>,
): Float32Array {
  const row = new Float32Array(featureNames.length);
  featureNames.forEach((name, i) => {
    if (name.startsWith('emb')) {
      if (!embedding) throw new Error(`${name} needs an embedding but none was given`);
      row[i] = embedding[Number(name.slice(3))];
    } else if (name in feats) {
      row[i] = feats[name];
    } else if (name in kws) {
      row[i] = kws[name];
    } else {
      throw new Error(`no value for feature "${name}"`);
    }
  });
  return row;
}

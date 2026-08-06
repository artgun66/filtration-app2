/**
 * BERT WordPiece tokenizer, enough of it to reproduce what the encoder was trained on.
 *
 * all-MiniLM-L6-v2 and bge-small-en-v1.5 both use the bert-base-uncased vocabulary
 * with do_lower_case = true, which implies accent stripping. Getting any of that wrong
 * shifts the token ids, and a shifted id is a different word to the encoder -- so the
 * golden fixture carries Python's input_ids and test/golden.ts checks them.
 *
 * Deliberately not supported: never-split special tokens beyond the ones added here,
 * and the Chinese-character spacing rule is included because the corpus contains
 * NUS Singapore SMS with CJK.
 */

export type Vocab = Map<string, number>;

export function loadVocab(text: string): Vocab {
  const v: Vocab = new Map();
  text.split('\n').forEach((line, i) => {
    const tok = line.replace(/\r$/, '');
    if (tok.length > 0 && !v.has(tok)) v.set(tok, i);
  });
  return v;
}

const UNK = '[UNK]';
const CLS = '[CLS]';
const SEP = '[SEP]';
const MAX_CHARS_PER_WORD = 100;

/** Control characters are dropped; \t \n \r become whitespace. BERT's _clean_text. */
function cleanText(s: string): string {
  let out = '';
  for (const c of s) {
    const cp = c.codePointAt(0) as number;
    if (cp === 0 || cp === 0xfffd) continue;
    if (isControl(c)) continue;
    out += isWhitespace(c) ? ' ' : c;
  }
  return out;
}

function isWhitespace(c: string): boolean {
  return c === ' ' || c === '\t' || c === '\n' || c === '\r' || /\s/u.test(c);
}

function isControl(c: string): boolean {
  if (c === '\t' || c === '\n' || c === '\r') return false;
  return /\p{Cc}|\p{Cf}|\p{Co}|\p{Cs}/u.test(c);
}

function isPunctuation(c: string): boolean {
  const cp = c.codePointAt(0) as number;
  // BERT treats every non-alphanumeric ASCII character as punctuation, plus the
  // Unicode punctuation categories.
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(c);
}

function isChinese(c: string): boolean {
  const cp = c.codePointAt(0) as number;
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

/** Lowercase, strip accents, isolate punctuation and CJK. BERT's BasicTokenizer. */
function basicTokenize(text: string): string[] {
  let s = cleanText(text);
  // space out CJK so each character becomes its own token
  s = Array.from(s).map((c) => (isChinese(c) ? ` ${c} ` : c)).join('');

  const out: string[] = [];
  for (const raw of s.split(/\s+/)) {
    if (!raw) continue;
    // NFD then drop combining marks -- the accent-stripping half of do_lower_case
    const token = raw.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '');
    let cur = '';
    for (const c of token) {
      if (isPunctuation(c)) {
        if (cur) { out.push(cur); cur = ''; }
        out.push(c);
      } else {
        cur += c;
      }
    }
    if (cur) out.push(cur);
  }
  return out;
}

/** Greedy longest-match-first, the WordpieceTokenizer half. */
function wordpiece(token: string, vocab: Vocab): string[] {
  const chars = Array.from(token);
  if (chars.length > MAX_CHARS_PER_WORD) return [UNK];

  const sub: string[] = [];
  let start = 0;
  while (start < chars.length) {
    let end = chars.length;
    let found: string | null = null;
    while (start < end) {
      const piece = (start > 0 ? '##' : '') + chars.slice(start, end).join('');
      if (vocab.has(piece)) { found = piece; break; }
      end--;
    }
    if (found === null) return [UNK];     // one bad piece makes the whole word [UNK]
    sub.push(found);
    start = end;
  }
  return sub;
}

export type Encoded = { inputIds: number[]; attentionMask: number[] };

/**
 * Text to model input, including [CLS]/[SEP] and truncation.
 *
 * maxLen must match manifest.max_seq_len -- the training runs capped at 256 tokens,
 * and the corpus p99 is 320 characters, so this truncates essentially nothing.
 */
export function encode(text: string, vocab: Vocab, maxLen = 256): Encoded {
  const pieces: string[] = [];
  for (const tok of basicTokenize(text)) pieces.push(...wordpiece(tok, vocab));

  const body = pieces.slice(0, Math.max(0, maxLen - 2));
  const tokens = [CLS, ...body, SEP];
  const unk = vocab.get(UNK) as number;
  const inputIds = tokens.map((t) => vocab.get(t) ?? unk);
  return { inputIds, attentionMask: inputIds.map(() => 1) };
}

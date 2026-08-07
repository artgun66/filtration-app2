/**
 * What the app says to the person reading it.
 *
 * This lives in core/ rather than in a component because the wording is a product
 * decision, not a presentation detail, and the phone and the web app must not drift
 * into telling people different things. Markup differs per platform; the sentences
 * do not.
 *
 * Three rules shaped all of it:
 *   - Lead with an action, not a score. "Do not tap any links" is usable; "0.94" is
 *     not. Confidence is still shown, in words, because a verdict with no visible
 *     uncertainty invites blind trust.
 *   - Never imply coverage the model does not have. Four of the thirteen scam types
 *     have no training data, so a safe verdict says so plainly rather than reading as
 *     an all-clear.
 *   - Do not name a scam type unless the head is confident. "Not sure what kind" is a
 *     real answer and a better one than a wrong label.
 */
import type { Verdict } from './model.ts';

/** Confidence in words. People act on "fairly confident", not on 0.83. */
export function confidenceWord(p: number, scam: boolean): string {
  const strength = scam ? p : 1 - p;
  if (strength >= 0.95) return 'Very confident';
  if (strength >= 0.8) return 'Fairly confident';
  return 'Not very confident';
}

export const HEADLINE = {
  scam: 'This looks like a scam',
  safe: 'This looks safe',
};

export const UNKNOWN_TYPE = 'Not sure what kind of scam it is.';

export const SCAM_ADVICE = [
  'Do not tap any links in the message.',
  'Do not reply, and do not call any number it gives.',
  'If it claims to be your bank or a company you use, contact them using the number on your card or their official app.',
  'Delete the message once you are done.',
];

/**
 * The line under a safe verdict. `notCovered` comes from manifest.types_not_covered,
 * so it shrinks on its own as labeling/ fills those classes in -- it is never a
 * hardcoded list that can quietly go stale.
 */
export function coverageCaveat(notCovered: string[]): string | null {
  if (notCovered.length === 0) return null;
  return `Cyber Scout cannot yet spot every kind of scam. It does not recognise ` +
    `${notCovered.join(', ')}. If a message feels wrong, trust that feeling and ` +
    `check with someone you know.`;
}

/** The heading above the signal list, which reads differently either way. */
export function signalsHeading(scam: boolean): string {
  return scam ? 'Why' : 'What we noticed';
}

/** The one-line summary, used where there is no room for the full card. */
export function summarise(v: Verdict): string {
  const head = v.scam ? HEADLINE.scam : HEADLINE.safe;
  const kind = v.scam && v.type ? ` (${v.type})` : '';
  return `${head}${kind} -- ${confidenceWord(v.prob, v.scam).toLowerCase()}`;
}

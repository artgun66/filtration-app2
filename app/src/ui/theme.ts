/**
 * Sizing and colour for an audience with older eyes.
 *
 * Body text starts at 18pt rather than the usual 14-16, touch targets are 56pt tall
 * (Android's minimum is 48), and the verdict colours were picked to stay separable
 * for red-green colour blindness -- the amber/blue pair carries the meaning, with the
 * wording rather than the hue doing the real work.
 */
export const C = {
  bg: '#F7F7F9',
  card: '#FFFFFF',
  text: '#14161A',
  muted: '#5B6270',
  border: '#D8DBE2',

  dangerBg: '#FDECEA',
  dangerFg: '#8E1B12',
  dangerAccent: '#C0392B',

  safeBg: '#E9F3FB',
  safeFg: '#14456E',
  safeAccent: '#1F6FB2',
};

export const T = {
  title: 30,
  verdict: 28,
  body: 18,
  small: 15,
};

export const S = {
  gap: 12,
  pad: 20,
  radius: 14,
  touch: 56,
};

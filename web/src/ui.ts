/**
 * Rendering. Every sentence here comes from core/copy.ts -- see the rules documented
 * there. This file decides where words go, never what they say.
 *
 * Plain DOM rather than a framework: the whole UI is one screen and two states, and a
 * framework would be a larger download than the code it replaced. The bytes matter
 * more than usual here because the model already costs 86 MB.
 */
import type { Verdict } from '../../core/model.ts';
import type { StorageState } from './assets.ts';
import {
  confidenceWord, HEADLINE, UNKNOWN_TYPE, SCAM_ADVICE, FALSE_ALARM, signalsHeading,
} from '../../core/copy.ts';

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K, cls?: string, text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function bulletList(items: string[], heading: string): HTMLElement {
  const block = el('div', 'block');
  block.append(el('h2', 'heading', heading));
  const ul = el('ul', 'items');
  for (const item of items) ul.append(el('li', undefined, item));
  block.append(ul);
  return block;
}

export function resultCard(verdict: Verdict): HTMLElement {
  const card = el('section', `card ${verdict.scam ? 'danger' : 'safe'}`);
  card.setAttribute('role', 'status');
  // Assertive rather than polite: the person asked a direct question and is waiting on
  // the answer, so a screen reader should interrupt with it.
  card.setAttribute('aria-live', 'assertive');

  card.append(el('h1', 'verdict', verdict.scam ? HEADLINE.scam : HEADLINE.safe));

  const kind = verdict.scam && verdict.type ? ` · ${verdict.type}` : '';
  card.append(el('p', 'confidence',
                 confidenceWord(verdict.prob, verdict.scam) + kind));

  if (verdict.falseAlarm) card.append(el('p', 'note', FALSE_ALARM));
  else if (verdict.scam && !verdict.type) card.append(el('p', 'note', UNKNOWN_TYPE));
  // The advice still shows on a suspected false alarm: not tapping the link costs
  // nothing if the message turns out to be genuine.
  if (verdict.scam) card.append(bulletList(SCAM_ADVICE, 'What to do'));
  if (verdict.signals.length) {
    card.append(bulletList(verdict.signals, signalsHeading(verdict.scam)));
  }

  return card;
}

/**
 * The loading screen. 86 MB is long enough that silence reads as breakage -- but a
 * progress bar on a load that is coming from local storage reads as a *re-download*,
 * which is worse: it makes working caching look broken. So the bar is shown only when
 * bytes are actually coming over the network.
 */
export function progressPanel(state: StorageState): {
  node: HTMLElement; update: (file: string, got: number, total: number) => void;
} {
  const downloading = state !== 'stored';
  const node = el('section', 'card loading');
  const title = el('h1', 'verdict',
                   downloading ? 'Getting Cyber Scout ready' : 'Starting Cyber Scout');
  const detail = el('p', 'confidence', downloading
    ? 'This downloads once, then it is kept on your device.'
    : 'Loading from your device.');
  node.append(title, detail);

  const bar = el('div', 'bar');
  const fill = el('div', 'bar-fill');
  const pct = el('p', 'note', '');
  if (downloading) {
    bar.append(fill);
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    node.append(bar, pct);
  }

  // Storage that does not work is invisible otherwise -- it just looks like the app is
  // slow every time. Private browsing is the usual cause.
  if (state === 'unavailable') {
    node.append(el('p', 'note',
      'Your browser will not let Cyber Scout save the file, so it has to download ' +
      'each time. Private browsing is the usual reason.'));
  }

  // Progress is reported per file, but the encoder is 99% of the bytes -- so a naive
  // per-file bar would sit at 0% for a minute and then jump. Weighting by size keeps
  // it honest.
  const seen = new Map<string, { got: number; total: number }>();
  return {
    node,
    update(file, got, total) {
      if (!downloading) return;
      seen.set(file, { got, total });
      let g = 0;
      let t = 0;
      for (const v of seen.values()) {
        g += v.got;
        t += v.total || v.got;
      }
      const ratio = t ? Math.min(1, g / t) : 0;
      fill.style.width = `${(ratio * 100).toFixed(1)}%`;
      bar.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
      pct.textContent = `${(g / 1024 / 1024).toFixed(0)} MB of about ` +
                        `${(t / 1024 / 1024).toFixed(0)} MB`;
    },
  };
}

export function errorPanel(err: unknown): HTMLElement {
  const card = el('section', 'card danger');
  card.append(el('h1', 'verdict', 'Cyber Scout could not start'));
  card.append(el('p', 'confidence',
                 'Check your internet connection and reload the page.'));
  card.append(el('p', 'note', String(err instanceof Error ? err.message : err)));
  return card;
}

/**
 * The add-to-home-screen nudge, shown only on iOS Safari in a browser tab.
 *
 * This is not a growth prompt. On iOS, storage for a site that was never added to the
 * home screen is evicted after about a week of non-use -- so skipping this step is what
 * makes someone re-download 86 MB. Android and installed PWAs never see it.
 */
export function installHint(): HTMLElement | null {
  const ua = navigator.userAgent;
  const iOS = /iPad|iPhone|iPod/.test(ua) ||
              (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const standalone = window.matchMedia('(display-mode: standalone)').matches ||
                     ('standalone' in navigator && Boolean(navigator.standalone));
  if (!iOS || standalone) return null;

  const box = el('aside', 'hint');
  box.append(el('strong', undefined, 'Keep Cyber Scout on your Home Screen'));
  box.append(el('p', undefined,
    'Tap the Share button at the bottom of the screen, then "Add to Home Screen". ' +
    'This keeps it working offline. Without it, iPhone may clear Cyber Scout after a ' +
    'week and it will have to download again.'));
  return box;
}

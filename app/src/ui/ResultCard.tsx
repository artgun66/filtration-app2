/**
 * The verdict, written for someone deciding what to do about a text message.
 *
 * Three rules shaped this:
 *   - Lead with an action, not a score. "Do not tap any links" is usable; "0.94" is
 *     not. The probability is still shown, in words, because a verdict with no visible
 *     uncertainty invites blind trust.
 *   - Never imply coverage the model does not have. Four of the thirteen scam types
 *     have no training data, so a safe verdict says so plainly rather than reading as
 *     an all-clear.
 *   - Do not name a scam type unless the head is confident. "Not sure what kind" is a
 *     real answer and a better one than a wrong label.
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

import { C, T, S } from './theme.ts';
import type { Verdict } from '../model.ts';

/** Confidence in words. People act on "fairly sure", not on 0.83. */
function confidenceWord(p: number, scam: boolean): string {
  const strength = scam ? p : 1 - p;
  if (strength >= 0.95) return 'Very confident';
  if (strength >= 0.8) return 'Fairly confident';
  return 'Not very confident';
}

const SCAM_ADVICE = [
  'Do not tap any links in the message.',
  'Do not reply, and do not call any number it gives.',
  'If it claims to be your bank or a company you use, contact them using the number on your card or their official app.',
  'Delete the message once you are done.',
];

export function ResultCard({ verdict, notCovered }: {
  verdict: Verdict;
  notCovered: string[];
}) {
  const scam = verdict.scam;
  const fg = scam ? C.dangerFg : C.safeFg;
  const bg = scam ? C.dangerBg : C.safeBg;
  const accent = scam ? C.dangerAccent : C.safeAccent;

  return (
    <View style={[styles.card, { backgroundColor: bg, borderColor: accent }]}>
      <Text style={[styles.verdict, { color: fg }]}>
        {scam ? 'This looks like a scam' : 'This looks safe'}
      </Text>

      <Text style={[styles.confidence, { color: fg }]}>
        {confidenceWord(verdict.prob, scam)}
        {verdict.type ? ` · ${verdict.type}` : ''}
      </Text>

      {scam && !verdict.type && (
        <Text style={[styles.note, { color: fg }]}>
          Not sure what kind of scam it is.
        </Text>
      )}

      {scam && (
        <View style={styles.block}>
          <Text style={[styles.heading, { color: fg }]}>What to do</Text>
          {SCAM_ADVICE.map((line) => (
            <Text key={line} style={[styles.item, { color: fg }]}>{'•  ' + line}</Text>
          ))}
        </View>
      )}

      {verdict.signals.length > 0 && (
        <View style={styles.block}>
          <Text style={[styles.heading, { color: fg }]}>
            {scam ? 'Why' : 'What we noticed'}
          </Text>
          {verdict.signals.map((s) => (
            <Text key={s} style={[styles.item, { color: fg }]}>{'•  ' + s}</Text>
          ))}
        </View>
      )}

      {!scam && notCovered.length > 0 && (
        <View style={[styles.block, styles.caveat, { borderTopColor: accent }]}>
          <Text style={[styles.caveatText, { color: fg }]}>
            Cyber Scout cannot yet spot every kind of scam. It does not recognise{' '}
            {notCovered.join(', ')}. If a message feels wrong, trust that feeling and
            check with someone you know.
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: S.radius,
    borderWidth: 2,
    padding: S.pad,
    gap: S.gap,
  },
  verdict: { fontSize: T.verdict, fontWeight: '700' },
  confidence: { fontSize: T.body, fontWeight: '600' },
  note: { fontSize: T.body },
  block: { gap: 6 },
  heading: { fontSize: T.body, fontWeight: '700', marginTop: 4 },
  item: { fontSize: T.body, lineHeight: 26 },
  caveat: { borderTopWidth: 1, paddingTop: S.gap },
  caveatText: { fontSize: T.small, lineHeight: 22 },
});

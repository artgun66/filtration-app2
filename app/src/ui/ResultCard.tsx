/**
 * The verdict, written for someone deciding what to do about a text message.
 *
 * Layout only. Every sentence comes from core/copy.ts, so the phone and the web app
 * cannot end up telling people different things -- see the rules documented there.
 */
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

import { C, T, S } from './theme.ts';
import type { Verdict } from '../../../core/model.ts';
import {
  confidenceWord, HEADLINE, UNKNOWN_TYPE, SCAM_ADVICE, FALSE_ALARM, signalsHeading,
} from '../../../core/copy.ts';

export function ResultCard({ verdict }: { verdict: Verdict }) {
  const scam = verdict.scam;
  const fg = scam ? C.dangerFg : C.safeFg;
  const bg = scam ? C.dangerBg : C.safeBg;
  const accent = scam ? C.dangerAccent : C.safeAccent;

  return (
    <View style={[styles.card, { backgroundColor: bg, borderColor: accent }]}>
      <Text style={[styles.verdict, { color: fg }]}>
        {scam ? HEADLINE.scam : HEADLINE.safe}
      </Text>

      <Text style={[styles.confidence, { color: fg }]}>
        {confidenceWord(verdict.prob, scam)}
        {verdict.type ? ` · ${verdict.type}` : ''}
      </Text>

      {verdict.falseAlarm && (
        <Text style={[styles.note, { color: fg }]}>{FALSE_ALARM}</Text>
      )}

      {!verdict.falseAlarm && scam && !verdict.type && (
        <Text style={[styles.note, { color: fg }]}>{UNKNOWN_TYPE}</Text>
      )}

      {/* Advice still shows on a suspected false alarm: not tapping the link costs
          nothing if the message turns out to be genuine. */}
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
          <Text style={[styles.heading, { color: fg }]}>{signalsHeading(scam)}</Text>
          {verdict.signals.map((s) => (
            <Text key={s} style={[styles.item, { color: fg }]}>{'•  ' + s}</Text>
          ))}
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
});

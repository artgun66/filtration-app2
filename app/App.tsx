/**
 * Cyber Scout -- check whether a text message is a scam.
 *
 * Everything runs on the phone. No message ever leaves the device, which is why the
 * app asks for no permissions at all: the user brings the message to it, either by
 * sharing from their messaging app or by pasting.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator, KeyboardAvoidingView, Platform, Pressable, ScrollView,
  StyleSheet, Text, TextInput, View,
} from 'react-native';
import * as Clipboard from 'expo-clipboard';
import { StatusBar } from 'expo-status-bar';
import { useShareIntent } from 'expo-share-intent';
import * as ort from 'onnxruntime-react-native';

import { Scanner, type Verdict, type OrtLike } from '../core/model.ts';
import { loadVocab } from '../core/tokenizer.ts';
import { manifest, featureConfig, vocabTokens, resolveModel } from './src/assets.ts';
import { ResultCard } from './src/ui/ResultCard.tsx';
import { C, T, S } from './src/ui/theme.ts';

export default function App() {
  const [scanner, setScanner] = useState<Scanner | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [text, setText] = useState('');
  const [verdict, setVerdict] = useState<Verdict | null>(null);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<ScrollView>(null);

  const { hasShareIntent, shareIntent, resetShareIntent } = useShareIntent();

  // Sessions are built once. This unpacks ~86 MB out of the APK, so it is the whole
  // cold start and the user waits behind a spinner for it exactly once per launch.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await Scanner.create({
          ort: ort as unknown as OrtLike,
          resolve: resolveModel,
          manifest,
          cfg: featureConfig,
          vocab: loadVocab(vocabTokens),
        });
        if (!cancelled) setScanner(s);
      } catch (e) {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const check = useCallback(async (message: string) => {
    const body = message.trim();
    if (!scanner || !body) return;
    setBusy(true);
    try {
      setVerdict(await scanner.scan(body));
      scrollRef.current?.scrollTo({ y: 0, animated: true });
    } finally {
      setBusy(false);
    }
  }, [scanner]);

  // A message shared from the messaging app: fill the box and check it immediately,
  // so the share flow is one tap rather than a paste-and-press.
  useEffect(() => {
    if (!hasShareIntent || !scanner) return;
    const shared = shareIntent?.text ?? '';
    if (shared) {
      setText(shared);
      void check(shared);
    }
    resetShareIntent();
  }, [hasShareIntent, shareIntent, scanner, check, resetShareIntent]);

  const paste = useCallback(async () => {
    const clip = await Clipboard.getStringAsync();
    if (clip) {
      setText(clip);
      void check(clip);
    }
  }, [check]);

  if (loadError) {
    return (
      <View style={[styles.screen, styles.centre]}>
        <Text style={styles.title}>Cyber Scout could not start</Text>
        <Text style={styles.muted}>{loadError}</Text>
      </View>
    );
  }

  if (!scanner) {
    return (
      <View style={[styles.screen, styles.centre]}>
        <ActivityIndicator size="large" color={C.safeAccent} />
        <Text style={styles.loading}>Getting ready…</Text>
        <Text style={styles.muted}>This takes a moment the first time.</Text>
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.screen}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <StatusBar style="dark" />
      <ScrollView ref={scrollRef} contentContainerStyle={styles.content}
                  keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Cyber Scout</Text>
        <Text style={styles.muted}>
          Paste a text message, or share one from your messages app, and Cyber Scout
          will tell you whether it looks like a scam. Nothing you paste leaves this
          phone.
        </Text>

        <TextInput
          style={styles.input}
          value={text}
          onChangeText={setText}
          placeholder="Paste the message here"
          placeholderTextColor={C.muted}
          multiline
          textAlignVertical="top"
        />

        <View style={styles.row}>
          <Pressable style={[styles.button, styles.secondary]} onPress={paste}>
            <Text style={styles.secondaryLabel}>Paste</Text>
          </Pressable>
          <Pressable
            style={[styles.button, styles.primary, (!text.trim() || busy) && styles.disabled]}
            disabled={!text.trim() || busy}
            onPress={() => void check(text)}
          >
            <Text style={styles.primaryLabel}>{busy ? 'Checking…' : 'Check message'}</Text>
          </Pressable>
        </View>

        {text.length > 0 && (
          <Pressable onPress={() => { setText(''); setVerdict(null); }}>
            <Text style={styles.clear}>Clear</Text>
          </Pressable>
        )}

        {verdict && (
          <ResultCard verdict={verdict} />
        )}
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: C.bg },
  centre: { alignItems: 'center', justifyContent: 'center', gap: S.gap, padding: S.pad },
  content: { padding: S.pad, paddingTop: 60, gap: S.gap, paddingBottom: 60 },
  title: { fontSize: T.title, fontWeight: '700', color: C.text },
  loading: { fontSize: T.body, fontWeight: '600', color: C.text },
  muted: { fontSize: T.small, color: C.muted, lineHeight: 22 },
  input: {
    minHeight: 150,
    backgroundColor: C.card,
    borderColor: C.border,
    borderWidth: 1,
    borderRadius: S.radius,
    padding: S.gap,
    fontSize: T.body,
    color: C.text,
    lineHeight: 26,
  },
  row: { flexDirection: 'row', gap: S.gap },
  button: {
    height: S.touch,
    borderRadius: S.radius,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: S.pad,
  },
  primary: { flex: 1, backgroundColor: C.safeAccent },
  primaryLabel: { color: '#FFFFFF', fontSize: T.body, fontWeight: '700' },
  secondary: { backgroundColor: C.card, borderWidth: 1, borderColor: C.border },
  secondaryLabel: { color: C.text, fontSize: T.body, fontWeight: '600' },
  disabled: { opacity: 0.5 },
  clear: { fontSize: T.small, color: C.safeAccent, fontWeight: '600', paddingVertical: 4 },
});

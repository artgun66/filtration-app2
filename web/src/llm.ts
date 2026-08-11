/**
 * Optional second opinion from the 3B teacher, for testing only.
 *
 * Everything else in this app runs on the device. This does not: the message is posted
 * to whatever machine is running scam-type-classification/serve_llm.py. That is a
 * deliberate exception for a research view, and the UI says so on every answer rather
 * than quietly making the app's central claim untrue.
 *
 * Off unless VITE_LLM_URL is set at build time, so a normal build cannot reach it and
 * the phone app never sees this file at all.
 */
const URL_: string | undefined = import.meta.env.VITE_LLM_URL;

export type LlmVerdict = {
  type: string;
  confidence: number;
  warning_signs: string[];
  explanation: string;
  model: string;
  seconds: number;
  left_the_device: boolean;
};

export const llmEnabled = Boolean(URL_);

/** True when the server is up. Probed once so the panel can stay hidden otherwise. */
export async function llmReady(): Promise<boolean> {
  if (!URL_) return false;
  try {
    const res = await fetch(URL_, { method: 'GET' });
    return res.ok && Boolean((await res.json()).ready);
  } catch {
    return false;      // not running is the normal case, not an error worth showing
  }
}

/** null rather than throwing: the on-device verdict must render either way. */
export async function llmClassify(text: string): Promise<LlmVerdict | null> {
  if (!URL_) return null;
  try {
    const res = await fetch(URL_, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return null;
    return (await res.json()) as LlmVerdict;
  } catch {
    return null;
  }
}

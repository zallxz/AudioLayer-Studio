import type { Category } from "./services/types";

export const CATEGORY_META: Record<Category, { label: string; short: string; color: string }> = {
  dialogue: { label: "Dialogue", short: "DLG", color: "#4cc3ff" },
  bgm: { label: "BGM", short: "BGM", color: "#b48cff" },
  sfx: { label: "SFX", short: "SFX", color: "#ffb347" },
  music: { label: "Music", short: "MUS", color: "#ff6fa8" },
  noise: { label: "Noise", short: "NSE", color: "#8fa3b0" },
};

export const fmtTime = (t: number, ms = false) => {
  if (!isFinite(t)) return "--:--";
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return ms ? `${String(m).padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}` : `${String(m).padStart(2, "0")}:${String(Math.floor(s)).padStart(2, "0")}`;
};

export const FLAG_TEXT: Record<string, string> = {
  unintelligible: "Unintelligible / gibberish speech",
  no_words_recognized: "Speech-like but no words recognised",
  language_uncertain: "Language uncertain",
  repetitive_or_hallucinated: "Repetitive (possible ASR hallucination)",
  role_uncertain: "BGM vs Music role uncertain",
  overlaps_music_split_estimated: "Overlaps music — split is estimated",
  not_removable: "Detection only — cannot be removed",
  heuristic_classifier: "Heuristic classifier (low accuracy)",
  clipping: "Clipping",
  hiss: "Hiss",
  hum: "Hum",
  rumble: "Rumble",
  asr_only: "Found by speech recognition",
  detected_by_classifier: "Detected by sound classifier",
};

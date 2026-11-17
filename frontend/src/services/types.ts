export type Category = "dialogue" | "bgm" | "sfx" | "music" | "noise";
export const CATEGORIES: Category[] = ["dialogue", "bgm", "sfx", "music", "noise"];

export interface LabelScore { label: string; score: number }

export interface Segment {
  id: string;
  category: Category;
  start: number;
  end: number;
  duration: number;
  confidence: number;
  uncertain: boolean;
  stream: "voice" | "music" | "fx" | "noise" | "mix";
  description: string;
  labels: LabelScore[];
  transcript: string | null;
  intelligibility: number | null;
  language: string | null;
  flags: string[];
  muted: boolean;
  solo: boolean;
  edited: boolean;
  original_category: Category | null;
}

export interface MediaInfo {
  filename: string;
  container: string;
  size_bytes: number;
  has_video: boolean;
  has_audio: boolean;
  video_duration: number | null;
  audio_duration: number | null;
  sample_rate: number | null;
  channels: number | null;
  audio_codec: string | null;
  video_codec: string | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  working_sample_rate: number;
  working_channels: number;
}

export interface QualityIssue {
  id: string;
  kind: string;
  severity: "info" | "warning" | "error";
  title: string;
  detail: string;
  start: number | null;
  end: number | null;
  fix: string | null;
}

export interface Analysis {
  classifier: string;
  separator: string;
  asr: string;
  separated: boolean;
  loudness: { integrated_lufs: number | null; true_peak_dbtp: number | null; lra: number | null };
  issues: QualityIssue[];
  limitations: string[];
  models: { role: string; name: string; license: string; note: string }[];
  category_presence: Record<string, number>;
  timings: Record<string, number>;
}

export interface Job {
  id: string;
  project_id: string;
  kind: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  progress: number;
  stage: string;
  message: string;
  error: string | null;
  result: Record<string, any>;
}

export interface Project {
  id: string;
  name: string;
  status: "uploaded" | "analyzing" | "analyzed" | "error";
  error: string | null;
  media: MediaInfo;
  original_file: string;
  analysis: Analysis | null;
  segments: Segment[];
  mix: { removed: Category[] };
  stems_version: number;
  exports: { id: string; dir: string; files: string[]; zip: string | null; skipped: { item: string; reason: string }[] }[];
  can_undo: boolean;
  can_redo: boolean;
  active_jobs: Job[];
  stems_available: boolean;
  limitations_for_selection: string[];
}

export interface ProjectSummary { id: string; name: string; status: string; created_at: number; duration: number | null; has_video: boolean }

export interface Peaks { duration: number; points_per_second: number; min: number[]; max: number[] }

export interface SystemInfo {
  device: string;
  classifier: string | null;
  classifier_calibrated: boolean;
  separator: string | null;
  asr: string | null;
  denoiser: string;
  notes: string[];
  categories: Record<Category, string>;
  presets: Record<string, { label: string; why: string; remove: Category[] }>;
  limits: { max_upload_mb: number; max_duration_s: number };
  auth_enabled: boolean;
}

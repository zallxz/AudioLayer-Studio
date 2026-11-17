import type { Category, Job, Peaks, Project, ProjectSummary, Segment, SystemInfo } from "./types";

const TOKEN_KEY = "audiolayer.token";
const BASE_KEY = "audiolayer.apiBase";

/**
 * Backend base URL. Empty = same origin (UI served by the backend, or the Vite dev proxy).
 * A hosted UI (e.g. on Vercel) points at a backend via VITE_API_BASE at build time or the
 * "Backend" field in the UI (stored per browser).
 */
export const getApiBase = (): string => {
  let v = "";
  try { v = localStorage.getItem(BASE_KEY) ?? ""; } catch { /* storage unavailable */ }
  return (v || (import.meta.env.VITE_API_BASE as string | undefined) || "").replace(/\/+$/, "");
};
export const setApiBase = (u: string) => {
  try { if (u.trim()) localStorage.setItem(BASE_KEY, u.trim()); else localStorage.removeItem(BASE_KEY); } catch { /* ignore */ }
};
export const getToken = (): string => {
  try { return localStorage.getItem(TOKEN_KEY) ?? ""; } catch { return ""; }
};
export const setToken = (t: string) => {
  try { localStorage.setItem(TOKEN_KEY, t); } catch { /* storage unavailable */ }
};

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public hint?: string | null) {
    super(message);
  }
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const tok = getToken();
  if (tok) headers.set("Authorization", `Bearer ${tok}`);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  let res: Response;
  try {
    res = await fetch(getApiBase() + path, { ...init, headers });
  } catch {
    throw new ApiError(0, "unreachable", `Cannot reach the AudioLayer backend at ${getApiBase() || location.origin}`);
  }
  if (!res.ok) {
    let body: any = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    const e = body?.error;
    throw new ApiError(res.status, e?.code ?? "http_error", e?.message ?? body?.detail?.[0]?.msg ?? res.statusText, e?.hint);
  }
  return res.json() as Promise<T>;
}

export const fileUrl = (projectId: string, rel: string, download = false) => {
  const q = new URLSearchParams();
  if (download) q.set("download", "true");
  const tok = getToken();
  if (tok) q.set("token", tok);
  const qs = q.toString();
  return `${getApiBase()}/api/projects/${projectId}/files/${rel}${qs ? `?${qs}` : ""}`;
};
export const mediaUrl = (projectId: string) => {
  const tok = getToken();
  return `${getApiBase()}/api/projects/${projectId}/media${tok ? `?token=${encodeURIComponent(tok)}` : ""}`;
};

export const api = {
  system: () => req<SystemInfo>("/api/system"),
  projects: () => req<ProjectSummary[]>("/api/projects"),
  project: (id: string) => req<Project>(`/api/projects/${id}`),
  deleteProject: (id: string) => req<{ deleted: string }>(`/api/projects/${id}`, { method: "DELETE" }),
  upload: (file: File, analyze: boolean, onProgress?: (f: number) => void) =>
    new Promise<{ project: Project; job?: Job }>((resolve, reject) => {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("analyze", String(analyze));
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${getApiBase()}/api/projects`);
      const tok = getToken();
      if (tok) xhr.setRequestHeader("Authorization", `Bearer ${tok}`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      xhr.onload = () => {
        let body: any = null;
        try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body);
        else reject(new ApiError(xhr.status, body?.error?.code ?? "upload_failed", body?.error?.message ?? "Upload failed", body?.error?.hint));
      };
      xhr.onerror = () => reject(new ApiError(0, "network", "Network error during upload"));
      xhr.send(fd);
    }),
  analyze: (id: string, opts: { separate: boolean; transcribe: boolean; denoise: string }) =>
    req<Job>(`/api/projects/${id}/analyze`, { method: "POST", body: JSON.stringify(opts) }),
  peaks: (id: string) => req<Peaks>(`/api/projects/${id}/peaks`),
  patchSegment: (id: string, segId: string, patch: Partial<Pick<Segment, "category" | "muted" | "solo" | "start" | "end">>) =>
    req<{ segment: Segment; render_job: Job | null; note: string | null }>(`/api/projects/${id}/segments/${segId}`, {
      method: "PATCH", body: JSON.stringify(patch),
    }),
  setMix: (id: string, body: { removed?: Category[]; preset?: string }) =>
    req<{ removed: Category[]; segments: Segment[]; limitations: string[] }>(`/api/projects/${id}/mix`, {
      method: "POST", body: JSON.stringify(body),
    }),
  undo: (id: string) => req<Project>(`/api/projects/${id}/undo`, { method: "POST" }),
  redo: (id: string) => req<Project>(`/api/projects/${id}/redo`, { method: "POST" }),
  reset: (id: string) => req<Project>(`/api/projects/${id}/reset`, { method: "POST" }),
  preview: (id: string) => req<Job>(`/api/projects/${id}/preview`, { method: "POST", body: "{}" }),
  export: (id: string, opts: Record<string, unknown>) =>
    req<Job>(`/api/projects/${id}/export`, { method: "POST", body: JSON.stringify(opts) }),
  job: (jobId: string) => req<Job>(`/api/jobs/${jobId}`),
  cancel: (jobId: string) => req<Job>(`/api/jobs/${jobId}/cancel`, { method: "POST" }),
};

export async function pollJob(jobId: string, onUpdate: (j: Job) => void, signal?: AbortSignal): Promise<Job> {
  for (;;) {
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    const j = await api.job(jobId);
    onUpdate(j);
    if (!["queued", "running"].includes(j.status)) return j;
    await new Promise((r) => setTimeout(r, 800));
  }
}

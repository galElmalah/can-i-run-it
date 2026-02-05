export type OllamaVersionResponse = {
  version: string;
};

export type OllamaModel = {
  name: string;
  model?: string;
  modified_at?: string;
  digest?: string;
  size?: number;
  details?: Record<string, unknown>;
};

export type OllamaTagsResponse = {
  models: OllamaModel[];
};

export type OllamaErrorKind = 'network' | 'cors_or_mixed' | 'http' | 'timeout' | 'unknown';

export type OllamaError = {
  kind: OllamaErrorKind;
  message: string;
  status?: number;
};

function trimSlash(s: string) {
  return s.replace(/\/+$/, '');
}

async function fetchJson<T>(url: string, opts: { timeoutMs: number }): Promise<T> {
  const ctrl = new AbortController();
  const t = window.setTimeout(() => ctrl.abort(), opts.timeoutMs);
  try {
    const res = await fetch(url, {
      method: 'GET',
      mode: 'cors',
      signal: ctrl.signal,
      headers: { Accept: 'application/json' },
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      const err: OllamaError = {
        kind: 'http',
        message: text || `HTTP ${res.status} from Ollama`,
        status: res.status,
      };
      throw err;
    }

    return (await res.json()) as T;
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      const err: OllamaError = { kind: 'timeout', message: `Timed out after ${opts.timeoutMs}ms` };
      throw err;
    }

    if (typeof e === 'object' && e && 'kind' in e) throw e;

    // Browsers intentionally hide details for CORS/mixed content; it often shows as TypeError.
    const msg = e instanceof Error ? e.message : String(e);
    const err: OllamaError = { kind: 'cors_or_mixed', message: msg || 'Failed to fetch' };
    throw err;
  } finally {
    window.clearTimeout(t);
  }
}

export async function getOllamaVersion(baseUrl: string, opts?: { timeoutMs?: number }) {
  const base = trimSlash(baseUrl);
  return await fetchJson<OllamaVersionResponse>(`${base}/api/version`, { timeoutMs: opts?.timeoutMs ?? 1600 });
}

export async function listOllamaModels(baseUrl: string, opts?: { timeoutMs?: number }) {
  const base = trimSlash(baseUrl);
  const res = await fetchJson<OllamaTagsResponse>(`${base}/api/tags`, { timeoutMs: opts?.timeoutMs ?? 2200 });
  return res.models ?? [];
}


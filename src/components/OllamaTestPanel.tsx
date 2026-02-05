import { useEffect, useMemo, useState } from 'react';
import type { OllamaError, OllamaModel } from '../lib/ollamaClient';
import { getOllamaVersion, listOllamaModels } from '../lib/ollamaClient';
import { CopyButton } from './CopyButton';

function formatBytes(bytes: number | undefined) {
  if (!bytes || !Number.isFinite(bytes)) return '—';
  const gb = bytes / 1024 / 1024 / 1024;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  const mb = bytes / 1024 / 1024;
  return `${mb.toFixed(0)} MB`;
}

function uniq(list: string[]) {
  return Array.from(new Set(list.filter(Boolean)));
}

export function OllamaTestPanel() {
  const [open, setOpen] = useState(false);
  const [baseUrl, setBaseUrl] = useState(() => {
    try {
      return localStorage.getItem('ollamaBaseUrl') || 'http://localhost:11434';
    } catch {
      return 'http://localhost:11434';
    }
  });

  const [status, setStatus] = useState<'idle' | 'checking' | 'ok' | 'error'>('idle');
  const [version, setVersion] = useState<string | null>(null);
  const [models, setModels] = useState<OllamaModel[] | null>(null);
  const [error, setError] = useState<OllamaError | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem('ollamaBaseUrl', baseUrl);
    } catch {
      // ignore
    }
  }, [baseUrl]);

  const origin = typeof window !== 'undefined' ? window.location.origin : '';
  const isHttps = typeof window !== 'undefined' ? window.location.protocol === 'https:' : false;
  const isHttpTarget = baseUrl.trim().startsWith('http://');

  const suggestedOrigins = useMemo(() => uniq([origin, 'http://localhost:5173']).join(','), [origin]);

  async function runCheck() {
    setStatus('checking');
    setError(null);
    setVersion(null);
    setModels(null);

    const [v, m] = await Promise.allSettled([
      getOllamaVersion(baseUrl, { timeoutMs: 1600 }),
      listOllamaModels(baseUrl, { timeoutMs: 2200 }),
    ]);

    const nextVersion = v.status === 'fulfilled' ? v.value.version : null;
    const nextModels = m.status === 'fulfilled' ? m.value : null;

    if (nextVersion) setVersion(nextVersion);
    if (nextModels) setModels(nextModels);

    const err =
      v.status === 'rejected'
        ? (v.reason as OllamaError)
        : m.status === 'rejected'
          ? (m.reason as OllamaError)
          : null;

    if (!nextVersion && !nextModels) {
      setStatus('error');
      setError(err ?? { kind: 'unknown', message: 'Unknown error' });
      return;
    }

    if (err) {
      setStatus('error');
      setError(err);
      return;
    }

    setStatus('ok');
  }

  const title = status === 'ok' ? 'Ollama: reachable' : status === 'error' ? 'Ollama: not reachable (yet)' : 'Test my Ollama';
  const tone = status === 'ok' ? 'green' : status === 'error' ? 'pink' : 'yellow';

  return (
    <details className="panel" open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="detailsSummary">
        <span className="panelTitle" style={{ margin: 0 }}>
          {title}
        </span>
        <span className="badge" data-tone={tone} aria-label="Ollama status badge">
          {status === 'checking' ? 'Checking…' : status.toUpperCase()}
        </span>
      </summary>

      <div className="fieldRow" style={{ marginTop: 12 }}>
        <div className="fieldRow">
          <div className="fieldLabel">Base URL</div>
          <div className="fieldHint">
            Usually <code>http://localhost:11434</code>. You can also try <code>http://127.0.0.1:11434</code>.
          </div>
          <div className="field">
            <input
              className="input"
              type="url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              aria-label="Ollama base URL"
            />
            <button className="btn" type="button" data-tone="green" onClick={runCheck} disabled={status === 'checking'}>
              {status === 'checking' ? 'Testing…' : 'Test'}
            </button>
          </div>
        </div>

        <div className="badgeRow" aria-label="Origin info">
          <span className="badge">
            Origin: <code>{origin || '—'}</code>
          </span>
          {isHttps && isHttpTarget ? (
            <span className="badge" data-tone="pink">
              HTTPS → HTTP localhost might be blocked
            </span>
          ) : null}
        </div>

        {status === 'ok' ? (
          <div className="fieldRow" aria-label="Ollama results">
            <div className="fieldLabel">Results</div>
            <div className="breakdownGrid">
              <div className="breakdownItem">
                <div className="breakdownKey">Version</div>
                <div className="breakdownVal">{version ?? '—'}</div>
              </div>
              <div className="breakdownItem">
                <div className="breakdownKey">Installed models</div>
                <div className="breakdownVal">{models ? models.length : '—'}</div>
              </div>
            </div>

            {models && models.length > 0 ? (
              <div className="fieldRow" style={{ gap: 10 }}>
                {models.slice(0, 20).map((m) => (
                  <div key={m.name} className="commandCard">
                    <div className="commandTop">
                      <div className="commandLabel">{m.name}</div>
                      <CopyButton text={`ollama run ${m.name}`} label="Copy run" tone="yellow" />
                    </div>
                    <div className="fieldHint">
                      Weights file size: <strong>{formatBytes(m.size)}</strong>
                    </div>
                  </div>
                ))}
                {models.length > 20 ? (
                  <div className="fieldHint">Showing first 20. (Yes, you’re a collector.)</div>
                ) : null}
              </div>
            ) : (
              <div className="fieldHint">No models found yet. Try running something once: <code>ollama run llama3.1</code></div>
            )}
          </div>
        ) : null}

        {status === 'error' ? (
          <div className="fieldRow" aria-label="Troubleshooting">
            <div className="fieldLabel">Troubleshooting</div>
            <div className="fieldHint">
              Browsers often hide the real reason (CORS / mixed-content / network) behind a generic “Failed to fetch”.
            </div>

            <div className="commandCard">
              <div className="commandTop">
                <div className="commandLabel">What I saw</div>
              </div>
              <pre className="command">
                <code>
                  {error?.kind ?? 'unknown'}: {error?.message ?? 'Unknown error'}
                </code>
              </pre>
            </div>

            <ul className="list">
              <li>
                Make sure Ollama is running. Try <code>ollama serve</code> (or just run any model once).
              </li>
              <li>
                If this page is on HTTPS, your browser may block calls to <code>http://localhost</code>. Easiest fix:
                run the app locally (<code>npm run dev</code>) when using the “Test” panel.
              </li>
              <li>
                If it’s a CORS issue, set <code>OLLAMA_ORIGINS</code> to include your site origin:
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', marginTop: 8 }}>
                  <code className="badge" style={{ textTransform: 'none' }}>
                    {suggestedOrigins}
                  </code>
                  <CopyButton text={suggestedOrigins} label="Copy origins" tone="green" />
                </div>
              </li>
              <li>
                Still stuck? Try switching base URL to <code>http://127.0.0.1:11434</code>.
              </li>
            </ul>
          </div>
        ) : null}
      </div>
    </details>
  );
}


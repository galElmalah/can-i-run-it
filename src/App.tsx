import { useEffect, useMemo, useState } from 'react';
import { OLLAMA_MODELS, searchModels, CATEGORY_LABELS, type Category as ModelCategory } from './data/models';
import { ModelCard } from './components/ModelCard';
import { ModelDetailView } from './components/ModelDetailView';
import { OllamaTestPanel } from './components/OllamaTestPanel';
import { SearchBar } from './components/SearchBar';
import { getVariantVerdict } from './lib/llmSizing';
import type { SystemInfo } from './lib/systemInfo';
import { getSystemInfo } from './lib/systemInfo';

type Category = ModelCategory | 'all';
type SortKey = 'popularity' | 'maxParams' | 'maxRunnableParams' | 'name';

const CEO_ROTATION = [
  { display: 'Altman', full: 'Sam Altman' },
  { display: 'Amodei', full: 'Dario Amodei' },
  { display: 'Pichai', full: 'Sundar Pichai' },
  { display: 'Hassabis', full: 'Demis Hassabis' },
  { display: 'Bezos', full: 'Jeff Bezos' },
  { display: 'Nadella', full: 'Satya Nadella' },
] as const;

function parsePullsNumeric(pulls: string | undefined): number {
  if (!pulls) return 0;
  const match = pulls.trim().match(/([\d,.]+)\s*(K|M|B)?/i);
  if (!match) return 0;

  const num = Number.parseFloat(match[1].replace(/,/g, ''));
  if (!Number.isFinite(num)) return 0;

  switch ((match[2] || '').toUpperCase()) {
    case 'K':
      return num * 1_000;
    case 'M':
      return num * 1_000_000;
    case 'B':
      return num * 1_000_000_000;
    default:
      return num;
  }
}

function getMaxParamsB(variants: Array<{ paramsB: number }>): number {
  let max = 0;
  for (const v of variants) max = Math.max(max, v.paramsB);
  return max;
}

function getMaxRunnableParamsB(args: {
  variants: Array<{ paramsB: number; sizeGB: number; quant: string; context: number; tag: string }>;
  ramGB: number | null;
  isMobile?: boolean;
}): number | null {
  if (!args.ramGB) return null;
  let max: number | null = null;
  for (const variant of args.variants) {
    const verdict = getVariantVerdict({ variant, ramGB: args.ramGB, isMobile: args.isMobile });
    if (verdict.status !== 'no') {
      max = max === null ? variant.paramsB : Math.max(max, variant.paramsB);
    }
  }
  return max;
}

export default function App() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [ramOverrideGB, setRamOverrideGB] = useState<number | null>(() => {
    try {
      const raw = localStorage.getItem('ramOverrideGB');
      if (!raw) return null;
      const n = Number(raw);
      return Number.isFinite(n) && n > 0 ? n : null;
    } catch {
      return null;
    }
  });

  useEffect(() => {
    setInfo(getSystemInfo());
  }, []);

  useEffect(() => {
    try {
      if (ramOverrideGB === null) localStorage.removeItem('ramOverrideGB');
      else localStorage.setItem('ramOverrideGB', String(ramOverrideGB));
    } catch {
      // ignore
    }
  }, [ramOverrideGB]);

  const effectiveRamGB = useMemo(() => ramOverrideGB ?? info?.memoryGB ?? null, [ramOverrideGB, info?.memoryGB]);

  // Search state
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<Category>('all');
  const [sortKey, setSortKey] = useState<SortKey>('popularity');

  // Fun rotating CTA
  const [ceoIdx, setCeoIdx] = useState(0);
  const [prevCeoIdx, setPrevCeoIdx] = useState<number | null>(null);
  const ceo = CEO_ROTATION[ceoIdx % CEO_ROTATION.length];
  const prevCeo = prevCeoIdx === null ? null : CEO_ROTATION[prevCeoIdx % CEO_ROTATION.length];
  useEffect(() => {
    const id = window.setInterval(() => {
      setCeoIdx((i) => {
        setPrevCeoIdx(i);
        return (i + 1) % CEO_ROTATION.length;
      });
    }, 5_000);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (prevCeoIdx === null) return;
    const id = window.setTimeout(() => setPrevCeoIdx(null), 520);
    return () => window.clearTimeout(id);
  }, [prevCeoIdx]);

  // Selected model (for detail view)
  const [selectedSlug, setSelectedSlug] = useState<string | null>(null);
  const selectedModel = selectedSlug ? OLLAMA_MODELS.find((m) => m.slug === selectedSlug) : null;

  // Filter models
  const filteredModels = useMemo(() => {
    let results = searchModels(query);
    if (category !== 'all') {
      results = results.filter((m) => m.category === category);
    }
    return results;
  }, [query, category]);

  const sortedModels = useMemo(() => {
    const ramGB = effectiveRamGB;
    const isMobile = info?.isMobile;

    const enriched = filteredModels.map((m) => ({
      model: m,
      popularity: parsePullsNumeric(m.pulls),
      maxParamsB: getMaxParamsB(m.variants),
      maxRunnableParamsB: getMaxRunnableParamsB({ variants: m.variants, ramGB, isMobile }),
    }));

    enriched.sort((a, b) => {
      switch (sortKey) {
        case 'popularity':
          return b.popularity - a.popularity || b.maxParamsB - a.maxParamsB || a.model.slug.localeCompare(b.model.slug);
        case 'maxParams':
          return b.maxParamsB - a.maxParamsB || b.popularity - a.popularity || a.model.slug.localeCompare(b.model.slug);
        case 'maxRunnableParams': {
          const ar = a.maxRunnableParamsB ?? -1;
          const br = b.maxRunnableParamsB ?? -1;
          return br - ar || b.popularity - a.popularity || a.model.slug.localeCompare(b.model.slug);
        }
        case 'name':
          return a.model.name.localeCompare(b.model.name) || b.popularity - a.popularity || a.model.slug.localeCompare(b.model.slug);
      }
    });

    return enriched.map((e) => e.model);
  }, [effectiveRamGB, filteredModels, info?.isMobile, sortKey]);

  // Category counts
  const categoryCounts = useMemo(() => {
    const base = searchModels(query);
    const counts: Record<string, number> = { all: base.length };
    for (const m of base) {
      counts[m.category] = (counts[m.category] ?? 0) + 1;
    }
    return counts;
  }, [query]);

  // If a model is selected, show detail view
  if (selectedModel) {
    return (
      <div className="container">
                <ModelDetailView
          model={selectedModel}
          ramGB={effectiveRamGB}
          cores={info?.cores ?? null}
          isMobile={info?.isMobile}
          onBack={() => setSelectedSlug(null)}
        />
      </div>
    );
  }

  return (
    <div className="container">
            <header className="pageHeader">
        <div className="headerTopRow">
          <h1 className="title">
            Run it or pay{' '}
            <span className="ceoSwap">
              {prevCeo && prevCeoIdx !== null && (
                <span key={`out-${prevCeo.display}-${ceoIdx}`} className="ceoName ceoNameOut" aria-hidden="true">
                  <span data-color={prevCeoIdx % 6}>{prevCeo.display}</span>?
                </span>
              )}
              <span
                key={`in-${ceo.display}-${ceoIdx}`}
                className={prevCeo ? 'ceoName ceoNameIn' : 'ceoName'}
                title={ceo.full}
              >
                <span data-color={ceoIdx % 6}>{ceo.display}</span>?
              </span>
            </span>
          </h1>
        </div>
        <p className="subtitle" style={{ opacity: 0.85, marginTop: 6 }}>
          Search {OLLAMA_MODELS.length} Ollama models and see which ones your machine can handle.
          Click a model to see all variants and detailed memory estimates.
        </p>
        <div className="badgeRow" aria-label="Project badges">
          <label className="ramBadge" data-tone="green">
            <input
              type="number"
              className="ramBadgeInput"
              min={1}
              max={512}
              value={ramOverrideGB ?? ''}
              placeholder={info?.memoryGB ? String(info.memoryGB) : '16'}
              onChange={(e) => {
                const n = Number(e.target.value);
                if (!e.target.value) return setRamOverrideGB(null);
                if (Number.isFinite(n)) setRamOverrideGB(Math.max(1, Math.round(n)));
              }}
              aria-label="RAM in GB"
            />
            <span>GB RAM</span>
          </label>
          <span className="badge">Ollama</span>
        </div>
        <p className="fieldHint" style={{ marginTop: 8 }}>
          You can edit this shit — set your RAM up top, then sort/filter below.
        </p>
      </header>

      <div style={{ marginTop: 14 }}>
        <SearchBar value={query} onChange={setQuery} placeholder="Search models (e.g., llama, tools, vision, 7b…)" />

        <div className="filterRow">
          {(['all', 'general', 'vision', 'thinking', 'embedding', 'tools'] as Category[]).map((cat) => {
            const count = categoryCounts[cat] ?? 0;
            if (cat !== 'all' && count === 0) return null;
            return (
              <button
                key={cat}
                type="button"
                className="chip"
                aria-pressed={category === cat}
                onClick={() => setCategory(cat)}
              >
                {cat === 'all' ? 'All' : CATEGORY_LABELS[cat]} ({count})
              </button>
            );
          })}
        </div>

        <div className="field" style={{ marginTop: 10, maxWidth: 320 }}>
          <span className="fieldLabel" style={{ margin: 0 }}>
            Sort
          </span>
          <select
            className="select"
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value as SortKey)}
            aria-label="Sort models"
          >
            <option value="popularity">Popularity</option>
            <option value="maxParams">Max parameters</option>
            <option value="maxRunnableParams">Max runnable parameters</option>
            <option value="name">Name (A–Z)</option>
          </select>
        </div>

        <p className="resultsCount">
          {sortedModels.length === OLLAMA_MODELS.length
            ? `${OLLAMA_MODELS.length} models`
            : `${sortedModels.length} of ${OLLAMA_MODELS.length} models`}
        </p>
      </div>

      {sortedModels.length === 0 ? (
        <div className="emptyState">
          <h3>No models found</h3>
          <p>Try a different search term or category.</p>
        </div>
      ) : (
        <div className="modelGrid">
          {sortedModels.map((model) => (
            <ModelCard
              key={model.slug}
              model={model}
              ramGB={effectiveRamGB}
              isMobile={info?.isMobile}
              onClick={() => setSelectedSlug(model.slug)}
            />
          ))}
        </div>
      )}

      <div className="divider" style={{ margin: '28px 0' }} />

      <div style={{ marginTop: 14 }}>
        <OllamaTestPanel />
      </div>

      <details className="panel faqPanel" style={{ marginTop: 14 }}>
        <summary className="detailsSummary">
          <span className="panelTitle" style={{ margin: 0 }}>
            How we calculate this
          </span>
          <span className="badge" data-tone="blue">FAQ</span>
        </summary>

        <div className="faqContent">
          <div className="faqItem">
            <h4 className="faqQuestion">How is memory usage estimated?</h4>
            <p className="faqAnswer">
              We use this formula: <strong>Total Memory = Model Weights + KV Cache + Runtime Overhead</strong>
            </p>
            <ul className="faqList">
              <li><strong>Model Weights:</strong> The file size plus 5% for metadata overhead</li>
              <li><strong>KV Cache:</strong> Memory that scales with context length (how much text the model can "remember")</li>
              <li><strong>Runtime Overhead:</strong> 1.2 GB base + 0.04 GB per billion parameters for CUDA buffers and fragmentation</li>
            </ul>
          </div>

          <div className="faqItem">
            <h4 className="faqQuestion">What do the quantization levels mean?</h4>
            <p className="faqAnswer">
              Quantization compresses model weights to use less memory at the cost of some quality. Lower numbers = smaller file, lower quality:
            </p>
            <ul className="faqList">
              <li><strong>Q4_K_M:</strong> 0.55 bytes/param — Best balance of size and quality (recommended)</li>
              <li><strong>Q5_K_M:</strong> 0.68 bytes/param — Slightly better quality</li>
              <li><strong>Q6_K:</strong> 0.78 bytes/param — Near-lossless</li>
              <li><strong>Q8_0:</strong> 1.07 bytes/param — High quality</li>
              <li><strong>F16:</strong> 2.0 bytes/param — Full precision (largest)</li>
            </ul>
          </div>

          <div className="faqItem">
            <h4 className="faqQuestion">Why does my OS affect the results?</h4>
            <p className="faqAnswer">
              Different operating systems reserve different amounts of RAM for system processes:
            </p>
            <ul className="faqList">
              <li><strong>Linux:</strong> ~1.5 GB base + 10% buffer — Most efficient for ML workloads</li>
              <li><strong>macOS:</strong> ~3.0 GB base + 15% buffer — Tighter memory management</li>
              <li><strong>Windows:</strong> ~3.5 GB base + 20% buffer — Higher system overhead</li>
              <li><strong>Mobile:</strong> ~2.0 GB base + 30% buffer — Aggressive memory limits</li>
            </ul>
          </div>

          <div className="faqItem">
            <h4 className="faqQuestion">What do the status colors mean?</h4>
            <ul className="faqList">
              <li><span className="badge" data-tone="green" style={{ fontSize: 10, padding: '4px 8px' }}>RUN</span> <strong>Comfortable:</strong> 2+ GB headroom — Will run smoothly</li>
              <li><span className="badge" style={{ fontSize: 10, padding: '4px 8px' }}>TIGHT</span> <strong>Tight fit:</strong> 0.5–2 GB headroom — Will run but close to the limit</li>
              <li><span className="badge" style={{ fontSize: 10, padding: '4px 8px' }}>MAYBE</span> <strong>Maybe:</strong> Under 0.5 GB headroom — Might work with tweaks (lower context, close apps)</li>
              <li><span className="badge" data-tone="pink" style={{ fontSize: 10, padding: '4px 8px' }}>NO</span> <strong>No go:</strong> Not enough memory for this model</li>
            </ul>
          </div>

          <div className="faqItem">
            <h4 className="faqQuestion">Why does context length matter?</h4>
            <p className="faqAnswer">
              Context length is how much text the model can process at once. Longer context = more KV cache memory.
              A 128K context model needs significantly more RAM than the same model at 8K context. If you're tight on memory,
              try a variant with shorter context.
            </p>
          </div>
        </div>
      </details>

      <p className="footerNote">
        This runs entirely in your browser. Model data is curated and may not reflect the latest Ollama library.
        Memory estimates are approximate.
      </p>
      <p className="footerNote" style={{ marginTop: 8, opacity: 0.5, fontSize: 10 }}>
        All references to tech CEOs are satirical and for humor only. We're not affiliated with any of them (and they definitely don't know we exist).
      </p>
    </div>
  );
}

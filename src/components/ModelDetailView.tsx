import { useState } from 'react';
import type { OllamaModelEntry, ModelVariant } from '../data/models';
import { CATEGORY_LABELS, CATEGORY_TONES } from '../data/models';
import {
  getFullVariantVerdict,
  getVariantVerdict,
  formatGB,
  getQuantMatrix,
  getUniqueParamSizes,
  QUANT_CONFIGS,
  QUANT_KEYS,
  type QuantMatrixRow,
} from '../lib/llmSizing';
import { CopyButton } from './CopyButton';

type Props = {
  model: OllamaModelEntry;
  ramGB: number | null;
  cores: number | null;
  isMobile?: boolean;
  onBack: () => void;
};

const STATUS_ICONS: Record<string, string> = {
  run: '✓',
  maybe: '~',
  no: '✗',
  unknown: '?',
};

// ============================================================================
// Quant Matrix Component
// ============================================================================

function QuantMatrix({
  matrixData,
  ramGB,
}: {
  matrixData: QuantMatrixRow[];
  ramGB: number | null;
}) {
  return (
    <div className="quantMatrixWrapper">
      <h3 className="quantMatrixTitle">Size vs Quantization</h3>
      <p className="fieldHint" style={{ marginBottom: 12 }}>
        Estimated memory requirements at 4K context. Actual tags may vary.
      </p>

      <div className="quantMatrix" role="table" aria-label="Quantization compatibility matrix">
        {/* Header Row */}
        <div className="quantMatrixHeaderRow" role="row">
          <div className="quantMatrixCorner" role="columnheader">Size</div>
          {QUANT_KEYS.map((quant) => (
            <div key={quant} className="quantMatrixColHeader" role="columnheader">
              <span className="quantLabel">{QUANT_CONFIGS[quant].label}</span>
              <span className="quantDesc">{QUANT_CONFIGS[quant].desc}</span>
            </div>
          ))}
        </div>

        {/* Data Rows */}
        {matrixData.map((row) => (
          <div key={row.paramsB} className="quantMatrixRow" role="row">
            <div className="quantMatrixRowHeader" role="rowheader">
              {row.label}
            </div>
            {QUANT_KEYS.map((quant) => {
              const cell = row.cells[quant];
              const statusClass = ramGB ? cell.status : 'unknown';
              return (
                <div
                  key={quant}
                  className="quantMatrixCell"
                  data-status={statusClass}
                  role="cell"
                  title={`${row.label} ${QUANT_CONFIGS[quant].label}: ${formatGB(cell.sizeGB)} download, ~${formatGB(cell.totalMemoryGB)} total memory`}
                >
                  <span className="cellIcon">{STATUS_ICONS[statusClass]}</span>
                  <span className="cellSize">{formatGB(cell.sizeGB)}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="quantMatrixLegend">
        <span className="legendItem" data-status="run">
          <span className="legendIcon">{STATUS_ICONS.run}</span> Can run
        </span>
        <span className="legendItem" data-status="maybe">
          <span className="legendIcon">{STATUS_ICONS.maybe}</span> Tight fit
        </span>
        <span className="legendItem" data-status="no">
          <span className="legendIcon">{STATUS_ICONS.no}</span> Too big
        </span>
        {!ramGB && (
          <span className="legendItem" data-status="unknown">
            <span className="legendIcon">{STATUS_ICONS.unknown}</span> Set RAM for verdict
          </span>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Variant Row Component
// ============================================================================

function VariantRow({
  variant,
  model,
  ramGB,
  cores,
  isMobile,
  isExpanded,
  onToggle,
}: {
  variant: ModelVariant;
  model: OllamaModelEntry;
  ramGB: number | null;
  cores: number | null;
  isMobile?: boolean;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const quickVerdict = getVariantVerdict({ variant, ramGB, isMobile });
  const fullVerdict = getFullVariantVerdict({
    variant,
    modelName: model.slug,
    ramGB,
    cores,
    isMobile,
  });

  const statusTone = quickVerdict.status === 'run' ? 'green' : quickVerdict.status === 'maybe' ? 'yellow' : 'pink';
  const command = `ollama run ${model.slug}:${variant.tag}`;

  return (
    <div className="variantRow" data-status={quickVerdict.status}>
      <button className="variantHeader" type="button" onClick={onToggle} aria-expanded={isExpanded}>
        <div className="variantLeft">
          <span className="variantTag">{variant.tag}</span>
          {variant.isDefault && <span className="variantDefault">default</span>}
        </div>

        <div className="variantMid">
          <span className="variantMeta">{variant.paramsB}B</span>
          <span className="variantMeta">{formatGB(variant.sizeGB)}</span>
          <span className="variantMeta">{variant.quant}</span>
          <span className="variantMeta">{variant.context >= 1024 ? `${Math.round(variant.context / 1024)}K ctx` : `${variant.context} ctx`}</span>
        </div>

        <div className="variantRight">
          <span className="variantStatus" data-tone={statusTone}>
            {quickVerdict.stamp}
          </span>
          <span className="variantChevron" aria-hidden="true">
            {isExpanded ? '▲' : '▼'}
          </span>
        </div>
      </button>

      {isExpanded && (
        <div className="variantDetail">
          <div className="stampCard" data-tone={statusTone} style={{ marginBottom: 12 }}>
            <div className="stampText" style={{ fontSize: 'clamp(24px, 4vw, 36px)' }}>
              {fullVerdict.stamp}
            </div>
            <div className="stampOneLiner">{fullVerdict.oneLiner}</div>
          </div>

          <div className="fieldRow" style={{ gap: 10 }}>
            <div className="fieldLabel">Memory estimate</div>
            <div className="breakdownGrid">
              <div className="breakdownItem">
                <div className="breakdownKey">Weights</div>
                <div className="breakdownVal">{formatGB(fullVerdict.estimate.weightsGB)}</div>
              </div>
              <div className="breakdownItem">
                <div className="breakdownKey">KV cache</div>
                <div className="breakdownVal">{formatGB(fullVerdict.estimate.kvCacheGB)}</div>
              </div>
              <div className="breakdownItem">
                <div className="breakdownKey">Overhead</div>
                <div className="breakdownVal">{formatGB(fullVerdict.estimate.runtimeOverheadGB)}</div>
              </div>
              <div className="breakdownItem" data-emphasis>
                <div className="breakdownKey">Total</div>
                <div className="breakdownVal">{formatGB(fullVerdict.estimate.totalGB)}</div>
              </div>
              <div className="breakdownItem">
                <div className="breakdownKey">Usable RAM</div>
                <div className="breakdownVal">
                  {fullVerdict.usableRamGB !== null ? formatGB(fullVerdict.usableRamGB) : '—'}
                </div>
              </div>
              <div className="breakdownItem">
                <div className="breakdownKey">Headroom</div>
                <div className="breakdownVal">
                  {quickVerdict.headroomGB !== null ? `${quickVerdict.headroomGB.toFixed(1)} GB` : '—'}
                </div>
              </div>
            </div>
          </div>

          {fullVerdict.notes.length > 0 && (
            <div className="fieldRow" style={{ marginTop: 10 }}>
              <div className="fieldLabel">Notes</div>
              <ul className="list">
                {fullVerdict.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </div>
          )}

          {fullVerdict.tips.length > 0 && (
            <div className="fieldRow" style={{ marginTop: 10 }}>
              <div className="fieldLabel">Tips</div>
              <ul className="list">
                {fullVerdict.tips.map((t) => (
                  <li key={t}>{t}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="commandCard" style={{ marginTop: 12 }}>
            <div className="commandTop">
              <div className="commandLabel">Run command</div>
              <CopyButton text={command} label="Copy" tone="green" />
            </div>
            <pre className="command">
              <code>{command}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export function ModelDetailView({ model, ramGB, cores, isMobile, onBack }: Props) {
  const tone = CATEGORY_TONES[model.category];

  // Track which variant is expanded (default to the default variant or first)
  const defaultIdx = model.variants.findIndex((v) => v.isDefault);
  const [expandedIdx, setExpandedIdx] = useState<number>(defaultIdx >= 0 ? defaultIdx : 0);

  // Count variants by status
  const counts = model.variants.reduce(
    (acc, v) => {
      const verdict = getVariantVerdict({ variant: v, ramGB, isMobile });
      acc[verdict.status]++;
      return acc;
    },
    { run: 0, maybe: 0, no: 0 } as Record<string, number>,
  );

  // Generate quant matrix data
  const uniqueSizes = getUniqueParamSizes(model.variants);
  const matrixData = getQuantMatrix({
    paramSizes: uniqueSizes,
    ramGB,
    context: 4096, // Default context for matrix comparison
    isMobile,
  });

  return (
    <div className="modelDetail">
      <button className="btn backBtn" type="button" onClick={onBack} data-tone="ink">
        ← Back to search
      </button>

      <header className="modelDetailHeader">
        <div className="badgeRow" style={{ marginBottom: 8 }}>
          <span className="badge" data-tone={tone}>
            {CATEGORY_LABELS[model.category]}
          </span>
          <span className="badge">{model.publisher}</span>
          {model.pulls && <span className="badge">{model.pulls} pulls</span>}
        </div>

        <h1 className="title" style={{ fontSize: 'clamp(28px, 4.5vw, 44px)' }}>
          {model.name}
        </h1>
        <p className="subtitle">{model.description}</p>

        <a
          href={`https://ollama.com/library/${model.slug}`}
          target="_blank"
          rel="noopener noreferrer"
          className="ollamaLink"
        >
          View on Ollama ↗
        </a>

        <div className="badgeRow" style={{ marginTop: 12 }}>
          {counts.run > 0 && (
            <span className="badge" data-tone="green">
              {counts.run} can run
            </span>
          )}
          {counts.maybe > 0 && (
            <span className="badge">
              {counts.maybe} maybe
            </span>
          )}
          {counts.no > 0 && (
            <span className="badge" data-tone="pink">
              {counts.no} too big
            </span>
          )}
        </div>
      </header>

      {/* Quant Matrix */}
      <QuantMatrix matrixData={matrixData} ramGB={ramGB} />

      <div className="divider" style={{ margin: '20px 0' }} />

      <h2 className="panelTitle">Variants ({model.variants.length})</h2>
      <p className="fieldHint" style={{ marginBottom: 12 }}>
        Click a variant to see the full memory breakdown and verdict for your machine.
      </p>

      <div className="variantList">
        {model.variants.map((variant, idx) => (
          <VariantRow
            key={variant.tag}
            variant={variant}
            model={model}
            ramGB={ramGB}
            cores={cores}
            isMobile={isMobile}
            isExpanded={expandedIdx === idx}
            onToggle={() => setExpandedIdx(expandedIdx === idx ? -1 : idx)}
          />
        ))}
      </div>

      <p className="footerNote" style={{ marginTop: 20 }}>
        Memory estimates are approximate and intentionally conservative. Actual usage depends on your exact hardware,
        GPU offload, and running applications.
      </p>
    </div>
  );
}

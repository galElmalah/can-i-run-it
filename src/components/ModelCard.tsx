import type { OllamaModelEntry } from '../data/models';
import { CATEGORY_LABELS, CATEGORY_TONES } from '../data/models';
import {
  getBestRunnableVariant,
  getVariantVerdict,
  getUniqueParamSizes,
} from '../lib/llmSizing';

type Props = {
  model: OllamaModelEntry;
  ramGB: number | null;
  isMobile?: boolean;
  onClick: () => void;
};

const STATUS_ICONS: Record<string, string> = {
  run: '✓',
  maybe: '~',
  no: '✗',
};

type SizePillData = {
  paramsB: number;
  label: string;
  status: 'run' | 'maybe' | 'no';
};

function formatParamsLabel(paramsB: number): string {
  return paramsB < 1 ? `${(paramsB * 1000).toFixed(0)}M` : `${paramsB}B`;
}

function getSizePills(
  model: OllamaModelEntry,
  ramGB: number | null,
  isMobile?: boolean
): SizePillData[] {
  const uniqueSizes = getUniqueParamSizes(model.variants);

  return uniqueSizes.map((paramsB) => {
    // Find the variant with this param size (prefer default quant)
    const variant = model.variants.find((v) => v.paramsB === paramsB)!;
    const verdict = getVariantVerdict({ variant, ramGB, isMobile });

    return {
      paramsB,
      label: formatParamsLabel(paramsB),
      status: ramGB ? verdict.status : 'maybe',
    };
  });
}

export function ModelCard({ model, ramGB, isMobile, onClick }: Props) {
  const displayCategory = model.category;
  const tone = CATEGORY_TONES[displayCategory];
  const best = getBestRunnableVariant({ variants: model.variants, ramGB, isMobile });

  // Grounded in scraped `capabilities`
  const supportsTools = model.tags.includes('tools');

  // Get size pills data
  const sizePills = getSizePills(model, ramGB, isMobile);
  const maxParamsB = sizePills.reduce((max, p) => Math.max(max, p.paramsB), 0);
  const runnableSizes = ramGB ? sizePills.filter((p) => p.status !== 'no') : [];
  const maxRunnableParamsB =
    ramGB && runnableSizes.length > 0 ? runnableSizes.reduce((max, p) => Math.max(max, p.paramsB), 0) : null;

  const MAX_PILLS = 5;
  const visiblePills = sizePills.slice(0, MAX_PILLS);
  const hiddenCount = sizePills.length - MAX_PILLS;

  // Determine overall card status
  const hasRunnable = best?.verdict.status === 'run';
  const hasMaybe = best?.verdict.status === 'maybe';
  const statusTone = hasRunnable ? 'green' : hasMaybe ? 'yellow' : ramGB ? 'pink' : undefined;

  return (
    <button className="modelCard" type="button" onClick={onClick} data-status={statusTone}>
      <div className="modelCardTop">
        <span className="badge" data-tone={tone} style={{ fontSize: 10, padding: '5px 8px' }}>
          {CATEGORY_LABELS[displayCategory]}
        </span>
        {supportsTools && (
          <img
            className="toolsIcon"
            src="/tools-icon.png"
            alt="Supports tools"
            width="16"
            height="16"
          />
        )}
      </div>

      <h3 className="modelCardName">{model.name}</h3>
      <p className="modelCardDesc">{model.description}</p>

      {/* Size Pills */}
      <div className="sizePillRow">
        {visiblePills.map((pill) => (
          <span
            key={pill.paramsB}
            className="sizePill"
            data-status={pill.status}
            title={`${pill.label}: ${pill.status === 'run' ? 'Can run' : pill.status === 'maybe' ? 'Tight fit' : 'Too big'}`}
          >
            {pill.label}
            <span className="sizePillIcon">{ramGB ? STATUS_ICONS[pill.status] : '?'}</span>
          </span>
        ))}
        {hiddenCount > 0 && (
          <span className="sizePillMore">+{hiddenCount}</span>
        )}
      </div>

      <div className="modelCardMeta">
        <span className="modelCardMetaItem">{model.publisher}</span>
        <span className="modelCardMetaItem">{model.variants.length} variant{model.variants.length !== 1 ? 's' : ''}</span>
        <span className="modelCardMetaItem">max {formatParamsLabel(maxParamsB)}</span>
        {ramGB && (
          <span className="modelCardMetaItem">
            {maxRunnableParamsB === null ? 'runs: none' : `runs up to ${formatParamsLabel(maxRunnableParamsB)}`}
          </span>
        )}
      </div>
    </button>
  );
}

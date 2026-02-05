import type { ModelVariant } from '../data/models';

// ============================================================================
// Memory Calculation Constants (Research-backed)
// ============================================================================

/**
 * Quantization configurations with accurate bytes-per-parameter values.
 * Based on llama.cpp documentation and community benchmarks (2025-2026).
 * 
 * The bytes/param includes quantization overhead and metadata.
 */
export const QUANT_CONFIGS = {
  Q2_K: { bytesPerParam: 0.31, label: 'Q2', desc: 'Extreme compression, lower quality' },
  Q3_K_M: { bytesPerParam: 0.41, label: 'Q3', desc: 'Very small, acceptable quality' },
  Q4_0: { bytesPerParam: 0.55, label: 'Q4_0', desc: 'Basic 4-bit' },
  Q4_K_M: { bytesPerParam: 0.55, label: 'Q4', desc: 'Best balance (recommended)' },
  Q5_K_M: { bytesPerParam: 0.68, label: 'Q5', desc: 'Good quality, moderate size' },
  Q6_K: { bytesPerParam: 0.78, label: 'Q6', desc: 'High quality' },
  Q8_0: { bytesPerParam: 1.07, label: 'Q8', desc: 'Near full precision' },
  F16: { bytesPerParam: 2.0, label: 'F16', desc: 'Full precision' },
} as const;

export type QuantKey = keyof typeof QUANT_CONFIGS;
export const QUANT_KEYS: QuantKey[] = ['Q4_K_M', 'Q5_K_M', 'Q8_0', 'F16'];
export const ALL_QUANT_KEYS: QuantKey[] = ['Q2_K', 'Q3_K_M', 'Q4_0', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0', 'F16'];

/**
 * Runtime overhead baseline (CUDA context, inference buffers, memory fragmentation).
 * This is added on top of model weights + KV cache.
 */
const RUNTIME_OVERHEAD_BASE_GB = 1.2;

/**
 * Additional overhead per billion parameters (for compute buffers, activations).
 */
const RUNTIME_OVERHEAD_PER_PARAM_B = 0.04;

/**
 * Estimate download/weight size for a given param count and quantization.
 */
export function estimateQuantSizeGB(paramsB: number, quant: QuantKey): number {
  const config = QUANT_CONFIGS[quant];
  return paramsB * config.bytesPerParam;
}

// ============================================================================
// Types
// ============================================================================

export type MemoryEstimate = {
  weightsGB: number;
  kvCacheGB: number;
  runtimeOverheadGB: number;
  totalGB: number;
};

export type VariantVerdict = {
  status: 'run' | 'maybe' | 'no';
  stamp: string;
  shortLabel: string;
  estimate: MemoryEstimate;
  headroomGB: number | null;
};

export type FullVerdict = {
  status: 'run' | 'maybe' | 'no' | 'unknown';
  stamp: string;
  oneLiner: string;
  notes: string[];
  tips: string[];
  estimate: MemoryEstimate;
  usableRamGB: number | null;
  assumedReserveGB: number | null;
};

function clamp(n: number, min: number, max: number) {
  return Math.min(max, Math.max(min, n));
}

/**
 * Estimate KV cache GB per 1K tokens based on param size.
 * 
 * Modern models (2024+) typically use GQA (Grouped Query Attention) which
 * dramatically reduces KV cache size. For example:
 * - Llama 3.1 8B: ~8 KV heads (vs 32 attention heads) = 4x reduction
 * - Llama 3.1 70B: ~8 KV heads (vs 64 attention heads) = 8x reduction
 * 
 * Formula (full precision): 2 × layers × kv_heads × head_dim × 2 bytes × tokens
 * Most GGUF models store KV cache in FP16 (2 bytes) or FP8 (1 byte).
 * 
 * These estimates are conservative, assuming typical GQA architectures.
 */
function kvGBPer1kTokens(paramsB: number): number {
  // Use lookup table with interpolation for smooth estimates
  if (paramsB <= 1) return 0.03;
  if (paramsB <= 3) return 0.06;
  if (paramsB <= 8) return 0.08;
  if (paramsB <= 14) return 0.12;
  if (paramsB <= 35) return 0.20;
  if (paramsB <= 72) return 0.35;
  if (paramsB <= 200) return 0.50;
  return 0.80;
}

/**
 * Detect if a model likely uses MoE (Mixture of Experts) architecture.
 * MoE models have different memory characteristics - only ~2 experts active at once.
 */
function isMoeModel(tag: string): boolean {
  return /\d+x\d+b/i.test(tag);
}

/**
 * Get effective parameters for memory calculation.
 * For MoE models, this returns the active parameters during inference.
 */
function getEffectiveParams(variant: ModelVariant): number {
  if (isMoeModel(variant.tag)) {
    // MoE: typically 2 experts active, plus shared layers
    // The sizeGB already accounts for this in most cases
    return variant.paramsB;
  }
  return variant.paramsB;
}

/**
 * Estimate memory needed for a specific model variant.
 * 
 * Memory formula: Weights + KV Cache + Runtime Overhead
 * 
 * - Weights: Model file size (already quantized in Ollama)
 * - KV Cache: Scales linearly with context length
 * - Runtime Overhead: CUDA context, compute buffers, memory fragmentation
 * 
 * The sizeGB from our data is the download/disk size, which closely matches
 * the VRAM/RAM usage for weights in GGUF format.
 */
export function estimateVariantMemoryGB(variant: ModelVariant, contextTokens?: number): MemoryEstimate {
  // Weights: Use model's stated size (already quantized, typically Q4_K_M)
  // Add ~5% overhead for model metadata and alignment
  const weightsGB = variant.sizeGB > 0 
    ? variant.sizeGB * 1.05 
    : variant.paramsB * QUANT_CONFIGS.Q4_K_M.bytesPerParam * 1.05;

  // KV cache: Scales with context length
  // Use model's default context if not specified
  const ctx = contextTokens ?? Math.min(variant.context, 8192); // Cap at 8K for default estimate
  const effectiveParams = getEffectiveParams(variant);
  const kvCacheGB = (ctx / 1024) * kvGBPer1kTokens(effectiveParams);

  // Runtime overhead: Base + per-param scaling
  // Includes CUDA context (~0.5GB), compute buffers, activation memory, fragmentation
  const runtimeOverheadGB = clamp(
    RUNTIME_OVERHEAD_BASE_GB + effectiveParams * RUNTIME_OVERHEAD_PER_PARAM_B,
    1.0,  // Minimum 1GB overhead
    6.0   // Cap at 6GB for very large models
  );

  const totalGB = weightsGB + kvCacheGB + runtimeOverheadGB;
  return { weightsGB, kvCacheGB, runtimeOverheadGB, totalGB };
}

/**
 * Estimate memory for a custom context length scenario.
 * Useful for showing impact of different context settings.
 */
export function estimateMemoryWithContext(
  variant: ModelVariant, 
  contextTokens: number
): MemoryEstimate & { contextImpactGB: number } {
  const baseEstimate = estimateVariantMemoryGB(variant, variant.context);
  const customEstimate = estimateVariantMemoryGB(variant, contextTokens);
  
  return {
    ...customEstimate,
    contextImpactGB: customEstimate.kvCacheGB - baseEstimate.kvCacheGB,
  };
}

/**
 * System memory overhead by platform.
 * Based on research: OS + typical background apps memory usage.
 */
const PLATFORM_OVERHEAD: Record<string, { base: number; percent: number }> = {
  macos: { base: 3.0, percent: 0.15 },    // macOS: 3-4GB base + 15% buffer
  windows: { base: 3.5, percent: 0.20 },  // Windows: 3.5-5GB base + 20% buffer
  linux: { base: 1.5, percent: 0.10 },    // Linux: 1.5-2GB base + 10% buffer
  mobile: { base: 2.0, percent: 0.30 },   // Mobile: Aggressive memory management
  default: { base: 3.0, percent: 0.20 },  // Conservative default
};

export type PlatformType = 'macos' | 'windows' | 'linux' | 'mobile' | 'default';

/**
 * Estimate usable RAM after reserving space for OS + other apps.
 * 
 * Research-backed recommendations:
 * - Keep 2-4GB reserved for OS and basic apps
 * - Reserve additional 15-25% as buffer for memory pressure
 * - Mobile devices need more aggressive reservation (30%+)
 * 
 * For LLM inference, we want to avoid swap/paging at all costs
 * as it devastates performance (100-1000x slower).
 */
export function estimateUsableRamGB(
  ramGB: number, 
  opts?: { isMobile?: boolean; platform?: PlatformType }
): { usableRamGB: number; assumedReserveGB: number } {
  // Determine platform
  let platform: PlatformType = opts?.platform || 'default';
  if (opts?.isMobile) platform = 'mobile';
  
  const overhead = PLATFORM_OVERHEAD[platform];
  
  // Calculate reserve: base + percentage of total RAM
  const baseReserve = overhead.base;
  const percentReserve = ramGB * overhead.percent;
  
  // Total reserve, clamped to reasonable bounds
  const assumedReserveGB = clamp(baseReserve + percentReserve, 2.5, 16);
  
  // Usable RAM (never less than 1GB to show meaningful results)
  const usableRamGB = Math.max(1, ramGB - assumedReserveGB);
  
  return { usableRamGB, assumedReserveGB };
}

/**
 * Detect platform from user agent (browser environment).
 * Returns platform type for memory estimation.
 */
export function detectPlatform(): PlatformType {
  if (typeof navigator === 'undefined') return 'default';
  
  const ua = navigator.userAgent.toLowerCase();
  const platform = (navigator as { userAgentData?: { platform?: string } })
    .userAgentData?.platform?.toLowerCase() || '';
  
  if (/iphone|ipad|android/i.test(ua)) return 'mobile';
  if (/mac/i.test(platform) || /mac/i.test(ua)) return 'macos';
  if (/win/i.test(platform) || /windows/i.test(ua)) return 'windows';
  if (/linux/i.test(platform) || /linux/i.test(ua)) return 'linux';
  
  return 'default';
}

/**
 * Headroom thresholds for verdict classification.
 * 
 * - COMFORTABLE: Plenty of room, should run smoothly
 * - TIGHT: Will run but may have occasional slowdowns
 * - MAYBE: Might work with reduced context or closed apps
 * - NO: Definitely won't fit, need smaller model
 */
const HEADROOM_THRESHOLDS = {
  COMFORTABLE: 2.0,  // 2GB+ headroom = comfortable
  TIGHT: 0.5,        // 0.5-2GB = tight but workable
  MAYBE: -1.5,       // -1.5 to 0.5 = might work with adjustments
  // Below -1.5 = NO
};

/**
 * Quick verdict for a single variant (used in model cards grid).
 * 
 * Uses conservative estimates to avoid disappointing users.
 * Better to under-promise and over-deliver.
 */
export function getVariantVerdict(args: {
  variant: ModelVariant;
  ramGB: number | null;
  contextTokens?: number;
  isMobile?: boolean;
  platform?: PlatformType;
}): VariantVerdict {
  const estimate = estimateVariantMemoryGB(args.variant, args.contextTokens);

  if (!args.ramGB) {
    return {
      status: 'maybe',
      stamp: '?',
      shortLabel: 'Set RAM',
      estimate,
      headroomGB: null,
    };
  }

  const { usableRamGB } = estimateUsableRamGB(args.ramGB, { 
    isMobile: args.isMobile,
    platform: args.platform,
  });
  const headroomGB = usableRamGB - estimate.totalGB;

  // Comfortable fit: Should run well
  if (headroomGB >= HEADROOM_THRESHOLDS.COMFORTABLE) {
    return {
      status: 'run',
      stamp: 'YES',
      shortLabel: 'Can run',
      estimate,
      headroomGB,
    };
  }

  // Tight fit: Will work but may need care
  if (headroomGB >= HEADROOM_THRESHOLDS.TIGHT) {
    return {
      status: 'maybe',
      stamp: 'TIGHT',
      shortLabel: 'Tight fit',
      estimate,
      headroomGB,
    };
  }

  // Maybe: Borderline, might work with adjustments
  if (headroomGB >= HEADROOM_THRESHOLDS.MAYBE) {
    return {
      status: 'maybe',
      stamp: 'MAYBE',
      shortLabel: 'Might work',
      estimate,
      headroomGB,
    };
  }

  // No: Definitely too big
  return {
    status: 'no',
    stamp: 'NO',
    shortLabel: 'Too big',
    estimate,
    headroomGB,
  };
}

/**
 * Full detailed verdict for a variant (used in detail view).
 * Provides comprehensive analysis with actionable recommendations.
 */
export function getFullVariantVerdict(args: {
  variant: ModelVariant;
  modelName: string;
  ramGB: number | null;
  cores: number | null;
  contextTokens?: number;
  isMobile?: boolean;
  platform?: PlatformType;
}): FullVerdict {
  const estimate = estimateVariantMemoryGB(args.variant, args.contextTokens);
  const notes: string[] = [];
  const tips: string[] = [];

  // CPU analysis (affects inference speed, not runnability)
  if (args.cores) {
    if (args.cores >= 16) {
      notes.push('CPU: Excellent (16+ cores) — fast inference expected.');
    } else if (args.cores >= 10) {
      notes.push('CPU: Good (10-15 cores) — solid performance.');
    } else if (args.cores >= 8) {
      notes.push('CPU: Decent (8-9 cores) — reasonable speed.');
    } else if (args.cores >= 6) {
      notes.push('CPU: Moderate (6-7 cores) — some waiting on larger models.');
    } else {
      notes.push('CPU: Limited (<6 cores) — stick to smaller models for speed.');
    }
  }

  // Model size analysis
  const paramsB = args.variant.paramsB;
  if (paramsB <= 1) {
    notes.push('Tiny model (<1B) — ultra-fast, fits anywhere.');
  } else if (paramsB <= 3) {
    notes.push('Small model (1-3B) — fast and light, good for most tasks.');
  } else if (paramsB <= 8) {
    notes.push('Medium model (3-8B) — sweet spot for quality/speed balance.');
  } else if (paramsB <= 14) {
    notes.push('Mid-size model (8-14B) — needs decent RAM.');
  } else if (paramsB <= 35) {
    notes.push('Large model (14-35B) — high quality, needs serious RAM.');
  } else if (paramsB <= 72) {
    notes.push('Very large model (35-72B) — excellent quality, 64GB+ recommended.');
  } else {
    notes.push('Massive model (70B+) — top-tier quality, workstation-class hardware needed.');
  }

  // Memory breakdown note
  notes.push(
    `Memory breakdown: ~${estimate.weightsGB.toFixed(1)}GB weights + ` +
    `~${estimate.kvCacheGB.toFixed(1)}GB KV cache + ` +
    `~${estimate.runtimeOverheadGB.toFixed(1)}GB overhead.`
  );

  if (!args.ramGB) {
    return {
      status: 'unknown',
      stamp: 'MAYBE?',
      oneLiner: "Set your RAM to get an accurate verdict.",
      notes: [...notes, "Browser couldn't detect RAM. Set an override for accurate sizing."],
      tips: [
        '8GB RAM → 1-3B models max',
        '16GB RAM → 7-8B models comfortably',
        '32GB RAM → 13-14B models with headroom',
        '64GB RAM → 34B models, 70B tight',
        '128GB+ RAM → Run almost anything',
      ],
      estimate,
      usableRamGB: null,
      assumedReserveGB: null,
    };
  }

  const { usableRamGB, assumedReserveGB } = estimateUsableRamGB(args.ramGB, { 
    isMobile: args.isMobile,
    platform: args.platform,
  });
  const margin = usableRamGB - estimate.totalGB;

  // Comfortable fit
  if (margin >= HEADROOM_THRESHOLDS.COMFORTABLE) {
    notes.push(`Headroom: ~${margin.toFixed(1)} GB available — comfortable fit.`);
    tips.push(`Run: ollama run ${args.modelName}:${args.variant.tag}`);
    tips.push('You can increase context length if needed.');
    if (paramsB >= 7) {
      tips.push('For faster responses, close other memory-heavy apps.');
    }
    return {
      status: 'run',
      stamp: 'RUN IT',
      oneLiner: 'Should run smoothly on your system.',
      notes,
      tips,
      estimate,
      usableRamGB,
      assumedReserveGB,
    };
  }

  // Tight fit
  if (margin >= HEADROOM_THRESHOLDS.TIGHT) {
    notes.push(`Headroom: ~${margin.toFixed(1)} GB — workable but tight.`);
    tips.push('Close Chrome/browsers before running (they use lots of RAM).');
    tips.push('Keep context length at default or lower.');
    tips.push(`Run: ollama run ${args.modelName}:${args.variant.tag}`);
    return {
      status: 'maybe',
      stamp: 'TIGHT FIT',
      oneLiner: 'Should work, but close other apps first.',
      notes,
      tips,
      estimate,
      usableRamGB,
      assumedReserveGB,
    };
  }

  // Maybe (borderline)
  if (margin >= HEADROOM_THRESHOLDS.MAYBE) {
    const shortBy = Math.abs(margin);
    notes.push(`Borderline: ~${shortBy.toFixed(1)} GB over ideal, but might work.`);
    tips.push('Try with minimal context (use --context-length 2048 flag).');
    tips.push('Quit all other applications before running.');
    tips.push('If it crashes, try the next smaller variant.');
    return {
      status: 'maybe',
      stamp: 'MAYBE',
      oneLiner: 'Borderline — might work with adjustments.',
      notes,
      tips,
      estimate,
      usableRamGB,
      assumedReserveGB,
    };
  }

  // No fit
  const overBy = Math.abs(margin);
  notes.push(`Short by ~${overBy.toFixed(1)} GB — won't fit comfortably.`);
  
  // Suggest alternatives
  tips.push('Try a smaller variant of this model if available.');
  
  // Calculate what RAM they'd need
  const neededRam = Math.ceil(estimate.totalGB + assumedReserveGB + 2);
  tips.push(`You'd need ~${neededRam}GB RAM for this model.`);
  
  // Suggest appropriate model sizes
  const maxParamsB = getMaxRunnableParams(args.ramGB);
  if (maxParamsB > 0) {
    tips.push(`With ${args.ramGB}GB RAM, aim for models ≤${maxParamsB}B parameters.`);
  }
  
  return {
    status: 'no',
    stamp: 'NOT TODAY',
    oneLiner: "This model won't fit in your available RAM.",
    notes,
    tips,
    estimate,
    usableRamGB,
    assumedReserveGB,
  };
}

/**
 * Estimate the max model size (in billions of params) that can run on given RAM.
 * Assumes Q4_K_M quantization (typical for Ollama).
 */
export function getMaxRunnableParams(ramGB: number): number {
  const { usableRamGB } = estimateUsableRamGB(ramGB);
  // Rough inverse: usableRAM ≈ paramsB * 0.55 + overhead
  // Solving for paramsB: paramsB ≈ (usableRAM - overhead) / 0.55
  const overhead = 2.0; // Conservative overhead
  const maxParams = (usableRamGB - overhead) / 0.7; // Use 0.7 to be conservative
  return Math.max(0, Math.floor(maxParams));
}

/**
 * Get the best runnable variant from a list (for showing "recommended" in cards).
 * Prioritizes: largest model that runs comfortably > largest that's tight fit > any maybe
 */
export function getBestRunnableVariant(args: {
  variants: ModelVariant[];
  ramGB: number | null;
  isMobile?: boolean;
  platform?: PlatformType;
}): { variant: ModelVariant; verdict: VariantVerdict } | null {
  if (!args.ramGB) return null;

  // Sort by params descending (prefer larger models for quality)
  const sorted = [...args.variants].sort((a, b) => b.paramsB - a.paramsB);
  
  // First pass: find largest that runs comfortably
  for (const variant of sorted) {
    const verdict = getVariantVerdict({ 
      variant, 
      ramGB: args.ramGB, 
      isMobile: args.isMobile,
      platform: args.platform,
    });
    if (verdict.status === 'run') {
      return { variant, verdict };
    }
  }

  // Second pass: find largest "maybe" (tight fit)
  for (const variant of sorted) {
    const verdict = getVariantVerdict({ 
      variant, 
      ramGB: args.ramGB, 
      isMobile: args.isMobile,
      platform: args.platform,
    });
    if (verdict.status === 'maybe') {
      return { variant, verdict };
    }
  }

  return null;
}

/**
 * Get all runnable variants for a model, sorted by recommendation.
 */
export function getRunnableVariants(args: {
  variants: ModelVariant[];
  ramGB: number | null;
  isMobile?: boolean;
  platform?: PlatformType;
}): Array<{ variant: ModelVariant; verdict: VariantVerdict }> {
  if (!args.ramGB) return [];

  return args.variants
    .map(variant => ({
      variant,
      verdict: getVariantVerdict({
        variant,
        ramGB: args.ramGB,
        isMobile: args.isMobile,
        platform: args.platform,
      }),
    }))
    .filter(({ verdict }) => verdict.status !== 'no')
    .sort((a, b) => {
      // Sort by status (run > maybe), then by params descending
      if (a.verdict.status !== b.verdict.status) {
        return a.verdict.status === 'run' ? -1 : 1;
      }
      return b.variant.paramsB - a.variant.paramsB;
    });
}

export function formatGB(n: number) {
  if (n < 1) return `${(n * 1024).toFixed(0)} MB`;
  return `${n.toFixed(1)} GB`;
}

// ============================================================================
// Quant Matrix Functions (Educational / What-If Analysis)
// ============================================================================

/**
 * Note: The quant matrix is primarily educational, showing what different
 * quantization levels would require. In practice, Ollama models are pre-quantized
 * (typically Q4_K_M) and the variant.sizeGB reflects the actual download size.
 */

export type QuantCellVerdict = {
  status: 'run' | 'maybe' | 'no' | 'unknown';
  sizeGB: number;
  totalMemoryGB: number;
  headroomGB: number | null;
};

/**
 * Get verdict for a specific param size + quant combination.
 * Useful for "what-if" analysis when comparing quantization options.
 */
export function getQuantCellVerdict(args: {
  paramsB: number;
  quant: QuantKey;
  ramGB: number | null;
  context?: number;
  isMobile?: boolean;
  platform?: PlatformType;
}): QuantCellVerdict {
  const sizeGB = estimateQuantSizeGB(args.paramsB, args.quant);
  const ctx = args.context ?? 4096; // Default to 4K context for matrix

  // Calculate total memory needed
  const weightsGB = sizeGB * 1.05; // Add 5% metadata overhead
  const kvCacheGB = (ctx / 1024) * kvGBPer1kTokens(args.paramsB);
  const runtimeOverheadGB = clamp(
    RUNTIME_OVERHEAD_BASE_GB + args.paramsB * RUNTIME_OVERHEAD_PER_PARAM_B,
    1.0,
    6.0
  );
  const totalMemoryGB = weightsGB + kvCacheGB + runtimeOverheadGB;

  if (!args.ramGB) {
    return { status: 'unknown', sizeGB, totalMemoryGB, headroomGB: null };
  }

  const { usableRamGB } = estimateUsableRamGB(args.ramGB, { 
    isMobile: args.isMobile,
    platform: args.platform,
  });
  const headroomGB = usableRamGB - totalMemoryGB;

  if (headroomGB >= HEADROOM_THRESHOLDS.COMFORTABLE) {
    return { status: 'run', sizeGB, totalMemoryGB, headroomGB };
  }
  if (headroomGB >= HEADROOM_THRESHOLDS.MAYBE) {
    return { status: 'maybe', sizeGB, totalMemoryGB, headroomGB };
  }
  return { status: 'no', sizeGB, totalMemoryGB, headroomGB };
}

export type QuantMatrixRow = {
  paramsB: number;
  label: string; // e.g., "7B"
  cells: Record<QuantKey, QuantCellVerdict>;
};

/**
 * Generate a full quant matrix for a list of param sizes.
 * Shows all quantization options (not just the ones available in Ollama).
 */
export function getQuantMatrix(args: {
  paramSizes: number[];
  ramGB: number | null;
  context?: number;
  isMobile?: boolean;
  platform?: PlatformType;
}): QuantMatrixRow[] {
  // Sort sizes ascending
  const sorted = [...args.paramSizes].sort((a, b) => a - b);

  return sorted.map((paramsB) => {
    const cells = {} as Record<QuantKey, QuantCellVerdict>;
    for (const quant of QUANT_KEYS) {
      cells[quant] = getQuantCellVerdict({
        paramsB,
        quant,
        ramGB: args.ramGB,
        context: args.context,
        isMobile: args.isMobile,
        platform: args.platform,
      });
    }
    return {
      paramsB,
      label: paramsB < 1 ? `${(paramsB * 1000).toFixed(0)}M` : `${paramsB}B`,
      cells,
    };
  });
}

/**
 * Get unique param sizes from variants (sorted ascending).
 */
export function getUniqueParamSizes(variants: ModelVariant[]): number[] {
  const sizes = new Set(variants.map((v) => v.paramsB));
  return [...sizes].sort((a, b) => a - b);
}

// ============================================================================
// Quick Reference Tables (for UI display)
// ============================================================================

/**
 * RAM-to-model-size quick reference.
 * Conservative estimates for comfortable running.
 */
export const RAM_RECOMMENDATIONS = [
  { ramGB: 8, maxParamsB: 3, note: '1-3B models only' },
  { ramGB: 16, maxParamsB: 8, note: '7-8B models comfortably' },
  { ramGB: 24, maxParamsB: 13, note: '13B models with headroom' },
  { ramGB: 32, maxParamsB: 14, note: '14B comfortable, 32B tight' },
  { ramGB: 48, maxParamsB: 32, note: '32B comfortable' },
  { ramGB: 64, maxParamsB: 45, note: '34B comfortable, 70B tight' },
  { ramGB: 96, maxParamsB: 70, note: '70B comfortable' },
  { ramGB: 128, maxParamsB: 100, note: 'Most models runnable' },
  { ramGB: 192, maxParamsB: 200, note: 'Large MoE models' },
  { ramGB: 256, maxParamsB: 400, note: 'Even 405B possible' },
] as const;

/**
 * Get recommended max model size for a given RAM amount.
 */
export function getRecommendedMaxParams(ramGB: number): { maxParamsB: number; note: string } {
  // Find the largest recommendation that fits
  for (let i = RAM_RECOMMENDATIONS.length - 1; i >= 0; i--) {
    if (ramGB >= RAM_RECOMMENDATIONS[i].ramGB) {
      return {
        maxParamsB: RAM_RECOMMENDATIONS[i].maxParamsB,
        note: RAM_RECOMMENDATIONS[i].note,
      };
    }
  }
  return { maxParamsB: 1, note: 'Very limited - try tiny models' };
}

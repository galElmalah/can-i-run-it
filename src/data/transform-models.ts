/**
 * Transformation utilities to convert scraped Ollama models to app format.
 * 
 * The scraped data lacks detailed variant info (sizeGB, quant, context),
 * so we estimate these values based on model size patterns.
 */

import type { ScrapedOllamaModel } from './scraped-models';
import type { OllamaModelEntry, ModelVariant } from './models';

// ============================================================================
// Size Parsing
// ============================================================================

/**
 * Parse size string like '8b', '1.5b', '270m' to billion parameters.
 */
export function parseSize(sizeStr: string): number {
  const normalized = sizeStr.toLowerCase().trim();
  
  // Match patterns like '8b', '1.5b', '70b', '405b'
  const bMatch = normalized.match(/^(\d+(?:\.\d+)?)\s*b$/);
  if (bMatch) {
    return parseFloat(bMatch[1]);
  }
  
  // Match patterns like '270m', '135m', '22m'
  const mMatch = normalized.match(/^(\d+(?:\.\d+)?)\s*m$/);
  if (mMatch) {
    return parseFloat(mMatch[1]) / 1000;
  }
  
  // Match MoE patterns like '8x7b', '8x22b'
  const moeMatch = normalized.match(/^(\d+)x(\d+(?:\.\d+)?)\s*b$/);
  if (moeMatch) {
    // For MoE, total params is roughly experts * size (simplified)
    const experts = parseInt(moeMatch[1]);
    const sizePerExpert = parseFloat(moeMatch[2]);
    // Active params during inference is typically ~2 experts
    return sizePerExpert * Math.min(experts, 2) * 1.2;
  }
  
  // Fallback: try to extract any number
  const numMatch = normalized.match(/(\d+(?:\.\d+)?)/);
  if (numMatch) {
    const num = parseFloat(numMatch[1]);
    // If it's a large number (> 100), assume it's millions
    if (num > 100) return num / 1000;
    return num;
  }
  
  return 7; // Default fallback
}

// ============================================================================
// Category Inference
// ============================================================================

type ModelCategory = OllamaModelEntry['category'];

/**
 * Infer the primary category from capabilities array.
 * Priority: embedding > vision > thinking > tools > general
 */
export function inferCategory(capabilities: string[]): ModelCategory {
  const caps = capabilities.map(c => c.toLowerCase());

  // Embedding models are specialized
  if (caps.includes('embedding')) return 'embedding';

  // Vision models are multimodal
  if (caps.includes('vision')) return 'vision';

  // Thinking models (reasoning with chain-of-thought)
  if (caps.includes('thinking')) return 'thinking';

  // Tools/function calling
  if (caps.includes('tools')) return 'tools';

  // Default to general
  return 'general';
}

// ============================================================================
// Publisher Inference
// ============================================================================

const PUBLISHER_PATTERNS: [RegExp, string][] = [
  // Major labs
  [/^llama/i, 'Meta'],
  [/^codellama/i, 'Meta'],
  [/^qwen/i, 'Alibaba'],
  [/^qwq/i, 'Alibaba'],
  [/^gemma/i, 'Google'],
  [/^codegemma/i, 'Google'],
  [/^gemini/i, 'Google'],
  [/^mistral/i, 'Mistral AI'],
  [/^mixtral/i, 'Mistral AI'],
  [/^codestral/i, 'Mistral AI'],
  [/^mathstral/i, 'Mistral AI'],
  [/^phi/i, 'Microsoft'],
  [/^wizardlm/i, 'Microsoft'],
  [/^deepseek/i, 'DeepSeek'],
  [/^command/i, 'Cohere'],
  [/^aya/i, 'Cohere'],
  
  // Other notable publishers
  [/^nomic/i, 'Nomic AI'],
  [/^mxbai/i, 'Mixedbread AI'],
  [/^starcoder/i, 'BigCode'],
  [/^falcon/i, 'TII'],
  [/^yi/i, '01.AI'],
  [/^solar/i, 'Upstage'],
  [/^granite/i, 'IBM'],
  [/^glm/i, 'Zhipu AI'],
  [/^internlm/i, 'Shanghai AI Lab'],
  [/^llava/i, 'LLaVA Team'],
  [/^bakllava/i, 'SkunkworksAI'],
  [/^moondream/i, 'Vikhyat'],
  [/^tinyllama/i, 'TinyLlama'],
  [/^smollm/i, 'Hugging Face'],
  [/^zephyr/i, 'Hugging Face'],
  [/^stablelm/i, 'Stability AI'],
  [/^stable-code/i, 'Stability AI'],
  [/^dolphin/i, 'Cognitive Computations'],
  [/^nous/i, 'Nous Research'],
  [/^hermes/i, 'Nous Research'],
  [/^openhermes/i, 'Teknium'],
  [/^openchat/i, 'OpenChat'],
  [/^vicuna/i, 'LMSYS'],
  [/^orca/i, 'Microsoft'],
  [/^neural-chat/i, 'Intel'],
  [/^sqlcoder/i, 'Defog'],
  [/^meditron/i, 'EPFL'],
  [/^olmo/i, 'Allen AI'],
  [/^exaone/i, 'LG AI Research'],
  [/^cogito/i, 'Deep Cogito'],
  [/^nemotron/i, 'NVIDIA'],
  [/^snowflake/i, 'Snowflake'],
  [/^bge/i, 'BAAI'],
  [/^all-minilm/i, 'Sentence Transformers'],
  [/^paraphrase/i, 'Sentence Transformers'],
];

/**
 * Infer publisher from model slug using pattern matching.
 */
export function inferPublisher(slug: string): string {
  for (const [pattern, publisher] of PUBLISHER_PATTERNS) {
    if (pattern.test(slug)) {
      return publisher;
    }
  }
  return 'Community';
}

// ============================================================================
// Variant Generation
// ============================================================================

/**
 * Generate a ModelVariant from a size string.
 * Estimates sizeGB, quant, and context based on model size.
 */
export function generateVariant(sizeStr: string, isDefault: boolean = false): ModelVariant {
  const paramsB = parseSize(sizeStr);
  
  // Estimate download size (Q4_K_M quantization typical ratio)
  // Roughly 0.5-0.6 bytes per parameter for Q4
  const sizeGB = Math.round(paramsB * 0.55 * 10) / 10;
  
  // Estimate context length based on model size
  // Larger models often have shorter default context to fit in memory
  let context = 4096;
  if (paramsB <= 3) context = 8192;
  if (paramsB <= 1) context = 4096;
  if (paramsB >= 30) context = 8192;
  if (paramsB >= 70) context = 4096;
  
  return {
    tag: sizeStr.toLowerCase(),
    paramsB,
    sizeGB: Math.max(0.1, sizeGB),
    quant: 'Q4_K_M',
    context,
    isDefault,
  };
}

// ============================================================================
// Tag Inference
// ============================================================================

/**
 * Generate additional search tags from model metadata.
 */
export function inferTags(model: ScrapedOllamaModel): string[] {
  const tags: string[] = [];
  
  // Add publisher-related tags
  const publisher = inferPublisher(model.slug).toLowerCase();
  if (publisher !== 'community') {
    tags.push(publisher.split(' ')[0].toLowerCase());
  }
  
  // Add size-related tags
  if (model.sizes.some(s => parseSize(s) <= 3)) {
    tags.push('small', 'lightweight');
  }
  if (model.sizes.some(s => parseSize(s) >= 70)) {
    tags.push('large');
  }
  
  return tags;
}

// ============================================================================
// Main Transformation
// ============================================================================

/**
 * Transform a single scraped model to OllamaModelEntry format.
 */
export function transformScrapedModel(scraped: ScrapedOllamaModel): OllamaModelEntry {
  // Generate variants from sizes, mark first as default
  const variants = scraped.sizes.length > 0
    ? scraped.sizes.map((size, idx) => generateVariant(size, idx === 0))
    : [generateVariant('7b', true)]; // Fallback for embedding models without sizes
  
  return {
    slug: scraped.slug,
    name: scraped.name,
    description: scraped.description,
    category: inferCategory(scraped.capabilities),
    publisher: inferPublisher(scraped.slug),
    variants,
    tags: [...scraped.capabilities, ...inferTags(scraped)],
    pulls: scraped.pulls,
    updated: scraped.updated,
  };
}

/**
 * Transform all scraped models to OllamaModelEntry format.
 */
export function transformScrapedModels(scraped: ScrapedOllamaModel[]): OllamaModelEntry[] {
  return scraped.map(transformScrapedModel);
}

/**
 * Merge curated models with transformed scraped models.
 * Transformed models are the base (category/tags grounded in scraped capabilities).
 * Curated models act as overrides for better variant data and nicer metadata.
 */
export function mergeModels(
  curated: OllamaModelEntry[],
  transformed: OllamaModelEntry[]
): OllamaModelEntry[] {
  const curatedBySlug = new Map(curated.map((m) => [m.slug, m] as const));
  const transformedSlugs = new Set(transformed.map((m) => m.slug));

  // Start from transformed (scraped-backed) models so category/tags stay data-driven
  const merged: OllamaModelEntry[] = transformed.map((t) => {
    const c = curatedBySlug.get(t.slug);
    if (!c) return t;

    return {
      slug: t.slug,

      // Keep curated copy/publisher/variants where available (better UX + variant data)
      name: c.name,
      description: c.description,
      publisher: c.publisher,
      variants: c.variants,

      // Ground these in scraped capabilities
      category: t.category,
      tags: t.tags,

      // Prefer scraped freshness/accuracy
      pulls: t.pulls,
      updated: t.updated,
    };
  });

  // If any curated models aren't in the scraped list, keep them but force safe defaults.
  // (This prevents made-up categories like `code/reasoning/math` from leaking into the UI.)
  for (const c of curated) {
    if (transformedSlugs.has(c.slug)) continue;
    merged.push({
      ...c,
      category: 'general',
      tags: [],
    });
  }
  
  // Sort by popularity (pulls) if available
  merged.sort((a, b) => {
    const pullsA = parsePulls(a.pulls || '0');
    const pullsB = parsePulls(b.pulls || '0');
    return pullsB - pullsA;
  });
  
  return merged;
}

/**
 * Parse pull count string to number for sorting.
 */
function parsePulls(pullStr: string): number {
  const match = pullStr.match(/([\d.]+)\s*(K|M|B)?/i);
  if (!match) return 0;
  
  const num = parseFloat(match[1]);
  const suffix = (match[2] || '').toUpperCase();
  
  switch (suffix) {
    case 'K': return num * 1_000;
    case 'M': return num * 1_000_000;
    case 'B': return num * 1_000_000_000;
    default: return num;
  }
}

/**
 * Curated Ollama model database.
 * This is a static list that can be updated periodically (e.g., daily via Cloudflare Worker).
 * Data sourced from https://ollama.com/library
 */

import { SCRAPED_OLLAMA_MODELS } from './scraped-models';
import { transformScrapedModels, mergeModels } from './transform-models';

export type ModelVariant = {
  tag: string; // e.g., "7b", "7b-q4_0", "70b-instruct"
  paramsB: number; // billion parameters (approximate)
  sizeGB: number; // download size in GB (approximate)
  quant: string; // quantization type
  context: number; // default context length
  isDefault?: boolean;
};

export type OllamaModelEntry = {
  slug: string; // e.g., "llama3.1"
  name: string; // display name
  description: string;
  category: 'general' | 'code' | 'vision' | 'embedding' | 'math' | 'reasoning' | 'tools' | 'thinking';
  publisher: string;
  variants: ModelVariant[];
  tags: string[]; // for search
  pulls?: string; // e.g., "1.2M"
  updated?: string; // e.g., "2024-01"
};

export type Category = OllamaModelEntry['category'];

// Curated models with accurate variant data
const CURATED_MODELS: OllamaModelEntry[] = [
  // === LLAMA FAMILY ===
  {
    slug: 'llama3.2',
    name: 'Llama 3.2',
    description: "Meta's latest lightweight models optimized for edge devices and multilingual tasks.",
    category: 'general',
    publisher: 'Meta',
    pulls: '15M',
    updated: '2024-09',
    tags: ['meta', 'llama', 'multilingual', 'lightweight', 'edge'],
    variants: [
      { tag: '1b', paramsB: 1, sizeGB: 1.3, quant: 'Q4_K_M', context: 131072, isDefault: true },
      { tag: '3b', paramsB: 3, sizeGB: 2.0, quant: 'Q4_K_M', context: 131072 },
    ],
  },
  {
    slug: 'llama3.1',
    name: 'Llama 3.1',
    description: "Meta's flagship open model with excellent general capabilities and tool use.",
    category: 'general',
    publisher: 'Meta',
    pulls: '45M',
    updated: '2024-07',
    tags: ['meta', 'llama', 'flagship', 'tools', 'function-calling'],
    variants: [
      { tag: '8b', paramsB: 8, sizeGB: 4.7, quant: 'Q4_K_M', context: 131072, isDefault: true },
      { tag: '70b', paramsB: 70, sizeGB: 40, quant: 'Q4_K_M', context: 131072 },
      { tag: '405b', paramsB: 405, sizeGB: 231, quant: 'Q4_K_M', context: 131072 },
    ],
  },
  {
    slug: 'llama3',
    name: 'Llama 3',
    description: "Meta's capable open model, great balance of speed and quality.",
    category: 'general',
    publisher: 'Meta',
    pulls: '30M',
    updated: '2024-04',
    tags: ['meta', 'llama', 'balanced'],
    variants: [
      { tag: '8b', paramsB: 8, sizeGB: 4.7, quant: 'Q4_K_M', context: 8192, isDefault: true },
      { tag: '70b', paramsB: 70, sizeGB: 40, quant: 'Q4_K_M', context: 8192 },
    ],
  },
  {
    slug: 'llama2',
    name: 'Llama 2',
    description: "Meta's previous generation, still widely used and well-supported.",
    category: 'general',
    publisher: 'Meta',
    pulls: '20M',
    updated: '2023-07',
    tags: ['meta', 'llama', 'legacy', 'stable'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 3.8, quant: 'Q4_K_M', context: 4096, isDefault: true },
      { tag: '13b', paramsB: 13, sizeGB: 7.4, quant: 'Q4_K_M', context: 4096 },
      { tag: '70b', paramsB: 70, sizeGB: 38, quant: 'Q4_K_M', context: 4096 },
    ],
  },

  // === QWEN FAMILY ===
  {
    slug: 'qwen2.5',
    name: 'Qwen 2.5',
    description: "Alibaba's latest model series with strong reasoning and multilingual support.",
    category: 'general',
    publisher: 'Alibaba',
    pulls: '8M',
    updated: '2024-09',
    tags: ['alibaba', 'qwen', 'multilingual', 'reasoning', 'chinese'],
    variants: [
      { tag: '0.5b', paramsB: 0.5, sizeGB: 0.4, quant: 'Q4_K_M', context: 32768 },
      { tag: '1.5b', paramsB: 1.5, sizeGB: 1.0, quant: 'Q4_K_M', context: 32768 },
      { tag: '3b', paramsB: 3, sizeGB: 1.9, quant: 'Q4_K_M', context: 32768, isDefault: true },
      { tag: '7b', paramsB: 7, sizeGB: 4.4, quant: 'Q4_K_M', context: 32768 },
      { tag: '14b', paramsB: 14, sizeGB: 9.0, quant: 'Q4_K_M', context: 32768 },
      { tag: '32b', paramsB: 32, sizeGB: 19, quant: 'Q4_K_M', context: 32768 },
      { tag: '72b', paramsB: 72, sizeGB: 41, quant: 'Q4_K_M', context: 32768 },
    ],
  },
  {
    slug: 'qwen2.5-coder',
    name: 'Qwen 2.5 Coder',
    description: 'Code-focused Qwen model with strong programming capabilities.',
    category: 'code',
    publisher: 'Alibaba',
    pulls: '3M',
    updated: '2024-09',
    tags: ['alibaba', 'qwen', 'code', 'programming', 'developer'],
    variants: [
      { tag: '1.5b', paramsB: 1.5, sizeGB: 1.0, quant: 'Q4_K_M', context: 32768 },
      { tag: '3b', paramsB: 3, sizeGB: 1.9, quant: 'Q4_K_M', context: 32768 },
      { tag: '7b', paramsB: 7, sizeGB: 4.4, quant: 'Q4_K_M', context: 32768, isDefault: true },
      { tag: '14b', paramsB: 14, sizeGB: 9.0, quant: 'Q4_K_M', context: 32768 },
      { tag: '32b', paramsB: 32, sizeGB: 19, quant: 'Q4_K_M', context: 32768 },
    ],
  },

  // === MISTRAL / MIXTRAL ===
  {
    slug: 'mistral',
    name: 'Mistral',
    description: 'Fast and capable 7B model from Mistral AI, great for general tasks.',
    category: 'general',
    publisher: 'Mistral AI',
    pulls: '12M',
    updated: '2024-02',
    tags: ['mistral', 'fast', 'efficient', '7b'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 32768, isDefault: true },
    ],
  },
  {
    slug: 'mistral-small',
    name: 'Mistral Small',
    description: 'Compact Mistral model optimized for speed and efficiency.',
    category: 'general',
    publisher: 'Mistral AI',
    pulls: '500K',
    updated: '2024-09',
    tags: ['mistral', 'small', 'fast', 'efficient'],
    variants: [
      { tag: '22b', paramsB: 22, sizeGB: 13, quant: 'Q4_K_M', context: 32768, isDefault: true },
    ],
  },
  {
    slug: 'mixtral',
    name: 'Mixtral 8x7B',
    description: 'Mixture-of-experts model with excellent quality and reasonable speed.',
    category: 'general',
    publisher: 'Mistral AI',
    pulls: '4M',
    updated: '2024-01',
    tags: ['mistral', 'moe', 'mixture-of-experts', 'quality'],
    variants: [
      { tag: '8x7b', paramsB: 47, sizeGB: 26, quant: 'Q4_K_M', context: 32768, isDefault: true },
    ],
  },
  {
    slug: 'mixtral-8x22b',
    name: 'Mixtral 8x22B',
    description: 'Large mixture-of-experts model for demanding tasks.',
    category: 'general',
    publisher: 'Mistral AI',
    pulls: '800K',
    updated: '2024-04',
    tags: ['mistral', 'moe', 'large', 'quality'],
    variants: [
      { tag: '8x22b', paramsB: 141, sizeGB: 80, quant: 'Q4_K_M', context: 65536, isDefault: true },
    ],
  },

  // === GEMMA ===
  {
    slug: 'gemma2',
    name: 'Gemma 2',
    description: "Google's open model with strong performance across sizes.",
    category: 'general',
    publisher: 'Google',
    pulls: '6M',
    updated: '2024-06',
    tags: ['google', 'gemma', 'efficient'],
    variants: [
      { tag: '2b', paramsB: 2, sizeGB: 1.6, quant: 'Q4_K_M', context: 8192 },
      { tag: '9b', paramsB: 9, sizeGB: 5.4, quant: 'Q4_K_M', context: 8192, isDefault: true },
      { tag: '27b', paramsB: 27, sizeGB: 16, quant: 'Q4_K_M', context: 8192 },
    ],
  },
  {
    slug: 'gemma',
    name: 'Gemma',
    description: "Google's first-gen open model, lightweight and efficient.",
    category: 'general',
    publisher: 'Google',
    pulls: '3M',
    updated: '2024-02',
    tags: ['google', 'gemma', 'lightweight'],
    variants: [
      { tag: '2b', paramsB: 2, sizeGB: 1.4, quant: 'Q4_K_M', context: 8192, isDefault: true },
      { tag: '7b', paramsB: 7, sizeGB: 4.8, quant: 'Q4_K_M', context: 8192 },
    ],
  },

  // === PHI ===
  {
    slug: 'phi3',
    name: 'Phi-3',
    description: 'Microsoft small language model with impressive capability-to-size ratio.',
    category: 'general',
    publisher: 'Microsoft',
    pulls: '5M',
    updated: '2024-04',
    tags: ['microsoft', 'phi', 'small', 'efficient', 'slm'],
    variants: [
      { tag: 'mini', paramsB: 3.8, sizeGB: 2.2, quant: 'Q4_K_M', context: 128000, isDefault: true },
      { tag: 'medium', paramsB: 14, sizeGB: 7.9, quant: 'Q4_K_M', context: 128000 },
    ],
  },
  {
    slug: 'phi4',
    name: 'Phi-4',
    description: "Microsoft's latest small language model with enhanced reasoning.",
    category: 'reasoning',
    publisher: 'Microsoft',
    pulls: '1M',
    updated: '2024-12',
    tags: ['microsoft', 'phi', 'reasoning', 'slm'],
    variants: [
      { tag: '14b', paramsB: 14, sizeGB: 8.4, quant: 'Q4_K_M', context: 16384, isDefault: true },
    ],
  },

  // === CODE MODELS ===
  {
    slug: 'codellama',
    name: 'Code Llama',
    description: "Meta's code-specialized Llama model for programming tasks.",
    category: 'code',
    publisher: 'Meta',
    pulls: '8M',
    updated: '2023-08',
    tags: ['meta', 'code', 'programming', 'llama'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 3.8, quant: 'Q4_K_M', context: 16384, isDefault: true },
      { tag: '13b', paramsB: 13, sizeGB: 7.4, quant: 'Q4_K_M', context: 16384 },
      { tag: '34b', paramsB: 34, sizeGB: 19, quant: 'Q4_K_M', context: 16384 },
      { tag: '70b', paramsB: 70, sizeGB: 38, quant: 'Q4_K_M', context: 16384 },
    ],
  },
  {
    slug: 'deepseek-coder',
    name: 'DeepSeek Coder',
    description: 'Code-focused model with strong programming capabilities.',
    category: 'code',
    publisher: 'DeepSeek',
    pulls: '2M',
    updated: '2024-01',
    tags: ['deepseek', 'code', 'programming'],
    variants: [
      { tag: '1.3b', paramsB: 1.3, sizeGB: 0.8, quant: 'Q4_K_M', context: 16384 },
      { tag: '6.7b', paramsB: 6.7, sizeGB: 3.8, quant: 'Q4_K_M', context: 16384, isDefault: true },
      { tag: '33b', paramsB: 33, sizeGB: 19, quant: 'Q4_K_M', context: 16384 },
    ],
  },
  {
    slug: 'deepseek-coder-v2',
    name: 'DeepSeek Coder V2',
    description: 'Advanced code model with MoE architecture.',
    category: 'code',
    publisher: 'DeepSeek',
    pulls: '1.5M',
    updated: '2024-06',
    tags: ['deepseek', 'code', 'moe', 'programming'],
    variants: [
      { tag: '16b', paramsB: 16, sizeGB: 8.9, quant: 'Q4_K_M', context: 128000, isDefault: true },
      { tag: '236b', paramsB: 236, sizeGB: 133, quant: 'Q4_K_M', context: 128000 },
    ],
  },
  {
    slug: 'starcoder2',
    name: 'StarCoder 2',
    description: 'BigCode code generation model trained on The Stack v2.',
    category: 'code',
    publisher: 'BigCode',
    pulls: '1M',
    updated: '2024-02',
    tags: ['bigcode', 'code', 'programming', 'starcoder'],
    variants: [
      { tag: '3b', paramsB: 3, sizeGB: 1.7, quant: 'Q4_K_M', context: 16384, isDefault: true },
      { tag: '7b', paramsB: 7, sizeGB: 4.0, quant: 'Q4_K_M', context: 16384 },
      { tag: '15b', paramsB: 15, sizeGB: 9.0, quant: 'Q4_K_M', context: 16384 },
    ],
  },

  // === VISION MODELS ===
  {
    slug: 'llava',
    name: 'LLaVA',
    description: 'Vision-language model that can understand images.',
    category: 'vision',
    publisher: 'LLaVA Team',
    pulls: '4M',
    updated: '2024-01',
    tags: ['vision', 'multimodal', 'images', 'llava'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.5, quant: 'Q4_K_M', context: 4096, isDefault: true },
      { tag: '13b', paramsB: 13, sizeGB: 8.0, quant: 'Q4_K_M', context: 4096 },
      { tag: '34b', paramsB: 34, sizeGB: 20, quant: 'Q4_K_M', context: 4096 },
    ],
  },
  {
    slug: 'llava-llama3',
    name: 'LLaVA Llama 3',
    description: 'LLaVA built on Llama 3 for improved vision understanding.',
    category: 'vision',
    publisher: 'LLaVA Team',
    pulls: '500K',
    updated: '2024-05',
    tags: ['vision', 'multimodal', 'llama3', 'images'],
    variants: [
      { tag: '8b', paramsB: 8, sizeGB: 5.0, quant: 'Q4_K_M', context: 8192, isDefault: true },
    ],
  },
  {
    slug: 'llama3.2-vision',
    name: 'Llama 3.2 Vision',
    description: "Meta's official multimodal Llama with vision capabilities.",
    category: 'vision',
    publisher: 'Meta',
    pulls: '2M',
    updated: '2024-09',
    tags: ['meta', 'llama', 'vision', 'multimodal', 'official'],
    variants: [
      { tag: '11b', paramsB: 11, sizeGB: 6.4, quant: 'Q4_K_M', context: 131072, isDefault: true },
      { tag: '90b', paramsB: 90, sizeGB: 52, quant: 'Q4_K_M', context: 131072 },
    ],
  },

  // === EMBEDDING MODELS ===
  {
    slug: 'nomic-embed-text',
    name: 'Nomic Embed Text',
    description: 'High-quality text embedding model for RAG and search.',
    category: 'embedding',
    publisher: 'Nomic AI',
    pulls: '3M',
    updated: '2024-02',
    tags: ['embedding', 'rag', 'search', 'nomic'],
    variants: [
      { tag: 'latest', paramsB: 0.137, sizeGB: 0.27, quant: 'F16', context: 8192, isDefault: true },
    ],
  },
  {
    slug: 'mxbai-embed-large',
    name: 'mxbai Embed Large',
    description: 'State-of-the-art embedding model with excellent retrieval performance.',
    category: 'embedding',
    publisher: 'Mixedbread AI',
    pulls: '1.5M',
    updated: '2024-03',
    tags: ['embedding', 'rag', 'search', 'retrieval'],
    variants: [
      { tag: 'latest', paramsB: 0.335, sizeGB: 0.67, quant: 'F16', context: 512, isDefault: true },
    ],
  },
  {
    slug: 'all-minilm',
    name: 'All MiniLM',
    description: 'Lightweight embedding model, fast and efficient.',
    category: 'embedding',
    publisher: 'Sentence Transformers',
    pulls: '2M',
    updated: '2023-06',
    tags: ['embedding', 'lightweight', 'fast', 'minilm'],
    variants: [
      { tag: 'l6-v2', paramsB: 0.023, sizeGB: 0.045, quant: 'F32', context: 256, isDefault: true },
    ],
  },

  // === REASONING / SPECIALIZED ===
  {
    slug: 'deepseek-r1',
    name: 'DeepSeek R1',
    description: 'Advanced reasoning model with chain-of-thought capabilities.',
    category: 'reasoning',
    publisher: 'DeepSeek',
    pulls: '2M',
    updated: '2025-01',
    tags: ['deepseek', 'reasoning', 'cot', 'thinking'],
    variants: [
      { tag: '1.5b', paramsB: 1.5, sizeGB: 1.1, quant: 'Q4_K_M', context: 65536 },
      { tag: '7b', paramsB: 7, sizeGB: 4.7, quant: 'Q4_K_M', context: 65536, isDefault: true },
      { tag: '8b', paramsB: 8, sizeGB: 4.9, quant: 'Q4_K_M', context: 65536 },
      { tag: '14b', paramsB: 14, sizeGB: 9.0, quant: 'Q4_K_M', context: 65536 },
      { tag: '32b', paramsB: 32, sizeGB: 19, quant: 'Q4_K_M', context: 65536 },
      { tag: '70b', paramsB: 70, sizeGB: 43, quant: 'Q4_K_M', context: 65536 },
      { tag: '671b', paramsB: 671, sizeGB: 404, quant: 'Q4_K_M', context: 65536 },
    ],
  },
  {
    slug: 'qwq',
    name: 'QwQ',
    description: "Alibaba's reasoning model with strong problem-solving abilities.",
    category: 'reasoning',
    publisher: 'Alibaba',
    pulls: '800K',
    updated: '2024-11',
    tags: ['alibaba', 'reasoning', 'problem-solving', 'qwen'],
    variants: [
      { tag: '32b', paramsB: 32, sizeGB: 19, quant: 'Q4_K_M', context: 32768, isDefault: true },
    ],
  },

  // === TOOLS / FUNCTION CALLING ===
  {
    slug: 'command-r',
    name: 'Command R',
    description: "Cohere's model optimized for RAG and tool use.",
    category: 'tools',
    publisher: 'Cohere',
    pulls: '1M',
    updated: '2024-03',
    tags: ['cohere', 'rag', 'tools', 'function-calling'],
    variants: [
      { tag: '35b', paramsB: 35, sizeGB: 20, quant: 'Q4_K_M', context: 131072, isDefault: true },
    ],
  },
  {
    slug: 'command-r-plus',
    name: 'Command R+',
    description: "Cohere's larger RAG and tool-use optimized model.",
    category: 'tools',
    publisher: 'Cohere',
    pulls: '500K',
    updated: '2024-04',
    tags: ['cohere', 'rag', 'tools', 'function-calling', 'large'],
    variants: [
      { tag: '104b', paramsB: 104, sizeGB: 59, quant: 'Q4_K_M', context: 131072, isDefault: true },
    ],
  },

  // === OTHER POPULAR MODELS ===
  {
    slug: 'neural-chat',
    name: 'Neural Chat',
    description: 'Intel-optimized chat model fine-tuned for conversations.',
    category: 'general',
    publisher: 'Intel',
    pulls: '1.5M',
    updated: '2023-11',
    tags: ['intel', 'chat', 'conversation', 'fine-tuned'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 8192, isDefault: true },
    ],
  },
  {
    slug: 'openchat',
    name: 'OpenChat',
    description: 'High-quality chat model with strong instruction following.',
    category: 'general',
    publisher: 'OpenChat',
    pulls: '1.2M',
    updated: '2023-12',
    tags: ['chat', 'instruction', 'conversation'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 8192, isDefault: true },
    ],
  },
  {
    slug: 'vicuna',
    name: 'Vicuna',
    description: 'Fine-tuned LLaMA model with strong chat capabilities.',
    category: 'general',
    publisher: 'LMSYS',
    pulls: '2M',
    updated: '2023-06',
    tags: ['lmsys', 'chat', 'llama', 'fine-tuned'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 3.8, quant: 'Q4_K_M', context: 4096, isDefault: true },
      { tag: '13b', paramsB: 13, sizeGB: 7.4, quant: 'Q4_K_M', context: 4096 },
      { tag: '33b', paramsB: 33, sizeGB: 19, quant: 'Q4_K_M', context: 4096 },
    ],
  },
  {
    slug: 'orca-mini',
    name: 'Orca Mini',
    description: 'Small but capable model trained with synthetic data.',
    category: 'general',
    publisher: 'Orca',
    pulls: '1M',
    updated: '2023-07',
    tags: ['orca', 'small', 'efficient'],
    variants: [
      { tag: '3b', paramsB: 3, sizeGB: 1.9, quant: 'Q4_K_M', context: 2048, isDefault: true },
      { tag: '7b', paramsB: 7, sizeGB: 3.8, quant: 'Q4_K_M', context: 2048 },
      { tag: '13b', paramsB: 13, sizeGB: 7.4, quant: 'Q4_K_M', context: 2048 },
    ],
  },
  {
    slug: 'yi',
    name: 'Yi',
    description: '01.AI bilingual model with strong Chinese and English capabilities.',
    category: 'general',
    publisher: '01.AI',
    pulls: '1.5M',
    updated: '2023-11',
    tags: ['01ai', 'bilingual', 'chinese', 'english'],
    variants: [
      { tag: '6b', paramsB: 6, sizeGB: 3.5, quant: 'Q4_K_M', context: 4096 },
      { tag: '9b', paramsB: 9, sizeGB: 5.0, quant: 'Q4_K_M', context: 4096, isDefault: true },
      { tag: '34b', paramsB: 34, sizeGB: 19, quant: 'Q4_K_M', context: 4096 },
    ],
  },
  {
    slug: 'solar',
    name: 'Solar',
    description: 'Upstage depth-upscaled model with strong performance.',
    category: 'general',
    publisher: 'Upstage',
    pulls: '800K',
    updated: '2024-01',
    tags: ['upstage', 'depth-upscaling', 'korean'],
    variants: [
      { tag: '10.7b', paramsB: 10.7, sizeGB: 6.1, quant: 'Q4_K_M', context: 4096, isDefault: true },
    ],
  },
  {
    slug: 'dolphin-mixtral',
    name: 'Dolphin Mixtral',
    description: 'Uncensored Mixtral fine-tune with broad capabilities.',
    category: 'general',
    publisher: 'Cognitive Computations',
    pulls: '1.2M',
    updated: '2024-01',
    tags: ['dolphin', 'mixtral', 'uncensored', 'moe'],
    variants: [
      { tag: '8x7b', paramsB: 47, sizeGB: 26, quant: 'Q4_K_M', context: 32768, isDefault: true },
      { tag: '8x22b', paramsB: 141, sizeGB: 80, quant: 'Q4_K_M', context: 65536 },
    ],
  },
  {
    slug: 'wizard-vicuna-uncensored',
    name: 'Wizard Vicuna Uncensored',
    description: 'Uncensored wizard model for unrestricted conversations.',
    category: 'general',
    publisher: 'Eric Hartford',
    pulls: '900K',
    updated: '2023-06',
    tags: ['wizard', 'vicuna', 'uncensored'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 3.8, quant: 'Q4_K_M', context: 2048, isDefault: true },
      { tag: '13b', paramsB: 13, sizeGB: 7.4, quant: 'Q4_K_M', context: 2048 },
      { tag: '30b', paramsB: 30, sizeGB: 17, quant: 'Q4_K_M', context: 2048 },
    ],
  },
  {
    slug: 'wizardlm2',
    name: 'WizardLM 2',
    description: 'Microsoft WizardLM with improved instruction following.',
    category: 'general',
    publisher: 'Microsoft',
    pulls: '700K',
    updated: '2024-04',
    tags: ['microsoft', 'wizard', 'instruction'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 32768, isDefault: true },
      { tag: '8x22b', paramsB: 141, sizeGB: 80, quant: 'Q4_K_M', context: 65536 },
    ],
  },
  {
    slug: 'stable-code',
    name: 'Stable Code',
    description: 'Stability AI code completion model.',
    category: 'code',
    publisher: 'Stability AI',
    pulls: '400K',
    updated: '2024-01',
    tags: ['stability', 'code', 'completion'],
    variants: [
      { tag: '3b', paramsB: 3, sizeGB: 1.6, quant: 'Q4_K_M', context: 16384, isDefault: true },
    ],
  },
  {
    slug: 'codegemma',
    name: 'CodeGemma',
    description: "Google's code-specialized Gemma model.",
    category: 'code',
    publisher: 'Google',
    pulls: '600K',
    updated: '2024-04',
    tags: ['google', 'gemma', 'code', 'programming'],
    variants: [
      { tag: '2b', paramsB: 2, sizeGB: 1.4, quant: 'Q4_K_M', context: 8192 },
      { tag: '7b', paramsB: 7, sizeGB: 5.0, quant: 'Q4_K_M', context: 8192, isDefault: true },
    ],
  },
  {
    slug: 'tinyllama',
    name: 'TinyLlama',
    description: 'Compact 1.1B model, great for constrained environments.',
    category: 'general',
    publisher: 'TinyLlama',
    pulls: '2M',
    updated: '2024-01',
    tags: ['tiny', 'small', 'efficient', 'edge'],
    variants: [
      { tag: '1.1b', paramsB: 1.1, sizeGB: 0.64, quant: 'Q4_K_M', context: 2048, isDefault: true },
    ],
  },
  {
    slug: 'tinydolphin',
    name: 'TinyDolphin',
    description: 'Tiny uncensored model based on TinyLlama.',
    category: 'general',
    publisher: 'Cognitive Computations',
    pulls: '300K',
    updated: '2024-01',
    tags: ['tiny', 'dolphin', 'uncensored', 'small'],
    variants: [
      { tag: '1.1b', paramsB: 1.1, sizeGB: 0.64, quant: 'Q4_K_M', context: 2048, isDefault: true },
    ],
  },
  {
    slug: 'smollm',
    name: 'SmolLM',
    description: 'Hugging Face small language model family.',
    category: 'general',
    publisher: 'Hugging Face',
    pulls: '400K',
    updated: '2024-07',
    tags: ['huggingface', 'small', 'efficient', 'smol'],
    variants: [
      { tag: '135m', paramsB: 0.135, sizeGB: 0.27, quant: 'F16', context: 2048 },
      { tag: '360m', paramsB: 0.36, sizeGB: 0.73, quant: 'F16', context: 2048 },
      { tag: '1.7b', paramsB: 1.7, sizeGB: 1.0, quant: 'Q4_K_M', context: 2048, isDefault: true },
    ],
  },
  {
    slug: 'mathstral',
    name: 'Mathstral',
    description: 'Mistral model specialized for math and reasoning.',
    category: 'math',
    publisher: 'Mistral AI',
    pulls: '200K',
    updated: '2024-07',
    tags: ['mistral', 'math', 'reasoning', 'stem'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 32768, isDefault: true },
    ],
  },
  {
    slug: 'meditron',
    name: 'Meditron',
    description: 'Medical domain LLM for healthcare applications.',
    category: 'general',
    publisher: 'EPFL',
    pulls: '150K',
    updated: '2023-11',
    tags: ['medical', 'healthcare', 'domain-specific'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 3.8, quant: 'Q4_K_M', context: 4096, isDefault: true },
      { tag: '70b', paramsB: 70, sizeGB: 38, quant: 'Q4_K_M', context: 4096 },
    ],
  },
  {
    slug: 'sqlcoder',
    name: 'SQLCoder',
    description: 'Model specialized for SQL query generation.',
    category: 'code',
    publisher: 'Defog',
    pulls: '300K',
    updated: '2023-09',
    tags: ['sql', 'database', 'code', 'query'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 8192, isDefault: true },
      { tag: '15b', paramsB: 15, sizeGB: 8.5, quant: 'Q4_K_M', context: 8192 },
    ],
  },
  {
    slug: 'nous-hermes2',
    name: 'Nous Hermes 2',
    description: 'Nous Research fine-tune with strong general capabilities.',
    category: 'general',
    publisher: 'Nous Research',
    pulls: '600K',
    updated: '2024-01',
    tags: ['nous', 'hermes', 'fine-tuned', 'general'],
    variants: [
      { tag: '10.7b', paramsB: 10.7, sizeGB: 6.1, quant: 'Q4_K_M', context: 4096, isDefault: true },
      { tag: '34b', paramsB: 34, sizeGB: 19, quant: 'Q4_K_M', context: 4096 },
    ],
  },
  {
    slug: 'openhermes',
    name: 'OpenHermes',
    description: 'Open fine-tune of Mistral with diverse training data.',
    category: 'general',
    publisher: 'Teknium',
    pulls: '500K',
    updated: '2023-10',
    tags: ['hermes', 'mistral', 'fine-tuned'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 8192, isDefault: true },
    ],
  },
  {
    slug: 'zephyr',
    name: 'Zephyr',
    description: 'Hugging Face fine-tuned Mistral with RLHF.',
    category: 'general',
    publisher: 'Hugging Face',
    pulls: '800K',
    updated: '2023-10',
    tags: ['huggingface', 'zephyr', 'rlhf', 'chat'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.1, quant: 'Q4_K_M', context: 8192, isDefault: true },
    ],
  },
  {
    slug: 'falcon',
    name: 'Falcon',
    description: 'TII open model trained on RefinedWeb.',
    category: 'general',
    publisher: 'TII',
    pulls: '400K',
    updated: '2023-06',
    tags: ['tii', 'falcon', 'refinedweb'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 3.8, quant: 'Q4_K_M', context: 2048, isDefault: true },
      { tag: '40b', paramsB: 40, sizeGB: 22, quant: 'Q4_K_M', context: 2048 },
      { tag: '180b', paramsB: 180, sizeGB: 102, quant: 'Q4_K_M', context: 2048 },
    ],
  },
  {
    slug: 'granite-code',
    name: 'Granite Code',
    description: 'IBM code model trained on permissively licensed data.',
    category: 'code',
    publisher: 'IBM',
    pulls: '250K',
    updated: '2024-05',
    tags: ['ibm', 'granite', 'code', 'enterprise'],
    variants: [
      { tag: '3b', paramsB: 3, sizeGB: 1.9, quant: 'Q4_K_M', context: 8192 },
      { tag: '8b', paramsB: 8, sizeGB: 4.6, quant: 'Q4_K_M', context: 8192, isDefault: true },
      { tag: '20b', paramsB: 20, sizeGB: 12, quant: 'Q4_K_M', context: 8192 },
      { tag: '34b', paramsB: 34, sizeGB: 19, quant: 'Q4_K_M', context: 8192 },
    ],
  },
  {
    slug: 'moondream',
    name: 'Moondream',
    description: 'Small vision-language model that fits on edge devices.',
    category: 'vision',
    publisher: 'Vikhyat',
    pulls: '300K',
    updated: '2024-03',
    tags: ['vision', 'small', 'edge', 'multimodal'],
    variants: [
      { tag: '1.8b', paramsB: 1.8, sizeGB: 1.7, quant: 'Q4_K_M', context: 2048, isDefault: true },
    ],
  },
  {
    slug: 'bakllava',
    name: 'BakLLaVA',
    description: 'Mistral-based vision-language model.',
    category: 'vision',
    publisher: 'SkunkworksAI',
    pulls: '200K',
    updated: '2023-10',
    tags: ['vision', 'mistral', 'multimodal'],
    variants: [
      { tag: '7b', paramsB: 7, sizeGB: 4.5, quant: 'Q4_K_M', context: 4096, isDefault: true },
    ],
  },
  {
    slug: 'glm4',
    name: 'GLM-4',
    description: 'Zhipu AI bilingual model with strong Chinese support.',
    category: 'general',
    publisher: 'Zhipu AI',
    pulls: '150K',
    updated: '2024-06',
    tags: ['zhipu', 'chinese', 'bilingual'],
    variants: [
      { tag: '9b', paramsB: 9, sizeGB: 5.5, quant: 'Q4_K_M', context: 131072, isDefault: true },
    ],
  },
  {
    slug: 'internlm2',
    name: 'InternLM 2',
    description: 'Shanghai AI Lab advanced language model.',
    category: 'general',
    publisher: 'Shanghai AI Lab',
    pulls: '200K',
    updated: '2024-01',
    tags: ['intern', 'chinese', 'research'],
    variants: [
      { tag: '1.8b', paramsB: 1.8, sizeGB: 1.1, quant: 'Q4_K_M', context: 32768 },
      { tag: '7b', paramsB: 7, sizeGB: 4.5, quant: 'Q4_K_M', context: 32768, isDefault: true },
      { tag: '20b', paramsB: 20, sizeGB: 12, quant: 'Q4_K_M', context: 32768 },
    ],
  },
];

// Transform scraped models and merge with curated ones
// Transformed (scraped) models are the base so categories/tags stay grounded in real capabilities.
// Curated models act as overrides for better variants/publisher where available.
const transformedModels = transformScrapedModels(SCRAPED_OLLAMA_MODELS);
export const OLLAMA_MODELS: OllamaModelEntry[] = mergeModels(CURATED_MODELS, transformedModels);

// Dev sanity check: ensure we don't surface made-up categories
if (import.meta.env.DEV) {
  const allowed = new Set<OllamaModelEntry['category']>(['general', 'embedding', 'vision', 'thinking', 'tools']);
  const unexpected = OLLAMA_MODELS.filter((m) => !allowed.has(m.category));
  if (unexpected.length > 0) {
    // eslint-disable-next-line no-console
    console.warn(
      `[can-i-run-it] Unexpected categories detected (${unexpected.length}). These should come only from scraped capabilities.`,
      unexpected.slice(0, 12).map((m) => ({ slug: m.slug, category: m.category })),
    );
  }
}

// Helper to search models
export function searchModels(query: string): OllamaModelEntry[] {
  const q = query.toLowerCase().trim();
  if (!q) return OLLAMA_MODELS;

  return OLLAMA_MODELS.filter((m) => {
    const haystack = [
      m.slug,
      m.name,
      m.description,
      m.category,
      m.publisher,
      ...m.tags,
      ...m.variants.map((v) => v.tag),
    ]
      .join(' ')
      .toLowerCase();
    return haystack.includes(q);
  });
}

// Get model by slug
export function getModelBySlug(slug: string): OllamaModelEntry | undefined {
  return OLLAMA_MODELS.find((m) => m.slug === slug);
}

// Category labels
export const CATEGORY_LABELS: Record<OllamaModelEntry['category'], string> = {
  general: 'General',
  code: 'Code',
  vision: 'Vision',
  embedding: 'Embedding',
  math: 'Math',
  reasoning: 'Reasoning',
  tools: 'Tools',
  thinking: 'Thinking',
};

// Category colors (for badges)
export const CATEGORY_TONES: Record<OllamaModelEntry['category'], 'pink' | 'green' | 'yellow' | 'blue' | 'purple'> = {
  general: 'green',
  code: 'blue',
  vision: 'purple',
  embedding: 'yellow',
  math: 'pink',
  reasoning: 'pink',
  tools: 'yellow',
  thinking: 'purple',
};

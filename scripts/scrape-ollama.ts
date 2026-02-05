#!/usr/bin/env npx tsx
/**
 * Scrapes the Ollama library page to extract model information.
 * 
 * Usage:
 *   npx tsx scripts/scrape-ollama.ts
 *   npx tsx scripts/scrape-ollama.ts --output data/ollama-models.json
 *   npx tsx scripts/scrape-ollama.ts --update-source  # Creates src/data/scraped-models.ts
 */

import * as fs from 'fs';
import * as path from 'path';

const OLLAMA_LIBRARY_URL = 'https://ollama.com/library';

interface ScrapedModel {
  slug: string;
  name: string;
  description: string;
  capabilities: string[];  // e.g., ['tools', 'vision', 'thinking']
  sizes: string[];         // e.g., ['8b', '70b', '405b']
  pulls: string;           // e.g., '109.7M'
  pullsNumeric: number;    // For sorting
  tagCount: number;        // Number of available tags/variants
  updated: string;         // e.g., 'Nov 3, 2024'
}

interface ScrapeResult {
  timestamp: string;
  source: string;
  modelCount: number;
  models: ScrapedModel[];
}

/**
 * Parse pull count string to a sortable number
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

/**
 * Fetch the Ollama library page HTML
 */
async function fetchOllamaLibrary(): Promise<string> {
  console.log(`Fetching ${OLLAMA_LIBRARY_URL}...`);
  
  const response = await fetch(OLLAMA_LIBRARY_URL, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.5',
    },
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch Ollama library: ${response.status} ${response.statusText}`);
  }
  
  return response.text();
}

/**
 * Parse the HTML to extract model data
 * HTML structure for each model:
 * <a href="/library/{slug}" class="group...">
 *   <h2>...{slug}...</h2>
 *   <p class="...text-neutral-800...">{description}</p>
 *   <span x-test-capability>{capability}</span>
 *   <span x-test-size>{size}</span>
 *   <span x-test-pull-count>{pulls}</span>
 *   <span x-test-tag-count>{tagCount}</span>
 * </a>
 */
function parseModelsFromHtml(html: string): ScrapedModel[] {
  const models: ScrapedModel[] = [];
  
  // Match each model block - from <a href="/library/slug" to the next model or end
  const modelPattern = /<a\s+href="\/library\/([a-z0-9._-]+)"[^>]*class="group[^"]*"[^>]*>([\s\S]*?)(?=<a\s+href="\/library\/[a-z]|<\/ul>|<\/main>)/gi;
  
  let match;
  while ((match = modelPattern.exec(html)) !== null) {
    const slug = match[1];
    const content = match[2];
    
    // Skip if it looks like a nested link or invalid
    if (slug.includes('/') || slug.length < 2) continue;
    
    // Extract description
    const descMatch = content.match(/<p[^>]*class="[^"]*text-neutral-800[^"]*"[^>]*>([^<]+)<\/p>/i);
    const description = descMatch 
      ? descMatch[1].trim().replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&')
      : '';
    
    // Extract capabilities (tools, vision, thinking, embedding, cloud)
    const capabilityMatches = content.matchAll(/x-test-capability[^>]*>([^<]+)</gi);
    const capabilities = [...capabilityMatches].map(m => m[1].trim().toLowerCase());
    
    // Extract sizes (8b, 70b, etc.)
    const sizeMatches = content.matchAll(/x-test-size[^>]*>([^<]+)</gi);
    const sizes = [...sizeMatches].map(m => m[1].trim().toLowerCase());
    
    // Extract pull count
    const pullMatch = content.match(/x-test-pull-count[^>]*>([^<]+)</i);
    const pulls = pullMatch ? pullMatch[1].trim() : '0';
    
    // Extract tag count
    const tagCountMatch = content.match(/x-test-tag-count[^>]*>([^<]+)</i);
    const tagCount = tagCountMatch ? parseInt(tagCountMatch[1].trim()) || 0 : 0;
    
    // Extract updated date (in title attribute)
    const updatedMatch = content.match(/title="([A-Z][a-z]{2}\s+\d{1,2}(?:,\s*\d{4})?)/i);
    const updated = updatedMatch ? updatedMatch[1] : '';
    
    // Generate display name from slug
    const name = slug
      .replace(/([0-9]+)\.([0-9]+)/g, '$1.$2')  // Preserve version numbers
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
    
    models.push({
      slug,
      name,
      description,
      capabilities,
      sizes,
      pulls,
      pullsNumeric: parsePulls(pulls),
      tagCount,
      updated,
    });
  }
  
  // Sort by pull count (most popular first)
  models.sort((a, b) => b.pullsNumeric - a.pullsNumeric);
  
  return models;
}

/**
 * Main scraping function
 */
async function scrapeOllamaLibrary(): Promise<ScrapeResult> {
  const html = await fetchOllamaLibrary();
  const models = parseModelsFromHtml(html);
  
  return {
    timestamp: new Date().toISOString(),
    source: OLLAMA_LIBRARY_URL,
    modelCount: models.length,
    models,
  };
}

/**
 * Sanitize string for safe TypeScript output.
 * Replaces curly quotes with straight quotes.
 */
function sanitizeForTs(str: string): string {
  return str
    // Replace curly quotes with straight quotes
    .replace(/[\u2018\u2019]/g, "'")  // Single curly quotes → straight
    .replace(/[\u201C\u201D]/g, '"'); // Double curly quotes → straight
}

/**
 * Generate TypeScript code for the scraped models
 */
function generateModelsTypeScript(models: ScrapedModel[]): string {
  const header = `/**
 * Auto-generated Ollama model database.
 * Generated: ${new Date().toISOString()}
 * Source: ${OLLAMA_LIBRARY_URL}
 * 
 * This file is auto-generated by scripts/scrape-ollama.ts
 * Do not edit manually - changes will be overwritten.
 */

export interface ScrapedOllamaModel {
  slug: string;
  name: string;
  description: string;
  capabilities: string[];
  sizes: string[];
  pulls: string;
  pullsNumeric: number;
  tagCount: number;
  updated: string;
}

`;

  // Sanitize string values before JSON stringifying
  const sanitizedModels = models.map(m => ({
    ...m,
    name: sanitizeForTs(m.name),
    description: sanitizeForTs(m.description),
    updated: sanitizeForTs(m.updated),
  }));

  // Use JSON.stringify with double quotes, then convert to single quotes with proper escaping
  const modelsJson = JSON.stringify(sanitizedModels, null, 2)
    .replace(/"([^"]+)":/g, '$1:');  // Remove quotes from keys only
  
  // Keep double quotes for string values since they're safer with apostrophes
  return header + `export const SCRAPED_OLLAMA_MODELS: ScrapedOllamaModel[] = ${modelsJson};\n`;
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  const outputArg = args.find(a => a.startsWith('--output='));
  const outputPath = outputArg ? outputArg.split('=')[1] : null;
  const updateSource = args.includes('--update-source');
  const dryRun = args.includes('--dry-run');
  const jsonOnly = args.includes('--json');
  
  if (!jsonOnly) {
    console.log('🦙 Ollama Library Scraper\n');
  }
  
  try {
    const result = await scrapeOllamaLibrary();
    
    if (jsonOnly) {
      // Output only JSON for piping
      console.log(JSON.stringify(result, null, 2));
      return;
    }
    
    console.log(`\n✅ Scraped ${result.modelCount} models`);
    console.log(`   Timestamp: ${result.timestamp}`);
    
    if (result.models.length > 0) {
      console.log('\n📊 Top 15 models by pulls:');
      result.models.slice(0, 15).forEach((m, i) => {
        const caps = m.capabilities.length > 0 ? ` [${m.capabilities.join(', ')}]` : '';
        console.log(`   ${String(i + 1).padStart(2)}. ${m.slug.padEnd(20)} ${m.pulls.padStart(8)} pulls | sizes: ${m.sizes.join(', ') || 'N/A'}${caps}`);
      });
      
      console.log('\n📈 Category breakdown:');
      const byCap: Record<string, number> = {};
      for (const m of result.models) {
        for (const cap of m.capabilities) {
          byCap[cap] = (byCap[cap] || 0) + 1;
        }
      }
      Object.entries(byCap)
        .sort((a, b) => b[1] - a[1])
        .forEach(([cap, count]) => {
          console.log(`   ${cap}: ${count} models`);
        });
    }
    
    if (dryRun) {
      console.log('\n🔍 Dry run - not writing files');
      console.log('\nSample model data (first 3):');
      console.log(JSON.stringify(result.models.slice(0, 3), null, 2));
      return;
    }
    
    if (outputPath) {
      const fullPath = path.resolve(process.cwd(), outputPath);
      const dir = path.dirname(fullPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(fullPath, JSON.stringify(result, null, 2));
      console.log(`\n📁 Wrote JSON to: ${fullPath}`);
    }
    
    if (updateSource) {
      const sourcePath = path.resolve(process.cwd(), 'src/data/scraped-models.ts');
      const sourceDir = path.dirname(sourcePath);
      if (!fs.existsSync(sourceDir)) {
        fs.mkdirSync(sourceDir, { recursive: true });
      }
      fs.writeFileSync(sourcePath, generateModelsTypeScript(result.models));
      console.log(`\n📁 Updated TypeScript source: ${sourcePath}`);
    }
    
    if (!outputPath && !updateSource && !dryRun) {
      // Default: output JSON path suggestion
      console.log('\n💡 Use --output=path/to/file.json to save results');
      console.log('   Use --update-source to generate src/data/scraped-models.ts');
    }
    
  } catch (error) {
    console.error('❌ Scraping failed:', error);
    process.exit(1);
  }
}

main();

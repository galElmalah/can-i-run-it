/**
 * Cloudflare Pages Function stub for refreshing the model list.
 *
 * This is a placeholder for a daily job that could:
 * 1. Fetch https://ollama.com/library
 * 2. Parse the HTML to extract model data
 * 3. Store the result in Cloudflare D1 or KV
 *
 * For now, this just returns the current static model count.
 *
 * To deploy:
 * 1. Add a D1 database or KV namespace in your Cloudflare dashboard
 * 2. Update wrangler.toml with the binding
 * 3. Implement the scraping logic below
 *
 * Trigger options:
 * - Cron trigger (via Cloudflare Workers scheduled events)
 * - Manual trigger (via this endpoint with a secret)
 */

interface Env {
  // Uncomment when you add D1:
  // DB: D1Database;
  // Or KV:
  // MODELS_KV: KVNamespace;
  // Secret for manual trigger:
  // REFRESH_SECRET: string;
}

export const onRequestGet: PagesFunction<Env> = async (context) => {
  // Placeholder response
  return new Response(
    JSON.stringify({
      status: 'stub',
      message: 'Model refresh endpoint is a placeholder. Implement scraping + storage to enable.',
      timestamp: new Date().toISOString(),
    }),
    {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-store',
      },
    }
  );
};

export const onRequestPost: PagesFunction<Env> = async (context) => {
  // Future: implement actual refresh logic here
  // 1. Verify secret/auth
  // 2. Fetch https://ollama.com/library
  // 3. Parse HTML (use cheerio or similar)
  // 4. Store in D1/KV
  // 5. Return success/failure

  return new Response(
    JSON.stringify({
      status: 'not_implemented',
      message: 'POST refresh not yet implemented. See comments in this file for guidance.',
    }),
    {
      status: 501,
      headers: { 'Content-Type': 'application/json' },
    }
  );
};

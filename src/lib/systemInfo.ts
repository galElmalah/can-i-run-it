export type SystemInfo = {
  os: { name: string; version?: string };
  browser: { name: string; version?: string };
  cores: number | null;
  memoryGB: number | null;
  gpu: { renderer?: string; vendor?: string } | null;
  isMobile: boolean;
  isSecureContext: boolean;
  userAgent: string;
};

function parseOS(userAgent: string): SystemInfo['os'] {
  // iOS (must come before macOS)
  const ios = userAgent.match(/(iPhone|iPad|iPod).*OS (\d+)[._](\d+)(?:[._](\d+))?/);
  if (ios) {
    const v = [ios[2], ios[3], ios[4]].filter(Boolean).join('.');
    return { name: 'iOS', version: v };
  }

  const mac = userAgent.match(/Mac OS X (\d+)[._](\d+)(?:[._](\d+))?/);
  if (mac) {
    const v = [mac[1], mac[2], mac[3]].filter(Boolean).join('.');
    return { name: 'macOS', version: v };
  }

  const win = userAgent.match(/Windows NT (\d+\.\d+)/);
  if (win) {
    return { name: 'Windows', version: win[1] };
  }

  const android = userAgent.match(/Android (\d+(?:\.\d+)?)/);
  if (android) {
    return { name: 'Android', version: android[1] };
  }

  if (/Linux/.test(userAgent)) return { name: 'Linux' };
  return { name: 'Unknown' };
}

function parseBrowser(userAgent: string): SystemInfo['browser'] {
  const edge = userAgent.match(/Edg\/(\d+(?:\.\d+)+)/);
  if (edge) return { name: 'Edge', version: edge[1] };

  const opera = userAgent.match(/OPR\/(\d+(?:\.\d+)+)/);
  if (opera) return { name: 'Opera', version: opera[1] };

  const firefox = userAgent.match(/Firefox\/(\d+(?:\.\d+)+)/);
  if (firefox) return { name: 'Firefox', version: firefox[1] };

  // Chrome (must come after Edge/Opera)
  const chrome = userAgent.match(/Chrome\/(\d+(?:\.\d+)+)/);
  if (chrome && !/Chromium/.test(userAgent)) return { name: 'Chrome', version: chrome[1] };

  // Safari (must come after Chrome)
  const safari = userAgent.match(/Version\/(\d+(?:\.\d+)+).*Safari/);
  if (safari && !/Chrome\//.test(userAgent) && !/Edg\//.test(userAgent)) {
    return { name: 'Safari', version: safari[1] };
  }

  return { name: 'Unknown' };
}

function getWebGLDebugInfo(gl: WebGLRenderingContext | WebGL2RenderingContext) {
  try {
    const debug = gl.getExtension('WEBGL_debug_renderer_info') as
      | {
          UNMASKED_VENDOR_WEBGL: number;
          UNMASKED_RENDERER_WEBGL: number;
        }
      | null;
    if (!debug) return null;

    const vendor = gl.getParameter(debug.UNMASKED_VENDOR_WEBGL);
    const renderer = gl.getParameter(debug.UNMASKED_RENDERER_WEBGL);
    return {
      vendor: typeof vendor === 'string' ? vendor : undefined,
      renderer: typeof renderer === 'string' ? renderer : undefined,
    };
  } catch {
    return null;
  }
}

function getGPUInfo(): SystemInfo['gpu'] {
  try {
    const canvas = document.createElement('canvas');
    const gl =
      (canvas.getContext('webgl2') as WebGL2RenderingContext | null) ??
      (canvas.getContext('webgl') as WebGLRenderingContext | null) ??
      (canvas.getContext('experimental-webgl') as WebGLRenderingContext | null);

    if (!gl) return null;

    const debug = getWebGLDebugInfo(gl);
    if (debug?.vendor || debug?.renderer) return debug;

    const vendor = gl.getParameter(gl.VENDOR);
    const renderer = gl.getParameter(gl.RENDERER);
    return {
      vendor: typeof vendor === 'string' ? vendor : undefined,
      renderer: typeof renderer === 'string' ? renderer : undefined,
    };
  } catch {
    return null;
  }
}

export function getSystemInfo(): SystemInfo {
  const userAgent = navigator.userAgent ?? '';
  const isMobile =
    // UA-CH (Chromium)
    (navigator as unknown as { userAgentData?: { mobile?: boolean } }).userAgentData?.mobile ??
    // fallback
    /Mobi|Android|iPhone|iPad|iPod/i.test(userAgent);

  const cores = typeof navigator.hardwareConcurrency === 'number' ? navigator.hardwareConcurrency : null;
  const memoryGB =
    typeof (navigator as unknown as { deviceMemory?: number }).deviceMemory === 'number'
      ? (navigator as unknown as { deviceMemory: number }).deviceMemory
      : null;

  return {
    os: parseOS(userAgent),
    browser: parseBrowser(userAgent),
    cores,
    memoryGB,
    gpu: typeof document === 'undefined' ? null : getGPUInfo(),
    isMobile,
    isSecureContext: typeof window !== 'undefined' ? window.isSecureContext : false,
    userAgent,
  };
}


import { useCallback, useEffect, useState } from 'react';

type Tone = 'pink' | 'green' | 'yellow' | 'ink';

type Props = {
  text: string;
  label?: string;
  tone?: Tone;
  className?: string;
};

async function fallbackCopy(text: string) {
  const el = document.createElement('textarea');
  el.value = text;
  el.setAttribute('readonly', 'true');
  el.style.position = 'fixed';
  el.style.top = '-9999px';
  el.style.left = '-9999px';
  document.body.appendChild(el);
  el.select();
  try {
    document.execCommand('copy');
  } finally {
    document.body.removeChild(el);
  }
}

export function CopyButton({ text, label = 'Copy', tone = 'yellow', className }: Props) {
  const [state, setState] = useState<'idle' | 'copied' | 'error'>('idle');

  useEffect(() => {
    if (state !== 'copied') return;
    const t = window.setTimeout(() => setState('idle'), 900);
    return () => window.clearTimeout(t);
  }, [state]);

  const onCopy = useCallback(async () => {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(text);
      else await fallbackCopy(text);
      setState('copied');
    } catch {
      setState('error');
      window.setTimeout(() => setState('idle'), 1200);
    }
  }, [text]);

  const caption = state === 'copied' ? 'Copied!' : state === 'error' ? 'Nope' : label;

  return (
    <button type="button" className={['btn', className].filter(Boolean).join(' ')} data-tone={tone} onClick={onCopy}>
      {caption}
    </button>
  );
}


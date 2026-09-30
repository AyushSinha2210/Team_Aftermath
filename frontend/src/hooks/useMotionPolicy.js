import { useEffect, useState } from 'react';

export function useMotionPolicy() {
  const [reduced, setReduced] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(media.matches);
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);
  return reduced;
}

export function shouldSkipIntro() {
  return import.meta.env.VITE_DISABLE_INTRO === 'true'
    || new URLSearchParams(window.location.search).get('intro') === 'off'
    || (navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 2)
    || (navigator.deviceMemory && navigator.deviceMemory <= 2);
}

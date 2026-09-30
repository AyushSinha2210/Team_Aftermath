import { useEffect, useRef, useState } from 'react';
import { useInView } from 'framer-motion';
import { useMotionPolicy } from '../hooks/useMotionPolicy';
import RevealOnScroll from './RevealOnScroll';
import evidence from '../data/evidence.json';
import './evidence.css';

export function CountUp({ value }) {
  const ref = useRef(null);
  const visible = useInView(ref, { once: true });
  const reduced = useMotionPolicy();
  const [current, setCurrent] = useState(0);
  useEffect(() => {
    if (!visible || reduced) return;
    let frame;
    let start;
    const step = now => {
      start ??= now;
      const progress = Math.min((now - start) / 1200, 1);
      setCurrent(value * (1 - (1 - progress) ** 3));
      if (progress < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [visible, reduced, value]);
  return <span ref={ref} className="benchmark-value mono" aria-label={value.toFixed(3)}><span aria-hidden="true">{(reduced ? value : current).toFixed(3)}</span></span>;
}

export default function Benchmarks() {
  return <section className="section" id="benchmarks">
    <RevealOnScroll><p className="section-number">02 / The evidence</p><h2>Measured, not claimed</h2></RevealOnScroll>
    <div className="benchmark-pair"><RevealOnScroll delay={.08}><CountUp value={evidence.official.ndcg} /><p className="metric-label mono">NDCG@10</p></RevealOnScroll><RevealOnScroll delay={.16}><CountUp value={evidence.official.mrr} /><p className="metric-label mono">MRR@10</p></RevealOnScroll></div>
    <p className="section-note">Evaluated on the CoIR <code>apps</code> benchmark via MTEB.</p>
    <p className="evidence-scope mono">Official test split / MTEB {evidence.official.mtebVersion} / checked-in result</p>
    <details className="evidence-details"><summary>Inspect the evaluation source</summary><p className="mono">{evidence.official.source}</p><p className="mono">Dataset revision: {evidence.official.datasetRevision}</p><p>These are existing evaluation results. This frontend work did not rerun the benchmark.</p></details>
  </section>;
}

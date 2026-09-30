import { motion } from 'framer-motion';
import { useMotionPolicy } from '../hooks/useMotionPolicy';
import './hero.css';

export default function Hero({ embedded = false }) {
  const reduced = useMotionPolicy();
  const Heading = embedded ? 'div' : 'h1';
  return <section id={embedded ? undefined : 'hero'} className={`hero ${embedded ? 'hero-embedded' : ''}`} aria-label={embedded ? undefined : 'Ariadne code intelligence'}>
    <motion.div className="hero-glow" aria-hidden="true" animate={reduced ? { opacity: .2 } : { opacity: [.15, .3, .15], scale: [1, 1.05, 1] }} transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }} />
    <div className="hero-content">
      <p className="hero-kicker mono"><span className="status-dot" /> Code intelligence, made precise</p>
      <Heading className="hero-title">Find the exact line.<br /><span>Every time.</span></Heading>
      <p className="hero-subhead">Ariadne ranks thousands of code snippets against a single question — combining dense retrieval, keyword search, and reranking into one answer, in milliseconds, on CPU.</p>
      <a id={embedded ? undefined : 'hero-cta'} className="button button-primary" href="#demo" tabIndex={embedded ? -1 : undefined}>Try a query</a>
      {!embedded && <p className="hero-qualification mono">Explore the pipeline. Inspect the evidence. Query the source.</p>}
    </div>
    {!embedded && <div className="hero-baseline mono"><span>01 / Natural language in</span><span>Source locations out / 02</span></div>}
  </section>;
}

import { motion } from 'framer-motion';
import { useMotionPolicy } from '../hooks/useMotionPolicy';

export default function MiniPipeline({ loading, mode = 'dense', completed }) {
  const reduced = useMotionPolicy();
  const stages = mode === 'hybrid' ? ['Query', 'Dense + sparse', 'RRF', 'Results'] : ['Query', 'Dense retrieval', 'Ranked results'];
  return <div className="mini-pipeline" aria-hidden="true">{stages.map((stage, i) => <div className="mini-stage" key={stage}><span>{stage}</span>{i < stages.length - 1 && <div className="mini-track"><motion.div className="mini-trace" initial={false} animate={loading && !reduced ? { scaleX: [0, 1, 0], opacity: [0, 1, 0] } : { scaleX: completed || reduced ? 1 : 0, opacity: completed || reduced ? .6 : 0 }} transition={loading && !reduced ? { duration: 1.2, delay: i * .2, repeat: Infinity, ease: 'easeInOut' } : { duration: .2 }} /></div>}</div>)}</div>;
}

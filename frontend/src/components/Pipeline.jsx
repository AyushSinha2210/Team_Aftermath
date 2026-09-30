import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useMotionPolicy } from '../hooks/useMotionPolicy';
import './pipeline.css';

const nodes = [
  { x: 12, y: 98, w: 100, label: ['Query'] },
  { x: 158, y: 32, w: 168, label: ['Dense Retrieval'] },
  { x: 158, y: 164, w: 168, label: ['Sparse Retrieval'] },
  { x: 374, y: 98, w: 178, label: ['Reciprocal Rank', 'Fusion'] },
  { x: 600, y: 98, w: 170, label: ['Cross-Encoder', 'Rerank'] },
  { x: 818, y: 98, w: 150, label: ['Ranked Results'] },
];
const paths = ['M112 126H135V60H158', 'M112 126H135V192H158', 'M326 60H350V126H374', 'M326 192H350V126H374', 'M552 126H600', 'M770 126H818'];

export function PipelineDrawing({ trace = false }) {
  return <svg className={`pipeline-svg ${trace ? 'pipeline-trace' : ''}`} viewBox="0 0 980 250" role="img" aria-label="Query splits into dense and sparse retrieval, then reciprocal rank fusion, cross-encoder reranking, and ranked results">
    <g fill="none" stroke="currentColor" strokeWidth="1.5">{paths.map(path => <path key={path} d={path} />)}</g>
    {!trace && nodes.map(node => <g key={node.x}><rect x={node.x} y={node.y} width={node.w} height="56" rx="8" /><text x={node.x + node.w / 2} y={node.y + (node.label.length > 1 ? 23 : 32)} textAnchor="middle">{node.label.map((line, i) => <tspan key={line} x={node.x + node.w / 2} dy={i ? 18 : 0}>{line}</tspan>)}</text></g>)}
  </svg>;
}

export default function Pipeline() {
  const ref = useRef(null);
  const reduced = useMotionPolicy();
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start 90%', 'end 75%'] });
  const scaleX = useTransform(scrollYProgress, [0, 1], [0, 1]);
  return <section ref={ref} className="section" id="pipeline">
    <p className="section-number">01 / The architecture</p>
    <h2>How a query becomes an answer</h2>
    <div className="pipeline-viewport"><div className="pipeline-diagram"><PipelineDrawing /><motion.div className="trace-reveal" style={{ scaleX: reduced ? 1 : scaleX }} aria-hidden="true"><PipelineDrawing trace /></motion.div></div></div>
    <p className="section-note mono">CPU-only inference. No GPU required.</p>
    <p className="section-note pipeline-context">Available retrieval stages are shown above. The demo uses the recommended dense configuration; hybrid fusion is an optional API mode. Reranking is not enabled in this demo.</p>
  </section>;
}

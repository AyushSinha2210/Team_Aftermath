import { motion } from 'framer-motion';
import { useMotionPolicy } from '../hooks/useMotionPolicy';

function Code({ code }) {
  // React text nodes escape source code; never render untrusted code as HTML.
  return <pre className="result-code"><code>{code.split('\n').map((line, i) => <span className={/^\s*\/\//.test(line) ? 'code-comment' : ''} key={i}>{line}{'\n'}</span>)}</code></pre>;
}

export default function Results({ response }) {
  const reduced = useMotionPolicy();
  return <div className="results-list" data-testid="results-container">
    {response?.results.map((result, i) => <motion.article className="result" key={result.id} initial={reduced ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .3, delay: reduced ? 0 : i * .08 }}>
      <header className="result-header"><h3>{result.title}</h3><span className="result-rank mono">{String(i + 1).padStart(2, '0')}</span></header>
      <Code code={result.code} />
      <p className="result-location mono">{result.file}:{result.start_line}–{result.end_line}</p>
      <div className="result-tags mono">{result.sources.map(source => <span key={source} className={`tag tag-${source}`}>{source}{source !== 'reranked' && result[`${source}_score`] !== null ? ` · ${result[`${source}_score`].toFixed(3)}` : ''}</span>)}{response.mode === 'hybrid' && <span className="tag">RRF · {result.score.toFixed(4)}</span>}</div>
    </motion.article>)}
  </div>;
}

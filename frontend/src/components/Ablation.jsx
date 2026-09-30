import evidence from '../data/evidence.json';
import RevealOnScroll from './RevealOnScroll';

const labels = ['Fine-tuned dense only', '+BM25+RRF', '+rerank'];
export default function Ablation() {
  return <section className="section" id="ablation"><RevealOnScroll calm>
    <p className="section-number">04 / The tradeoffs</p><h2>What the fine-tune actually bought us</h2>
    <p className="section-note">Retrieval-stage comparison on {evidence.validation.queries} validation queries against {evidence.validation.candidates} candidates. Dense retrieval is the recommended live mode.</p>
    <div className="table-scroll"><table className="comparison"><caption className="sr-only">Validation retrieval-stage comparison; latency was not recorded in this result file.</caption><thead><tr><th scope="col">Configuration</th><th scope="col">NDCG@10</th><th scope="col">MRR@10</th><th scope="col">Latency (ms)</th></tr></thead><tbody>
      <tr><th scope="row">pretrained only</th><td>Not in this run</td><td>Not in this run</td><td>Not recorded</td></tr>
      {evidence.validation.rows.map((row, i) => <tr key={row.key} className={i === 0 ? 'recommended' : ''}><th scope="row">{labels[i]}</th><td>{row.ndcg.toFixed(3)}</td><td>{row.mrr.toFixed(3)}</td><td>Not recorded</td></tr>)}
    </tbody></table></div>
    <p className="section-note">Adding fusion or this cross-encoder reduced ranking quality in this run. These rows compare retrieval stages; they do not isolate the effect of fine-tuning.</p>
    <details className="evidence-details"><summary>View comparison provenance</summary><p className="mono">{evidence.validation.source}</p><p>Split: {evidence.validation.split}. This candidate-pool result is separate from the official MTEB test above.</p></details>
  </RevealOnScroll></section>;
}

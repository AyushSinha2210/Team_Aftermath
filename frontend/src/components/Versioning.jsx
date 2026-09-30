import RevealOnScroll from './RevealOnScroll';
import './versioning.css';

const before = ['async function verifySession(sessionId) {', '  if (!sessionId) return false;', "  return String(sessionId).startsWith('sess_');", '}'];
const after = ['async function verifySession(sessionId) {', "  if (typeof sessionId !== 'string') return false;", "  return sessionId.startsWith('sess_');", '}'];
export default function Versioning() {
  return <section className="section" id="versioning"><RevealOnScroll calm>
    <p className="section-number">05 / Keep the index current</p><h2>Retrieval across versions</h2>
    <p className="section-note">An illustrative edit to <code>tools/authTool.js</code>. Changed lines are highlighted in teal.</p>
    <div className="version-pair">{[before, after].map((lines, i) => <div className="version-pane" key={i}><p className="version-header mono">{i ? 'Proposed edit' : 'Current function'}<span>{i ? '2 lines changed' : 'Source lines 5–8'}</span></p><pre aria-label={i ? 'Proposed edited function' : 'Current source function'}><code>{lines.map((line, j) => <span key={j} className={i && (j === 1 || j === 2) ? 'changed-line' : ''}><span className="line-number" aria-hidden="true">{j + 5}</span>{line}{'\n'}</span>)}</code></pre></div>)}</div>
    <p className="section-note">Only changed functions are re-embedded — not the whole index — so retrieval stays current as the codebase evolves.</p>
  </RevealOnScroll></section>;
}

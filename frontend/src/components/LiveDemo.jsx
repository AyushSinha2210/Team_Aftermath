import { Button } from './Controls';
import { useSearch } from '../hooks/useSearch';
import QueryForm from './QueryForm';
import Results from './Results';
import MiniPipeline from './MiniPipeline';
import './demo.css';

export default function LiveDemo() {
  const search = useSearch();
  const loading = search.phase === 'loading';
  return <section className="section" id="demo">
    <p className="section-number">03 / Your question, the source</p><h2>Try it yourself</h2>
    <QueryForm onSubmit={search.submit} loading={loading} error={search.error} invalid={search.phase === 'invalid'} />
    <div className="search-status mono" role="status" aria-live="polite">{loading ? 'Processing…' : search.recorded ? 'Recorded example / live backend unreachable' : search.response ? `${search.response.mode} / ${search.response.results.length} matches / ${search.response.elapsed_ms.toFixed(1)} ms server inference` : 'Ready for a question'}</div>
    {search.recorded && <p className="recorded-notice">Showing a previously captured response for this exact example. This is not a live search. <Button onClick={search.retry} disabled={loading}>Retry live search</Button></p>}
    {search.phase === 'error' && <Button onClick={search.retry}>Try again</Button>}
    <MiniPipeline loading={loading} mode={search.response?.mode || 'dense'} completed={Boolean(search.response)} />
    {search.phase === 'empty' && <p className="empty-state">No confident match — try rephrasing.</p>}
    <div aria-busy={loading}><Results response={search.response} /></div>
  </section>;
}

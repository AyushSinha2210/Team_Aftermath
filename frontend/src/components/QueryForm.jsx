import { useRef, useState } from 'react';
import { Button, QueryInput } from './Controls';
import { exampleQueries } from '../lib/fallback';

export default function QueryForm({ loading, error, invalid, onSubmit }) {
  const [query, setQuery] = useState('');
  const input = useRef(null);
  function submit(event) { event.preventDefault(); onSubmit(query); }
  function choose(example) { setQuery(example); input.current?.focus(); onSubmit(example); }
  return <form className="query-form" onSubmit={submit} noValidate>
    <label htmlFor="query" className="sr-only">Ask about this codebase</label>
    <div className="query-row"><QueryInput ref={input} id="query" data-testid="query-input" placeholder="Ask about this codebase…" value={query} onChange={event => setQuery(event.target.value)} maxLength={2000} error={invalid} aria-describedby="query-help query-error" disabled={loading} autoComplete="off" /><Button className="button-primary" data-testid="submit-button" type="submit" disabled={loading}>{loading ? 'Searching' : 'Search code'}</Button></div>
    <p id="query-help" className="query-help">Search the JavaScript voice-assistant sample. Scores express ranking similarity, not certainty.</p>
    <p id="query-error" className="inline-error" role={error ? 'alert' : undefined}>{error}</p>
    <div className="example-queries" aria-label="Example queries">{exampleQueries.map((example, i) => <button className="pill" type="button" key={example} data-testid={`example-query-${i}`} disabled={loading} onClick={() => choose(example)}>{example}</button>)}</div>
  </form>;
}

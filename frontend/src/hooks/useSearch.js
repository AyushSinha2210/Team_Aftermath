import { useEffect, useRef, useState } from 'react';
import { OfflineError, searchApi } from '../lib/search';
import { recordedFallback } from '../lib/fallback';

const initial = { phase: 'idle', response: null, error: '', recorded: false };
export function useSearch() {
  const [state, setState] = useState(initial);
  const active = useRef(null);
  const lastQuery = useRef('');
  useEffect(() => () => active.current?.abort(), []);
  async function submit(rawQuery) {
    if (active.current) return;
    const query = rawQuery.trim();
    if (!query || query.length > 2000) {
      setState({ ...initial, phase: 'invalid', error: query ? 'Keep your query under 2001 characters.' : 'Enter a question to search the codebase.' });
      return;
    }
    const controller = new AbortController();
    active.current = controller;
    lastQuery.current = query;
    setState({ ...initial, phase: 'loading' });
    try {
      const response = await searchApi(query, { signal: controller.signal });
      if (!controller.signal.aborted) setState({ ...initial, phase: response.results.length ? 'success' : 'empty', response });
    } catch (error) {
      if (controller.signal.aborted) return;
      const fallback = error instanceof OfflineError ? recordedFallback(query) : null;
      if (fallback) setState({ ...initial, phase: fallback.results.length ? 'success' : 'empty', response: fallback, recorded: true });
      else setState({ ...initial, phase: 'error', error: error.message || 'Search is unavailable. Please try again.' });
    } finally {
      if (active.current === controller) active.current = null;
    }
  }
  return { ...state, submit, retry: () => submit(lastQuery.current) };
}

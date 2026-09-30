import recording from '../data/recorded-examples.json';
import { normalizeResponse } from './response';

// Safety net only: actual captured responses for the exact example queries.
// Never synthesize matches or use another query's recording for arbitrary input.
export function recordedFallback(query) {
  const record = recording.records.find(item => item.query.trim().toLowerCase() === query.trim().toLowerCase());
  if (!record) return null;
  return { ...normalizeResponse(record.response), recordedAt: recording.recorded_at, sourceCommit: recording.source_commit };
}
export const exampleQueries = recording.records.map(record => record.query);

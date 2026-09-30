export class ProtocolError extends Error {}

const finite = value => typeof value === 'number' && Number.isFinite(value);
export function normalizeResponse(payload) {
  if (!payload || payload.api_version !== 1 || !['success', 'empty'].includes(payload.status)
    || !['dense', 'hybrid'].includes(payload.mode) || !Array.isArray(payload.results)
    || typeof payload.query !== 'string' || !finite(payload.elapsed_ms) || payload.elapsed_ms < 0) {
    throw new ProtocolError('Search returned an unexpected response. Please try again.');
  }
  if ((payload.status === 'empty') !== (payload.results.length === 0) || payload.results.length > 10) {
    throw new ProtocolError('Search returned inconsistent results. Please try again.');
  }
  const ids = new Set();
  const results = payload.results.map(result => {
    if (!result || !['id', 'file', 'title', 'code'].every(key => typeof result[key] === 'string' && result[key].length > 0)
      || !Number.isInteger(result.start_line) || result.start_line < 1 || !Number.isInteger(result.end_line) || result.end_line < result.start_line
      || !finite(result.score) || !Array.isArray(result.sources) || !result.sources.length
      || !result.sources.every(source => ['dense', 'sparse', 'reranked'].includes(source))
      || ![result.dense_score, result.sparse_score].every(value => value === null || finite(value)) || ids.has(result.id)) {
      throw new ProtocolError('Search returned invalid snippet metadata. Please try again.');
    }
    ids.add(result.id);
    return { ...result, sources: [...new Set(result.sources)] };
  });
  return { ...payload, results };
}

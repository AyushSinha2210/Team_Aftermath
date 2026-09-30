import { normalizeResponse, ProtocolError } from './response';

export class TimeoutError extends Error {}
export class OfflineError extends Error {}
export class ServiceError extends Error {}

export async function searchApi(query, { signal, timeoutMs = 8000, fetchImpl = fetch } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort();
  if (signal?.aborted) abort();
  signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetchImpl('/api/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, top_k: 5 }), signal: controller.signal,
    });
    if (!response.ok) {
      // Vite / reverse proxies return a non-JSON gateway error if Python is down.
      if ([500, 502, 504].includes(response.status) && !response.headers.get('content-type')?.includes('application/json')) {
        throw new OfflineError('The live backend is unreachable.');
      }
      throw new ServiceError(response.status === 503 ? 'Search is busy. Please try again.' : 'Search is temporarily unavailable. Please try again.');
    }
    let payload;
    try { payload = await response.json(); }
    catch (error) {
      if (controller.signal.aborted) throw error;
      throw new ProtocolError('Search returned an unexpected response. Please try again.');
    }
    return normalizeResponse(payload);
  } catch (error) {
    if (timedOut) throw new TimeoutError('Search took longer than 8 seconds. Please try again.');
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    if (error instanceof TypeError) throw new OfflineError('The live backend is unreachable.');
    throw error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

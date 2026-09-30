import { afterEach, describe, expect, it, vi } from 'vitest';
import recording from '../data/recorded-examples.json';
import { normalizeResponse, ProtocolError } from './response';
import { OfflineError, searchApi, ServiceError, TimeoutError } from './search';
import { recordedFallback } from './fallback';

const valid = recording.records[0].response;
afterEach(() => vi.useRealTimers());
describe('API envelope', () => {
  it('accepts actual captured responses', () => { for (const record of recording.records) expect(normalizeResponse(record.response)).toEqual(record.response); });
  it.each([{}, null, { ...valid, api_version: 2 }, { ...valid, elapsed_ms: NaN }, { ...valid, results: [{}] }, { ...valid, status: 'empty' }])('rejects malformed payloads', payload => expect(() => normalizeResponse(payload)).toThrow(ProtocolError));
  it('rejects invented line bounds and duplicate identifiers', () => {
    const snippet = valid.results[0];
    expect(() => normalizeResponse({ ...valid, results: [{ ...snippet, start_line: 0 }] })).toThrow(ProtocolError);
    expect(() => normalizeResponse({ ...valid, results: [snippet, snippet] })).toThrow(ProtocolError);
  });
});
describe('network boundaries', () => {
  it('posts the known schema', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(new Response(JSON.stringify(valid), { status: 200 }));
    expect(await searchApi('auth', { fetchImpl })).toEqual(valid);
    expect(JSON.parse(fetchImpl.mock.calls[0][1].body)).toEqual({ query: 'auth', top_k: 5 });
  });
  it('distinguishes network failure', async () => { await expect(searchApi('auth', { fetchImpl: () => Promise.reject(new TypeError('fetch failed')) })).rejects.toBeInstanceOf(OfflineError); });
  it('distinguishes structured server failures', async () => { await expect(searchApi('auth', { fetchImpl: async () => new Response('{}', { status: 500, headers: { 'Content-Type': 'application/json' } }) })).rejects.toBeInstanceOf(ServiceError); });
  it('recognizes a failed backend proxy', async () => { await expect(searchApi('auth', { fetchImpl: async () => new Response('', { status: 500 }) })).rejects.toBeInstanceOf(OfflineError); });
  it('rejects non-JSON success bodies', async () => { await expect(searchApi('auth', { fetchImpl: async () => new Response('<html>error</html>', { status: 200 }) })).rejects.toBeInstanceOf(ProtocolError); });
  it('bounds a hanging request', async () => {
    vi.useFakeTimers();
    const fetchImpl = (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Abort', 'AbortError'))));
    const pending = searchApi('auth', { fetchImpl, timeoutMs: 50 });
    const assertion = expect(pending).rejects.toBeInstanceOf(TimeoutError);
    await vi.advanceTimersByTimeAsync(51);
    await assertion;
  });
  it('supports caller cancellation', async () => {
    const controller = new AbortController();
    const fetchImpl = (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Abort', 'AbortError'))));
    const pending = searchApi('auth', { fetchImpl, signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
  });
});
describe('offline safety net', () => {
  it('returns only an exact recorded query', () => {
    expect(recordedFallback(recording.records[0].query.toUpperCase()).sourceCommit).toBeTruthy();
    expect(recordedFallback('invent an arbitrary retry handler')).toBeNull();
  });
});

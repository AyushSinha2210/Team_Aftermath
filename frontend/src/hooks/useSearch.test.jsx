import { act, renderHook } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { useSearch } from './useSearch';
import * as api from '../lib/search';
import recording from '../data/recorded-examples.json';

afterEach(() => vi.restoreAllMocks());
it('rejects whitespace without a request', async () => {
  const request = vi.spyOn(api, 'searchApi');
  const { result } = renderHook(() => useSearch());
  await act(() => result.current.submit('   '));
  expect(result.current.phase).toBe('invalid');
  expect(request).not.toHaveBeenCalled();
});
it('guards rapid repeated submissions', async () => {
  let finish;
  const request = vi.spyOn(api, 'searchApi').mockImplementation(() => new Promise(resolve => { finish = resolve; }));
  const { result } = renderHook(() => useSearch());
  act(() => { result.current.submit('auth'); result.current.submit('auth'); });
  expect(request).toHaveBeenCalledTimes(1);
  await act(async () => finish(recording.records[0].response));
  expect(result.current.phase).toBe('success');
});
it('labels exact offline recordings', async () => {
  vi.spyOn(api, 'searchApi').mockRejectedValue(new api.OfflineError('offline'));
  const { result } = renderHook(() => useSearch());
  await act(() => result.current.submit(recording.records[0].query));
  expect(result.current.recorded).toBe(true);
});
it('does not invent offline results', async () => {
  vi.spyOn(api, 'searchApi').mockRejectedValue(new api.OfflineError('offline'));
  const { result } = renderHook(() => useSearch());
  await act(() => result.current.submit('unrecorded question'));
  expect(result.current.phase).toBe('error');
  expect(result.current.response).toBeNull();
});
it('cancels in-flight work on unmount', () => {
  let capturedSignal;
  vi.spyOn(api, 'searchApi').mockImplementation((_query, { signal }) => { capturedSignal = signal; return new Promise(() => {}); });
  const { result, unmount } = renderHook(() => useSearch());
  act(() => { result.current.submit('auth'); });
  unmount();
  expect(capturedSignal.aborted).toBe(true);
});

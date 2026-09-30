import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';
import { readFileSync, mkdirSync } from 'node:fs';

const recording = JSON.parse(readFileSync(new URL('../src/data/recorded-examples.json', import.meta.url), 'utf8'));
const response = recording.records[0].response;
test.beforeEach(async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/?intro=off');
});

test('renders all sections, three bundled font families, and no horizontal page overflow', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'Find the exact line. Every time.' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Measured, not claimed' })).toBeVisible();
  await page.evaluate(() => document.fonts.ready);
  expect(await page.evaluate(() => ['Fraunces Variable', 'Manrope Variable', 'JetBrains Mono'].every(font => document.fonts.check(`16px "${font}"`)))).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
test('empty query shows inline validation', async ({ page }) => {
  await page.getByTestId('submit-button').click();
  await expect(page.getByRole('alert')).toContainText('Enter a question');
  await expect(page.getByTestId('query-input')).toHaveAttribute('aria-invalid', 'true');
});
test('Enter submits and real-schema snippets render safely', async ({ page }) => {
  await page.route('**/api/search', route => route.fulfill({ json: response }));
  await page.getByTestId('query-input').fill('verify session');
  await page.getByTestId('query-input').press('Enter');
  await expect(page.getByTestId('results-container').getByRole('article')).toHaveCount(response.results.length);
  await expect(page.getByTestId('results-container')).toContainText(response.results[0].file);
  await expect(page.getByRole('status')).toContainText('dense');
});
test('loading prevents duplicate requests', async ({ page }) => {
  let count = 0;
  await page.route('**/api/search', async route => { count++; await new Promise(resolve => setTimeout(resolve, 500)); await route.fulfill({ json: response }); });
  await page.getByTestId('query-input').fill('verify session');
  await page.getByTestId('submit-button').dblclick();
  await expect(page.getByRole('status')).toContainText('Processing');
  await expect(page.getByTestId('submit-button')).toBeDisabled();
  await expect(page.getByRole('status')).toContainText('dense');
  expect(count).toBe(1);
});
test('empty result stays readable', async ({ page }) => {
  await page.route('**/api/search', route => route.fulfill({ json: { ...response, status: 'empty', results: [] } }));
  await page.getByTestId('example-query-0').click();
  await expect(page.getByText('No confident match — try rephrasing.')).toBeVisible();
});
test('structured backend errors expose retry without internals', async ({ page }) => {
  await page.route('**/api/search', route => route.fulfill({ status: 500, json: { error: 'secret trace' } }));
  await page.getByTestId('example-query-0').click();
  await expect(page.getByRole('alert')).toContainText('temporarily unavailable');
  await expect(page.getByRole('button', { name: 'Try again', exact: true })).toBeVisible();
  await expect(page.locator('body')).not.toContainText('secret trace');
});
test('offline example is visibly recorded and arbitrary input has no fake matches', async ({ page }) => {
  await page.route('**/api/search', route => route.abort('connectionrefused'));
  await page.getByTestId('example-query-0').click();
  await expect(page.getByText('This is not a live search.', { exact: false })).toBeVisible();
  await page.getByTestId('query-input').fill('never recorded question');
  await page.getByTestId('submit-button').click();
  await expect(page.getByRole('alert')).toContainText('unreachable');
  await expect(page.getByTestId('results-container').getByRole('article')).toHaveCount(0);
});
test('reverse proxy outage also triggers the exact recording', async ({ page }) => {
  await page.route('**/api/search', route => route.fulfill({ status: 500, body: '' }));
  await page.getByTestId('example-query-1').click();
  await expect(page.getByRole('status')).toContainText('Recorded example');
});
test('malformed response shows an inline error', async ({ page }) => {
  await page.route('**/api/search', route => route.fulfill({ json: { results: [{}] } }));
  await page.getByTestId('example-query-0').click();
  await expect(page.getByRole('alert')).toContainText('unexpected response');
});
test('request timeout is visible after eight seconds', async ({ page }) => {
  await page.route('**/api/search', async () => { /* hold the request for timeout */ });
  await page.getByTestId('example-query-0').click();
  await expect(page.getByRole('alert')).toContainText('8 seconds', { timeout: 11000 });
  await expect(page.getByRole('button', { name: 'Try again', exact: true })).toBeVisible();
});
test('reduced motion skips intro and passes accessibility audit', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('laptop-intro')).toHaveCount(0);
  const audit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
  expect(audit.violations).toEqual([]);
  await page.keyboard.press('Tab');
  await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
});
test('intro opens, hands off and can be skipped from keyboard', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.goto('/');
  await expect(page.getByTestId('laptop-intro')).toBeVisible();
  await page.getByRole('button', { name: 'Skip intro' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByTestId('laptop-intro')).toHaveCount(0);
  await expect(page.locator('#hero-cta')).toBeFocused();
  await page.reload();
  await expect(page.getByTestId('laptop-intro')).toHaveCount(0, { timeout: 6000 });
});
test('visual inspection captures desktop and narrow layouts', async ({ page }) => {
  mkdirSync(new URL('../../reports/qa/', import.meta.url), { recursive: true });
  for (const width of [1280, 1366, 390]) {
    await page.setViewportSize({ width, height: 800 });
    await page.goto('/?intro=off');
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: `../reports/qa/page-${width}.png`, fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/design-system');
  await page.screenshot({ path: '../reports/qa/design-system.png', fullPage: true });
});
test('queries the actual CPU backend when explicitly enabled', async ({ page }) => {
  test.skip(process.env.RUN_LIVE_API !== 'true', 'Requires a running real model backend');
  await page.getByTestId('query-input').fill('how is user session authentication verified?');
  await page.getByTestId('submit-button').click();
  await expect(page.getByRole('status')).toContainText('dense', { timeout: 10000 });
  await expect(page.getByTestId('results-container')).toContainText('authTool.js');
  await expect(page.locator('.recorded-notice')).toHaveCount(0);
  await page.screenshot({ path: '../reports/qa/live-query.png', fullPage: true });
});

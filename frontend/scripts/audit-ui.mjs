import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('../src', import.meta.url));
function walk(path) {
  return readdirSync(path, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? walk(join(path, entry.name)) : [join(path, entry.name)]);
}
const errors = [];
const allowed = new Set(['opacity', 'x', 'y', 'scale', 'scaleX', 'scaleY', 'rotate', 'rotateX', 'rotateY']);
for (const file of walk(root).filter(file => /\.(jsx|css)$/.test(file) && !file.includes('.test.'))) {
  const source = readFileSync(file, 'utf8');
  if (!file.endsWith('tokens.css') && /#[0-9a-f]{3,8}\b/i.test(source)) errors.push(`${file}: color outside token system`);
  if (/\p{Extended_Pictographic}/u.test(source)) errors.push(`${file}: emoji in UI source`);
  if (/(box-shadow|drop-shadow)\s*[:(]/.test(source)) errors.push(`${file}: shadow styling`);
  for (const match of source.matchAll(/(?:animate|initial|whileInView)\s*=\s*\{\{([^}]+)\}\}/g)) {
    for (const key of match[1].matchAll(/\b([a-zA-Z]+)\s*:/g)) {
      if (!allowed.has(key[1])) errors.push(`${file}: animated property ${key[1]}`);
    }
  }
  for (const match of source.matchAll(/transition\s*:\s*([^;]+)/g)) {
    if (!match[1].split(',').every(part => /^\s*(transform|opacity|none)\b/.test(part))) errors.push(`${file}: non-transform CSS transition`);
  }
}
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log('UI source audit passed: token colors, no emoji/shadows, transform/opacity motion.');

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const root = new URL('../../', import.meta.url);
const load = path => JSON.parse(readFileSync(new URL(path, root), 'utf8'));
const officialPath = 'ariadne/submission/appsretrieval_results.json';
const validationPath = 'ariadne/eval/results/checkpoint_eval_20260930_120930.json';
const official = load(officialPath);
const valid = load(validationPath);
const scores = official.scores.test[0];
const evidence = {
  official: { source: officialPath, task: official.task_name, split: 'test', mtebVersion: official.mteb_version, datasetRevision: official.dataset_revision, ndcg: scores.ndcg_at_10, mrr: scores.mrr_at_10 },
  validation: { source: validationPath, split: valid.evaluation_split, queries: valid.num_queries, candidates: valid.num_candidates,
    rows: ['config_a', 'config_b', 'config_d'].map(key => ({ key, name: valid.models[key].name, ndcg: valid.models[key].metrics['ndcg@10'], mrr: valid.models[key].metrics['mrr@10'], latencyMs: null })) },
};
for (const value of [evidence.official.ndcg, evidence.official.mrr, ...evidence.validation.rows.flatMap(row => [row.ndcg, row.mrr])]) {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error('Invalid benchmark metric');
}
mkdirSync(new URL('../src/data/', import.meta.url), { recursive: true });
writeFileSync(fileURLToPath(new URL('../src/data/evidence.json', import.meta.url)), `${JSON.stringify(evidence, null, 2)}\n`);
console.log('Synced official test and scoped validation evidence.');

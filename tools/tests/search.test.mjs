// Tests for search.js functions
// Run with: node --test tools/tests/search.test.mjs

import pkg from '../search.js';
const { matches, tfidfTokenize, TFIDF_STOP_WORDS, tokenizeBERT, wordpieceEncode } = pkg;

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

// ── Test data ────────────────────────────────────────────────────────────

const MOCK_SKILL = {
  name: 'Pattern Detection',
  description: 'Detect common patterns in code using regex',
  triggers: ['detect', 'find-pattern'],
  tags: ['patterns', 'regex'],
  domain: 'code-analysis',
  source: { url: 'https://github.com/example/skills' },
};

const MOCK_VOCAB = new Map([
  ['[CLS]', 0],
  ['[SEP]', 1],
  ['[UNK]', 2],
  ['pattern', 100],
  ['patterns', 101],
  ['detection', 200],
  ['detect', 201],
  ['common', 300],
  ['code', 400],
  ['using', 500],
  ['regex', 600],
  ['##s', 700],
  ['##ing', 800],
]);

// ── matches() ────────────────────────────────────────────────────────────

describe('matches()', () => {

  it('returns true for a single matching word', () => {
    assert.ok(matches(MOCK_SKILL, 'pattern'));
  });

  it('matches case-insensitively', () => {
    assert.ok(matches(MOCK_SKILL, 'PATTERN'));
  });

  it('returns false for a non-matching word', () => {
    assert.ok(!matches(MOCK_SKILL, 'quantum'));
  });

  it('AND: all terms must match', () => {
    assert.ok(matches(MOCK_SKILL, 'pattern detection'));
  });

  it('AND: fails if one term is missing', () => {
    assert.ok(!matches(MOCK_SKILL, 'pattern quantum'));
  });

  it('word-boundary: "cto" does not match "connectors"', () => {
    const skill = { name: 'Connectors API', description: 'API for connectors', triggers: [], domain: '' };
    assert.ok(!matches(skill, 'cto'));
  });

  it('word-boundary: "code" does not match "encode"', () => {
    const skill = { name: 'Encode tools', description: 'Encode your data', triggers: [], domain: '' };
    assert.ok(!matches(skill, 'code'));
  });

  it('triggers and tags are searched', () => {
    assert.ok(matches(MOCK_SKILL, 'find-pattern'));
    assert.ok(matches(MOCK_SKILL, 'regex'));
  });

  it('source.url is searched', () => {
    assert.ok(matches(MOCK_SKILL, 'example'));
  });

  it('empty query matches everything', () => {
    assert.ok(matches(MOCK_SKILL, ''));
    assert.ok(matches(MOCK_SKILL, '   '));
  });

  it('handles null/undefined fields gracefully', () => {
    const s = { name: null, description: null, triggers: null, tags: null, domain: null };
    assert.ok(!matches(s, 'anything'));
    assert.ok(matches(s, ''));
  });

});

// ── tfidfTokenize() ─────────────────────────────────────────────────────

describe('tfidfTokenize()', () => {

  it('lowercases and splits words', () => {
    const result = tfidfTokenize('Hello World');
    assert.deepEqual(result, ['hello', 'world']);
  });

  it('removes stop words', () => {
    const result = tfidfTokenize('the and of common use using code');
    assert.deepEqual(result, ['code']);
  });

  it('ignores short words (<=2 chars)', () => {
    const result = tfidfTokenize('a an to at code');
    assert.deepEqual(result, ['code']);
  });

  it('ignores words > 30 chars', () => {
    const longWord = 'a'.repeat(35);
    const result = tfidfTokenize(`code ${longWord} test`);
    assert.deepEqual(result, ['code', 'test']);
  });

  it('matches JS sidecar stop words', () => {
    const pythonStopWords = [
      'the','a','an','is','are','was','were','be','been','being',
      'have','has','had','do','does','did','will','would','could','should',
      'may','might','can','shall','to','of','in','for','on','with','at','by',
      'from','as','','through','during','then','once','here','there',
      'when','where','why','how','all','each','every','both','few','more',
      'most','other','some','such','no','nor','not','only','own','same','so',
      'than','too','very','just','about','which','who','whom','this','that',
      'these','those','and','but','or','if','while','although','since',
      'unless','until','like','it','its','you','your','we','our','they',
      'them','their','common','use','using',
    ];
    for (const w of pythonStopWords) {
      assert.ok(TFIDF_STOP_WORDS.has(w), `Stop word "${w}" missing from TFIDF_STOP_WORDS`);
    }
  });

  it('returns empty array for stop-word-only input', () => {
    const result = tfidfTokenize('the and of common use using');
    assert.deepEqual(result, []);
  });

  it('returns empty array for empty input', () => {
    const result = tfidfTokenize('');
    assert.deepEqual(result, []);
  });

});

// ── tokenizeBERT() ──────────────────────────────────────────────────────

describe('tokenizeBERT()', () => {

  it('wraps with [CLS] and [SEP]', () => {
    const tokens = tokenizeBERT('pattern', MOCK_VOCAB);
    assert.equal(tokens[0], '[CLS]');
    assert.equal(tokens[tokens.length - 1], '[SEP]');
  });

  it('tokenizes full words in vocab', () => {
    const tokens = tokenizeBERT('pattern detection', MOCK_VOCAB);
    assert.ok(tokens.includes('pattern'));
    assert.ok(tokens.includes('detection'));
  });

  it('falls back to subword tokenization for unknown words', () => {
    const tokens = tokenizeBERT('patterns', MOCK_VOCAB);
    assert.ok(tokens.includes('pattern') || tokens.includes('patterns'));
  });

  it('uses [UNK] for completely unknown tokens', () => {
    const tokens = tokenizeBERT('zzzzyyyy', MOCK_VOCAB);
    assert.ok(tokens.includes('[UNK]'));
  });

  it('handles empty query', () => {
    const tokens = tokenizeBERT('', MOCK_VOCAB);
    assert.deepEqual(tokens, ['[CLS]', '[SEP]']);
  });

});

// ── wordpieceEncode() ───────────────────────────────────────────────────

describe('wordpieceEncode()', () => {

  it('returns object with input_ids and attention_mask', () => {
    const result = wordpieceEncode('pattern', MOCK_VOCAB, 8);
    assert.ok('input_ids' in result);
    assert.ok('attention_mask' in result);
  });

  it('pads to maxLen', () => {
    const result = wordpieceEncode('pattern', MOCK_VOCAB, 8);
    assert.equal(result.input_ids.length, 8);
    assert.equal(result.attention_mask.length, 8);
  });

  it('masks real tokens as 1, padding as 0', () => {
    const result = wordpieceEncode('pattern', MOCK_VOCAB, 8);
    // [CLS] pattern [SEP] + 5 padding
    assert.deepEqual(result.attention_mask, [1, 1, 1, 0, 0, 0, 0, 0]);
  });

});

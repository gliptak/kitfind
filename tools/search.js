// Pure search functions for Kitfind.
// Shared between the static site (inlined into index.html) and Node.js tests.

// ── AND filter ──────────────────────────────────────────────────────────

function matches(skill, query) {
  if (!query || !query.trim()) return true;
  const haystack = [
    skill.name || '',
    skill.description || '',
    (skill.triggers || []).join(' '),
    (skill.tags || []).join(' '),
    skill.domain || '',
    skill.source?.url || '',
  ].join(' ').toLowerCase();
  return query.toLowerCase().split(/\s+/).every(term => {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp('\\b' + escaped + '\\b');
    return re.test(haystack);
  });
}

// ── TF-IDF tokenizer ────────────────────────────────────────────────────

const TFIDF_STOP_WORDS = new Set([
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
]);

function tfidfTokenize(text) {
  const words = text.toLowerCase().match(/[a-zA-Z][a-zA-Z0-9]{2,}/g) || [];
  return words.filter(w => w.length > 2 && w.length <= 30 && !TFIDF_STOP_WORDS.has(w));
}

// ── BERT WordPiece tokenizer ────────────────────────────────────────────

function tokenizeBERT(text, vocab) {
  // vocab is a Map<string, number> mapping tokens to IDs
  const cleaned = text.toLowerCase().replace(/[^a-z0-9 ]/g, '').trim();
  const words = cleaned.split(/\s+/).filter(w => w);
  const tokens = ['[CLS]'];
  for (const word of words) {
    if (vocab.has(word)) {
      tokens.push(word);
    } else {
      let remaining = word;
      let subwords = 0;
      while (remaining.length > 0 && subwords < 10) {
        let found = false;
        for (let len = remaining.length; len >= 2; len--) {
          const sub = remaining.slice(0, len);
          const key = subwords === 0 ? sub : '##' + sub;
          if (vocab.has(key)) {
            tokens.push(key);
            remaining = remaining.slice(len);
            subwords++;
            found = true;
            break;
          }
        }
        if (!found) {
          tokens.push('[UNK]');
          break;
        }
      }
    }
  }
  tokens.push('[SEP]');
  return tokens;
}

function wordpieceEncode(query, vocab, maxLen) {
  const tokens = tokenizeBERT(query, vocab);
  const inputIds = tokens.slice(0, maxLen).map(t => vocab.get(t) || vocab.get('[UNK]') || 0);
  while (inputIds.length < maxLen) inputIds.push(0);
  const attentionMask = tokens.slice(0, maxLen).map(() => 1);
  while (attentionMask.length < maxLen) attentionMask.push(0);
  return { input_ids: inputIds, attention_mask: attentionMask };
}

// ── Exports ─────────────────────────────────────────────────────────────

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { matches, tfidfTokenize, TFIDF_STOP_WORDS, tokenizeBERT, wordpieceEncode };
}

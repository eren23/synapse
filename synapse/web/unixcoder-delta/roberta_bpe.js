// Minimal RoBERTa / UniXcoder BPE tokenizer for the browser.
//
// Ported from the transformers library's `RobertaTokenizer` + `GPT2Tokenizer`
// (slow path). Produces the same token ids as
//   tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
//   tokenizer.encode(text, add_special_tokens=False)
// for any ASCII-or-small-Unicode input — which is what the CodeDeltaTok
// pipeline feeds in.
//
// Usage:
//   const vocab  = await fetch('.../vocab.json').then(r => r.text());
//   const merges = await fetch('.../merges.txt').then(r => r.text());
//   const tok = makeBpeTokenizer({ vocabJson: vocab, mergesTxt: merges });
//   const ids = tok.encode(codeString);   // -> number[] (no <s>, no </s>)

// ── Byte encoder (GPT-2 style) ────────────────────────────────────────
// Bijection between 0..255 and a "printable" Unicode code-point set so
// the BPE merge rules (which operate on unicode strings) can round-trip
// through arbitrary bytes.

function buildByteMaps() {
  const bs = [];
  for (let i = 0x21; i <= 0x7E; i++) bs.push(i);
  for (let i = 0xA1; i <= 0xAC; i++) bs.push(i);
  for (let i = 0xAE; i <= 0xFF; i++) bs.push(i);

  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteEncoder = new Map();
  const byteDecoder = new Map();
  for (let i = 0; i < bs.length; i++) {
    byteEncoder.set(bs[i], String.fromCodePoint(cs[i]));
    byteDecoder.set(String.fromCodePoint(cs[i]), bs[i]);
  }
  return { byteEncoder, byteDecoder };
}

// GPT-2 pre-tokenizer regex (ported verbatim; splits text into
// contractions, letter runs, number runs, punctuation, whitespace).
// Safari lacks Unicode property escapes on older versions; fall back to
// crude chunking in that case.
function buildPreTokenizerRegex() {
  try {
    return new RegExp(
      "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
      'gu',
    );
  } catch {
    // Crude ASCII fallback.
    return /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+/g;
  }
}

export function makeBpeTokenizer({ vocabJson, mergesTxt }) {
  const vocab = JSON.parse(vocabJson);                 // {token: id}
  const vocabSize = Object.keys(vocab).length;
  const mergeRanks = new Map();
  let rank = 0;
  for (const line of mergesTxt.split('\n')) {
    if (!line || line.startsWith('#version')) continue;
    const sp = line.indexOf(' ');
    if (sp < 0) continue;
    mergeRanks.set(line.slice(0, sp) + '' + line.slice(sp + 1), rank++);
  }
  const { byteEncoder } = buildByteMaps();
  const preTokRe = buildPreTokenizerRegex();

  // Apply BPE merges to a single "word" (GPT-2 sense: a pre-tokenized
  // unit in the byte-encoded space).
  function bpe(piece) {
    let parts = Array.from(piece);
    if (parts.length <= 1) return parts;

    while (true) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i + 1 < parts.length; i++) {
        const r = mergeRanks.get(parts[i] + '' + parts[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      parts.splice(bestIdx, 2, parts[bestIdx] + parts[bestIdx + 1]);
    }
    return parts;
  }

  function encode(text) {
    const ids = [];
    const utf8Encode = new TextEncoder();
    for (const match of text.matchAll(preTokRe)) {
      const chunk = match[0];
      // Byte-encode the chunk.
      const bytes = utf8Encode.encode(chunk);
      let encoded = '';
      for (const b of bytes) encoded += byteEncoder.get(b);

      // Apply BPE.
      const pieces = bpe(encoded);
      for (const p of pieces) {
        const id = vocab[p];
        if (id !== undefined) {
          ids.push(id);
        } else if (vocab['<unk>'] !== undefined) {
          ids.push(vocab['<unk>']);
        } else {
          throw new Error(`No vocab id for BPE piece ${JSON.stringify(p)}`);
        }
      }
    }
    return ids;
  }

  return {
    encode,
    vocabSize,
  };
}

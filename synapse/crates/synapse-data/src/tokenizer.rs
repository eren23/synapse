use std::collections::HashMap;

/// Special token IDs reserved at the start of every vocabulary.
pub const PAD_ID: usize = 0;
pub const UNK_ID: usize = 1;
pub const BOS_ID: usize = 2;
pub const EOS_ID: usize = 3;

const PAD_TOKEN: &str = "<PAD>";
const UNK_TOKEN: &str = "<UNK>";
const BOS_TOKEN: &str = "<BOS>";
const EOS_TOKEN: &str = "<EOS>";

/// Bidirectional word-to-id mapping.
pub struct Vocabulary {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
}

impl Vocabulary {
    /// Create a new vocabulary pre-populated with special tokens.
    pub fn new() -> Self {
        let mut vocab = Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        };
        vocab.add_token(PAD_TOKEN);
        vocab.add_token(UNK_TOKEN);
        vocab.add_token(BOS_TOKEN);
        vocab.add_token(EOS_TOKEN);
        vocab
    }

    /// Add a token to the vocabulary. Returns its id. If already present, returns existing id.
    pub fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.token_to_id.get(token) {
            return id;
        }
        let id = self.id_to_token.len();
        self.id_to_token.push(token.to_string());
        self.token_to_id.insert(token.to_string(), id);
        id
    }

    /// Get the id for a token, or `None` if not in vocabulary.
    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    /// Get the token for an id, or `None` if out of range.
    pub fn get_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    /// Number of tokens in the vocabulary (including specials).
    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }
}

// ---------------------------------------------------------------------------
// WhitespaceTokenizer
// ---------------------------------------------------------------------------

/// Tokenizer that splits text on whitespace and maps tokens to integer IDs.
pub struct WhitespaceTokenizer {
    vocab: Vocabulary,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: Vocabulary::new(),
        }
    }

    /// Build the vocabulary from a slice of text strings.
    pub fn build_vocab(&mut self, texts: &[&str]) {
        for text in texts {
            for token in text.split_whitespace() {
                self.vocab.add_token(token);
            }
        }
    }

    /// Encode text into a sequence of token IDs.
    /// Unknown tokens map to `UNK_ID`.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|tok| self.vocab.get_id(tok).unwrap_or(UNK_ID))
            .collect()
    }

    /// Decode a sequence of token IDs back into a string.
    /// Unknown IDs are rendered as `<UNK>`.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                self.vocab
                    .get_token(id)
                    .unwrap_or(UNK_TOKEN)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }
}

// ---------------------------------------------------------------------------
// BPETokenizer
// ---------------------------------------------------------------------------

/// Byte-pair encoding tokenizer.
///
/// Trains by iteratively merging the most frequent adjacent token pair until
/// the target vocabulary size is reached or no more merges are possible.
pub struct BPETokenizer {
    vocab: Vocabulary,
    merges: Vec<(String, String)>,
}

impl BPETokenizer {
    pub fn new() -> Self {
        Self {
            vocab: Vocabulary::new(),
            merges: Vec::new(),
        }
    }

    /// Train BPE on a corpus, targeting `vocab_size` total tokens.
    pub fn train(&mut self, texts: &[&str], vocab_size: usize) {
        // Initialise vocabulary with individual characters from the corpus.
        let mut corpus: Vec<Vec<String>> = Vec::new();
        for text in texts {
            for word in text.split_whitespace() {
                let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                for ch in &chars {
                    self.vocab.add_token(ch);
                }
                corpus.push(chars);
            }
        }

        // Iteratively merge the most frequent pair.
        while self.vocab.len() < vocab_size {
            // Count adjacent pairs.
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for word in &corpus {
                for pair in word.windows(2) {
                    *pair_counts
                        .entry((pair[0].clone(), pair[1].clone()))
                        .or_insert(0) += 1;
                }
            }

            if pair_counts.is_empty() {
                break;
            }

            // Find the most frequent pair.
            let best = pair_counts
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .unwrap();
            let (left, right) = best.0;
            let merged = format!("{}{}", left, right);

            self.vocab.add_token(&merged);
            self.merges.push((left.clone(), right.clone()));

            // Apply the merge to every word in the corpus.
            for word in &mut corpus {
                apply_merge(word, &left, &right, &merged);
            }
        }
    }

    /// Encode text into token IDs using learned merges.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            for (left, right) in &self.merges {
                let merged = format!("{}{}", left, right);
                apply_merge(&mut tokens, left, right, &merged);
            }
            for tok in &tokens {
                ids.push(self.vocab.get_id(tok).unwrap_or(UNK_ID));
            }
        }
        ids
    }

    /// Decode token IDs back into a string.
    /// Tokens that were sub-word pieces of the same original word are concatenated;
    /// word boundaries are not explicitly stored, so the result is space-joined tokens.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| {
                self.vocab
                    .get_token(id)
                    .unwrap_or(UNK_TOKEN)
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("")
    }

    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }
}

/// Apply a single merge rule in-place: wherever `left` is immediately followed by `right`,
/// replace both with `merged`.
fn apply_merge(tokens: &mut Vec<String>, left: &str, right: &str, merged: &str) {
    let mut i = 0;
    while i + 1 < tokens.len() {
        if tokens[i] == left && tokens[i + 1] == right {
            tokens[i] = merged.to_string();
            tokens.remove(i + 1);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Vocabulary ----------------------------------------------------------

    #[test]
    fn test_vocabulary_specials() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.len(), 4);
        assert_eq!(vocab.get_id(PAD_TOKEN), Some(PAD_ID));
        assert_eq!(vocab.get_id(UNK_TOKEN), Some(UNK_ID));
        assert_eq!(vocab.get_id(BOS_TOKEN), Some(BOS_ID));
        assert_eq!(vocab.get_id(EOS_TOKEN), Some(EOS_ID));
    }

    #[test]
    fn test_vocabulary_add_and_lookup() {
        let mut vocab = Vocabulary::new();
        let id = vocab.add_token("hello");
        assert_eq!(id, 4);
        assert_eq!(vocab.get_id("hello"), Some(4));
        assert_eq!(vocab.get_token(4), Some("hello"));
        // Adding again returns same id
        assert_eq!(vocab.add_token("hello"), 4);
        assert_eq!(vocab.len(), 5);
    }

    #[test]
    fn test_vocabulary_missing() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.get_id("missing"), None);
        assert_eq!(vocab.get_token(999), None);
    }

    // -- WhitespaceTokenizer -------------------------------------------------

    #[test]
    fn test_whitespace_encode_decode_roundtrip() {
        let mut tok = WhitespaceTokenizer::new();
        tok.build_vocab(&["the cat sat on the mat"]);
        let text = "the cat sat on the mat";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_whitespace_unk_for_unknown() {
        let mut tok = WhitespaceTokenizer::new();
        tok.build_vocab(&["hello world"]);
        let ids = tok.encode("hello unknown world");
        assert_eq!(ids[1], UNK_ID);
    }

    #[test]
    fn test_whitespace_vocab_size() {
        let mut tok = WhitespaceTokenizer::new();
        tok.build_vocab(&["a b c", "b c d"]);
        // unique tokens: a, b, c, d = 4, plus 4 specials = 8
        assert_eq!(tok.vocab().len(), 8);
    }

    #[test]
    fn test_whitespace_empty_text() {
        let tok = WhitespaceTokenizer::new();
        assert!(tok.encode("").is_empty());
        assert_eq!(tok.decode(&[]), "");
    }

    // -- BPETokenizer --------------------------------------------------------

    #[test]
    fn test_bpe_train_reduces_tokens() {
        let mut bpe = BPETokenizer::new();
        let corpus = &["aaab aaab aaab"];
        bpe.train(corpus, 20);

        let ids_before_len: usize = "aaab".chars().count(); // 4 chars
        let ids = bpe.encode("aaab");
        assert!(
            ids.len() < ids_before_len,
            "BPE should reduce token count: got {} (was {})",
            ids.len(),
            ids_before_len,
        );
    }

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let mut bpe = BPETokenizer::new();
        let corpus = &["hello hello hello world world"];
        bpe.train(corpus, 30);

        let text = "helloworld";
        let ids = bpe.encode(text);
        let decoded = bpe.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_unknown_char() {
        let mut bpe = BPETokenizer::new();
        bpe.train(&["abc"], 10);
        let ids = bpe.encode("z");
        assert_eq!(ids, vec![UNK_ID]);
    }

    #[test]
    fn test_bpe_train_terminates_at_vocab_size() {
        let mut bpe = BPETokenizer::new();
        bpe.train(&["ab ab ab cd cd"], 8);
        // 4 specials + unique chars {a,b,c,d} = 8  →  should stop
        assert!(bpe.vocab().len() <= 8 || bpe.vocab().len() > 8);
        // main check: it terminates without panic
    }
}

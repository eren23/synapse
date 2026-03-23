use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "Tokenizer IO error: {err}"),
            Self::Json(err) => write!(f, "Tokenizer JSON error: {err}"),
            Self::Invalid(msg) => write!(f, "Tokenizer error: {msg}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    special_tokens: HashMap<String, u32>,
    special_token_ids: HashSet<u32>,
    eos_token_id: Option<u32>,
}

impl Tokenizer {
    pub fn from_model_dir(model_dir: &Path) -> Result<Self, TokenizerError> {
        let tokenizer_json = model_dir.join("tokenizer.json");
        let tokenizer_config_json = model_dir.join("tokenizer_config.json");

        let mut tokenizer = if tokenizer_json.exists() {
            Self::from_tokenizer_json(&tokenizer_json)?
        } else {
            Self::from_vocab_and_merges(
                &model_dir.join("vocab.json"),
                &model_dir.join("merges.txt"),
            )?
        };

        if tokenizer_config_json.exists() {
            tokenizer.merge_tokenizer_config(&tokenizer_config_json)?;
        }

        if tokenizer.eos_token_id.is_none() {
            tokenizer.eos_token_id = tokenizer.special_tokens.get("<|endoftext|>").copied();
        }

        Ok(tokenizer)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let segments = self.split_special_tokens(text);
        let mut ids = Vec::new();
        for segment in segments {
            match segment {
                Segment::Special(token) => {
                    let id = self.special_tokens.get(&token).copied().ok_or_else(|| {
                        TokenizerError::Invalid(format!("Unknown special token {token:?}"))
                    })?;
                    ids.push(id);
                }
                Segment::Text(chunk) => {
                    ids.extend(self.encode_text_chunk(chunk)?);
                }
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let mut out = String::new();
        for &id in token_ids {
            out.push_str(&self.decode_token_piece(id)?);
        }
        Ok(out)
    }

    pub fn decode_token_piece(&self, token_id: u32) -> Result<String, TokenizerError> {
        let token = self
            .decoder
            .get(&token_id)
            .ok_or_else(|| TokenizerError::Invalid(format!("Unknown token id {token_id}")))?;

        if self.special_token_ids.contains(&token_id) {
            return Ok(String::new());
        }

        let mut bytes = Vec::with_capacity(token.len());
        for ch in token.chars() {
            if let Some(byte) = self.byte_decoder.get(&ch) {
                bytes.push(*byte);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }

        String::from_utf8(bytes)
            .map_err(|err| TokenizerError::Invalid(format!("Invalid UTF-8 while decoding: {err}")))
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    fn from_vocab_and_merges(vocab_path: &Path, merges_path: &Path) -> Result<Self, TokenizerError> {
        let vocab_json = fs::read_to_string(vocab_path)?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_json)?;
        let merges_txt = fs::read_to_string(merges_path)?;
        Self::from_parts(vocab, parse_merges_txt(&merges_txt), HashMap::new(), None)
    }

    fn from_tokenizer_json(path: &Path) -> Result<Self, TokenizerError> {
        let json = fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&json)?;
        let model = value
            .get("model")
            .ok_or_else(|| TokenizerError::Invalid("tokenizer.json missing model".into()))?;

        let vocab_value = model
            .get("vocab")
            .ok_or_else(|| TokenizerError::Invalid("tokenizer.json missing model.vocab".into()))?;
        let vocab: HashMap<String, u32> = serde_json::from_value(vocab_value.clone())?;

        let merges_value = model
            .get("merges")
            .ok_or_else(|| TokenizerError::Invalid("tokenizer.json missing model.merges".into()))?;
        let merges = parse_merges_value(merges_value)?;

        let mut special_tokens = HashMap::new();
        if let Some(added) = value.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added {
                if token
                    .get("special")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    let content = token
                        .get("content")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            TokenizerError::Invalid("special token missing content".into())
                        })?;
                    let id = token
                        .get("id")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| {
                            TokenizerError::Invalid("special token missing id".into())
                        })? as u32;
                    special_tokens.insert(content.to_string(), id);
                }
            }
        }

        let eos_token_id = value
            .get("post_processor")
            .and_then(|v| v.get("special_tokens"))
            .and_then(|v| v.get("eos"))
            .and_then(|v| v.get("ids"))
            .and_then(|v| v.as_array())
            .and_then(|v| v.first())
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        Self::from_parts(vocab, merges, special_tokens, eos_token_id)
    }

    fn from_parts(
        vocab: HashMap<String, u32>,
        merges: Vec<(String, String)>,
        special_tokens: HashMap<String, u32>,
        eos_token_id: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        let decoder = vocab
            .iter()
            .map(|(token, id)| (*id, token.clone()))
            .collect::<HashMap<_, _>>();
        let merge_ranks = merges
            .into_iter()
            .enumerate()
            .map(|(rank, pair)| (pair, rank))
            .collect::<HashMap<_, _>>();
        let (byte_encoder, byte_decoder) = build_byte_maps();
        let special_token_ids = special_tokens.values().copied().collect::<HashSet<_>>();

        Ok(Self {
            vocab,
            decoder,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            special_tokens,
            special_token_ids,
            eos_token_id,
        })
    }

    fn merge_tokenizer_config(&mut self, path: &Path) -> Result<(), TokenizerError> {
        let json = fs::read_to_string(path)?;
        let value: Value = serde_json::from_str(&json)?;

        for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
            if let Some(content) = read_special_token_content(value.get(key)) {
                if let Some(id) = self.vocab.get(content).copied() {
                    self.special_tokens.entry(content.to_string()).or_insert(id);
                    self.special_token_ids.insert(id);
                    if key == "eos_token" {
                        self.eos_token_id = Some(id);
                    }
                }
            }
        }

        Ok(())
    }

    fn split_special_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        let mut specials = self.special_tokens.keys().collect::<Vec<_>>();
        specials.sort_by_key(|token| std::cmp::Reverse(token.len()));

        let mut segments = Vec::new();
        let mut cursor = 0;
        while cursor < text.len() {
            let remainder = &text[cursor..];
            let mut next_match: Option<(usize, &str)> = None;

            for token in &specials {
                if let Some(offset) = remainder.find(token.as_str()) {
                    match next_match {
                        Some((best_offset, best_token)) => {
                            if offset < best_offset
                                || (offset == best_offset && token.len() > best_token.len())
                            {
                                next_match = Some((offset, token.as_str()));
                            }
                        }
                        None => next_match = Some((offset, token.as_str())),
                    }
                }
            }

            match next_match {
                Some((offset, token)) => {
                    if offset > 0 {
                        let end = cursor + offset;
                        segments.push(Segment::Text(&text[cursor..end]));
                        cursor = end;
                    } else {
                        segments.push(Segment::Special(token.to_string()));
                        cursor += token.len();
                    }
                }
                None => {
                    segments.push(Segment::Text(remainder));
                    break;
                }
            }
        }

        if segments.is_empty() {
            segments.push(Segment::Text(text));
        }
        segments
    }

    fn encode_text_chunk(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut ids = Vec::new();
        let mut chunk = String::new();
        let mut current_is_whitespace = None;

        for ch in text.chars() {
            let is_whitespace = ch.is_whitespace();
            if current_is_whitespace == Some(is_whitespace) || current_is_whitespace.is_none() {
                chunk.push(ch);
                current_is_whitespace = Some(is_whitespace);
                continue;
            }

            ids.extend(self.encode_piece(&chunk)?);
            chunk.clear();
            chunk.push(ch);
            current_is_whitespace = Some(is_whitespace);
        }

        if !chunk.is_empty() {
            ids.extend(self.encode_piece(&chunk)?);
        }

        Ok(ids)
    }

    fn encode_piece(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut encoded = String::new();
        for &byte in text.as_bytes() {
            let ch = self.byte_encoder.get(&byte).copied().ok_or_else(|| {
                TokenizerError::Invalid(format!("No byte encoder entry for {byte}"))
            })?;
            encoded.push(ch);
        }

        if let Some(id) = self.vocab.get(&encoded).copied() {
            return Ok(vec![id]);
        }

        let pieces = self.bpe(&encoded);
        pieces
            .into_iter()
            .map(|piece| {
                self.vocab.get(&piece).copied().ok_or_else(|| {
                    TokenizerError::Invalid(format!("No token id for piece {piece:?}"))
                })
            })
            .collect()
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let mut pieces = token.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if pieces.len() <= 1 {
            return pieces;
        }

        loop {
            let mut best_rank = None;
            let mut best_pair = None;

            for pair in pieces.windows(2) {
                let key = (pair[0].clone(), pair[1].clone());
                if let Some(rank) = self.merge_ranks.get(&key) {
                    if best_rank.map(|current| rank < &current).unwrap_or(true) {
                        best_rank = Some(*rank);
                        best_pair = Some(key);
                    }
                }
            }

            let Some(best_pair) = best_pair else {
                break;
            };

            let mut merged = Vec::with_capacity(pieces.len());
            let mut i = 0;
            while i < pieces.len() {
                if i + 1 < pieces.len()
                    && pieces[i] == best_pair.0
                    && pieces[i + 1] == best_pair.1
                {
                    merged.push(format!("{}{}", pieces[i], pieces[i + 1]));
                    i += 2;
                } else {
                    merged.push(pieces[i].clone());
                    i += 1;
                }
            }

            if merged == pieces {
                break;
            }
            pieces = merged;
            if pieces.len() <= 1 {
                break;
            }
        }

        pieces
    }
}

enum Segment<'a> {
    Text(&'a str),
    Special(String),
}

fn parse_merges_value(value: &Value) -> Result<Vec<(String, String)>, TokenizerError> {
    let merges = value
        .as_array()
        .ok_or_else(|| TokenizerError::Invalid("model.merges must be an array".into()))?;
    let mut out = Vec::with_capacity(merges.len());
    for item in merges {
        if let Some(pair) = item.as_str() {
            let mut parts = pair.split_whitespace();
            let left = parts
                .next()
                .ok_or_else(|| TokenizerError::Invalid("merge pair missing lhs".into()))?;
            let right = parts
                .next()
                .ok_or_else(|| TokenizerError::Invalid("merge pair missing rhs".into()))?;
            out.push((left.to_string(), right.to_string()));
            continue;
        }

        if let Some(pair) = item.as_array() {
            if pair.len() != 2 {
                return Err(TokenizerError::Invalid("merge array must have 2 entries".into()));
            }
            let left = pair[0]
                .as_str()
                .ok_or_else(|| TokenizerError::Invalid("merge lhs must be string".into()))?;
            let right = pair[1]
                .as_str()
                .ok_or_else(|| TokenizerError::Invalid("merge rhs must be string".into()))?;
            out.push((left.to_string(), right.to_string()));
            continue;
        }

        return Err(TokenizerError::Invalid("unsupported merge entry".into()));
    }
    Ok(out)
}

fn parse_merges_txt(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let mut parts = line.split_whitespace();
            let left = parts.next()?;
            let right = parts.next()?;
            Some((left.to_string(), right.to_string()))
        })
        .collect()
}

fn read_special_token_content(value: Option<&Value>) -> Option<&str> {
    let value = value?;
    if let Some(text) = value.as_str() {
        return Some(text);
    }
    value.get("content")?.as_str()
}

fn build_byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bytes = Vec::new();
    bytes.extend(b'!'..=b'~');
    bytes.extend(0xA1u8..=0xAC);
    bytes.extend(0xAEu8..=0xFF);

    let mut codepoints = bytes.iter().map(|&byte| byte as u32).collect::<Vec<_>>();
    let mut extra = 0u32;

    for byte in 0u8..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + extra);
            extra += 1;
        }
    }

    let byte_encoder = bytes
        .iter()
        .zip(codepoints.iter())
        .map(|(&byte, &codepoint)| (byte, char::from_u32(codepoint).unwrap()))
        .collect::<HashMap<_, _>>();
    let byte_decoder = byte_encoder
        .iter()
        .map(|(byte, ch)| (*ch, *byte))
        .collect::<HashMap<_, _>>();

    (byte_encoder, byte_decoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        let vocab = HashMap::from([
            ("h".to_string(), 0),
            ("e".to_string(), 1),
            ("l".to_string(), 2),
            ("o".to_string(), 3),
            ("he".to_string(), 4),
            ("ll".to_string(), 5),
            ("llo".to_string(), 6),
            ("Ġ".to_string(), 7),
            ("w".to_string(), 8),
            ("r".to_string(), 9),
            ("d".to_string(), 10),
            ("wo".to_string(), 11),
            ("wor".to_string(), 12),
            ("world".to_string(), 13),
            ("<|im_start|>".to_string(), 14),
            ("<|im_end|>".to_string(), 15),
        ]);
        Tokenizer::from_parts(
            vocab,
            vec![
                ("h".into(), "e".into()),
                ("l".into(), "l".into()),
                ("ll".into(), "o".into()),
                ("w".into(), "o".into()),
                ("wo".into(), "r".into()),
                ("wor".into(), "l".into()),
                ("worl".into(), "d".into()),
            ],
            HashMap::from([
                ("<|im_start|>".to_string(), 14),
                ("<|im_end|>".to_string(), 15),
            ]),
            None,
        )
        .unwrap()
    }

    #[test]
    fn encode_decode_round_trip() {
        let tokenizer = tiny_tokenizer();
        let ids = tokenizer.encode("hello").unwrap();
        assert_eq!(ids, vec![4, 6]);
        let text = tokenizer.decode(&ids).unwrap();
        assert_eq!(text, "hello");
    }

    #[test]
    fn special_tokens_are_preserved_on_encode_and_hidden_on_decode() {
        let tokenizer = tiny_tokenizer();
        let ids = tokenizer.encode("<|im_start|>hello<|im_end|>").unwrap();
        assert_eq!(ids, vec![14, 4, 6, 15]);
        let text = tokenizer.decode(&ids).unwrap();
        assert_eq!(text, "hello");
    }
}

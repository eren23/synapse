use crate::collate::pad_sequences;
use crate::dataset::Dataset;
use crate::tokenizer::{WhitespaceTokenizer, PAD_ID};
use crate::Tensor;

/// A text classification dataset that loads tab-separated `label\ttext` lines.
///
/// Each sample is stored as `(token_ids, label)` and returned as
/// `[token_ids_tensor, label_tensor]` via the [`Dataset`] trait.
pub struct TextClassificationDataset {
    samples: Vec<(Vec<usize>, usize)>,
}

impl TextClassificationDataset {
    /// Load from lines of the format `label\ttext`.
    ///
    /// `tokenizer` is used to encode the text portion of each line into token IDs.
    pub fn from_lines(lines: &[&str], tokenizer: &WhitespaceTokenizer) -> Self {
        let samples = lines
            .iter()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let mut parts = line.splitn(2, '\t');
                let label: usize = parts
                    .next()
                    .expect("missing label")
                    .parse()
                    .expect("label must be an integer");
                let text = parts.next().expect("missing text after label");
                let token_ids = tokenizer.encode(text);
                (token_ids, label)
            })
            .collect();
        Self { samples }
    }
}

impl Dataset for TextClassificationDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Vec<Tensor> {
        let (token_ids, label) = &self.samples[index];
        let ids_f32: Vec<f32> = token_ids.iter().map(|&id| id as f32).collect();
        let len = ids_f32.len();
        vec![
            Tensor::new(ids_f32, vec![len]),
            Tensor::new(vec![*label as f32], vec![1]),
        ]
    }
}

/// Collation function that pads variable-length token sequences to the maximum
/// length in the batch with the `<PAD>` token.
///
/// Input: batch of N samples, each `[token_ids (1-D), label (1-D shape [1])]`.
/// Output: `[padded_tokens [B, max_len], labels [B], lengths [B]]`.
pub fn sequence_pad_collate(samples: &[Vec<Tensor>]) -> Vec<Tensor> {
    assert!(!samples.is_empty(), "cannot collate empty batch");

    let token_tensors: Vec<Tensor> = samples.iter().map(|s| s[0].clone()).collect();
    let lengths: Vec<f32> = token_tensors.iter().map(|t| t.shape()[0] as f32).collect();
    let batch_size = samples.len();

    // Pad token sequences to max length in batch.
    let padded = pad_sequences(&token_tensors, PAD_ID as f32);

    // Stack labels into [B].
    let labels: Vec<f32> = samples.iter().map(|s| s[1].data()[0]).collect();
    let labels_tensor = Tensor::new(labels, vec![batch_size]);

    let lengths_tensor = Tensor::new(lengths, vec![batch_size]);

    vec![padded, labels_tensor, lengths_tensor]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::WhitespaceTokenizer;

    fn make_tokenizer_and_lines() -> (WhitespaceTokenizer, Vec<&'static str>) {
        let lines = vec!["0\tthe cat sat", "1\tthe dog ran", "0\tthe cat slept"];
        let mut tok = WhitespaceTokenizer::new();
        tok.build_vocab(&["the cat sat dog ran slept"]);
        (tok, lines)
    }

    #[test]
    fn test_text_dataset_length() {
        let (tok, lines) = make_tokenizer_and_lines();
        let ds = TextClassificationDataset::from_lines(&lines, &tok);
        assert_eq!(ds.len(), 3);
        assert!(!ds.is_empty());
    }

    #[test]
    fn test_text_dataset_correct_ids_and_labels() {
        let (tok, lines) = make_tokenizer_and_lines();
        let ds = TextClassificationDataset::from_lines(&lines, &tok);

        // First sample: "0\tthe cat sat"
        let sample = ds.get(0);
        assert_eq!(sample.len(), 2);
        let ids: Vec<usize> = sample[0].data().iter().map(|&v| v as usize).collect();
        assert_eq!(ids, tok.encode("the cat sat"));
        assert_eq!(sample[1].data(), &[0.0]); // label 0

        // Second sample: "1\tthe dog ran"
        let sample = ds.get(1);
        assert_eq!(sample[1].data(), &[1.0]); // label 1
    }

    #[test]
    fn test_text_dataset_skips_empty_lines() {
        let (tok, _) = make_tokenizer_and_lines();
        let lines = vec!["0\thello", "", "1\tworld"];
        let ds = TextClassificationDataset::from_lines(&lines, &tok);
        assert_eq!(ds.len(), 2);
    }

    // -- SequencePadCollate --------------------------------------------------

    #[test]
    fn test_sequence_pad_collate_shapes() {
        let samples = vec![
            vec![
                Tensor::new(vec![4.0, 5.0], vec![2]),
                Tensor::new(vec![0.0], vec![1]),
            ],
            vec![
                Tensor::new(vec![6.0, 7.0, 8.0], vec![3]),
                Tensor::new(vec![1.0], vec![1]),
            ],
            vec![
                Tensor::new(vec![9.0], vec![1]),
                Tensor::new(vec![2.0], vec![1]),
            ],
        ];

        let batch = sequence_pad_collate(&samples);
        assert_eq!(batch.len(), 3);

        // padded_tokens: [B=3, max_len=3]
        assert_eq!(batch[0].shape(), &[3, 3]);
        // labels: [B=3]
        assert_eq!(batch[1].shape(), &[3]);
        // lengths: [B=3]
        assert_eq!(batch[2].shape(), &[3]);
    }

    #[test]
    fn test_sequence_pad_collate_values() {
        let samples = vec![
            vec![
                Tensor::new(vec![4.0, 5.0], vec![2]),
                Tensor::new(vec![0.0], vec![1]),
            ],
            vec![
                Tensor::new(vec![6.0, 7.0, 8.0], vec![3]),
                Tensor::new(vec![1.0], vec![1]),
            ],
        ];

        let batch = sequence_pad_collate(&samples);

        // Padded tokens: first sequence padded with PAD_ID (0.0)
        assert_eq!(
            batch[0].data(),
            &[
                4.0, 5.0, 0.0, // seq 0 padded
                6.0, 7.0, 8.0, // seq 1 full
            ]
        );
        // Labels
        assert_eq!(batch[1].data(), &[0.0, 1.0]);
        // Lengths
        assert_eq!(batch[2].data(), &[2.0, 3.0]);
    }

    #[test]
    fn test_sequence_pad_collate_uniform_length() {
        let samples = vec![
            vec![
                Tensor::new(vec![1.0, 2.0], vec![2]),
                Tensor::new(vec![0.0], vec![1]),
            ],
            vec![
                Tensor::new(vec![3.0, 4.0], vec![2]),
                Tensor::new(vec![1.0], vec![1]),
            ],
        ];

        let batch = sequence_pad_collate(&samples);
        // No padding needed — all same length
        assert_eq!(batch[0].shape(), &[2, 2]);
        assert_eq!(batch[0].data(), &[1.0, 2.0, 3.0, 4.0]);
    }
}

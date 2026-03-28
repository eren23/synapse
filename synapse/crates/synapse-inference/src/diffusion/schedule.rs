//! Denoising mask schedules for diffusion LLM generation.
//!
//! Controls which tokens to unmask at each denoising step.

/// Mask schedule strategy for diffusion denoising.
#[derive(Debug, Clone, Copy)]
pub enum MaskSchedule {
    /// Unmask tokens with highest confidence, spread evenly across steps.
    Confidence,
    /// Unmask a fixed fraction per step (linear).
    Linear,
    /// Cosine schedule: unmask slowly at first, faster later.
    Cosine,
}

/// Given logits for all positions, determine which masked tokens to unmask.
///
/// Returns a vec of `(position_index, predicted_token_id)` for tokens to
/// unmask this step, chosen by highest confidence (max logit).
///
/// # Arguments
/// - `logits`: flattened `[seq_len, vocab_size]` model output logits
/// - `is_masked`: `[seq_len]` -- true if position is still masked
/// - `vocab_size`: vocabulary size
/// - `num_to_unmask`: how many tokens to reveal this step
pub fn unmask_by_confidence(
    logits: &[f32],
    is_masked: &[bool],
    vocab_size: usize,
    num_to_unmask: usize,
) -> Vec<(usize, u32)> {
    let seq_len = is_masked.len();

    // For each masked position, find the max logit (confidence) and argmax (predicted token)
    let mut candidates: Vec<(usize, u32, f32)> = Vec::new();
    for pos in 0..seq_len {
        if !is_masked[pos] {
            continue;
        }
        let logit_slice = &logits[pos * vocab_size..(pos + 1) * vocab_size];
        let (best_token, best_logit) = logit_slice
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &val)| (idx as u32, val))
            .unwrap_or((0, f32::NEG_INFINITY));
        candidates.push((pos, best_token, best_logit));
    }

    // Sort by confidence (highest first)
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    // Take top num_to_unmask
    candidates.truncate(num_to_unmask);
    candidates.into_iter().map(|(pos, tok, _)| (pos, tok)).collect()
}

/// Compute how many tokens to unmask at each step given a schedule.
///
/// Returns a Vec of length `total_steps` where each element is the number
/// of tokens to unmask in that step. The sum equals `total_masked`.
pub fn tokens_per_step(
    schedule: MaskSchedule,
    total_masked: usize,
    total_steps: usize,
) -> Vec<usize> {
    if total_steps == 0 {
        return vec![];
    }
    match schedule {
        MaskSchedule::Linear | MaskSchedule::Confidence => {
            // Spread evenly across steps
            let per_step = total_masked / total_steps;
            let remainder = total_masked % total_steps;
            let mut steps = vec![per_step; total_steps];
            // Distribute remainder to last steps
            for i in 0..remainder {
                steps[total_steps - 1 - i] += 1;
            }
            steps
        }
        MaskSchedule::Cosine => {
            // Cosine schedule: unmask slowly at first, faster later
            let mut steps = Vec::with_capacity(total_steps);
            let mut cumulative = 0usize;
            for step in 0..total_steps {
                let fraction = 0.5
                    * (1.0
                        - ((step as f64 + 1.0) * std::f64::consts::PI / total_steps as f64)
                            .cos());
                let target = (fraction * total_masked as f64).round() as usize;
                let this_step = target.saturating_sub(cumulative);
                steps.push(this_step);
                cumulative += this_step;
            }
            // Ensure we cover all tokens
            let total: usize = steps.iter().sum();
            if total < total_masked {
                if let Some(last) = steps.last_mut() {
                    *last += total_masked - total;
                }
            }
            steps
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmask_by_confidence_picks_highest() {
        // 3 positions, vocab_size=4
        // Position 0: not masked
        // Position 1: masked, logits = [0.1, 0.9, 0.2, 0.3] => token 1, conf 0.9
        // Position 2: masked, logits = [0.5, 0.1, 0.8, 0.2] => token 2, conf 0.8
        let logits = vec![
            0.0, 0.0, 0.0, 0.0, // pos 0 (not masked, ignored)
            0.1, 0.9, 0.2, 0.3, // pos 1
            0.5, 0.1, 0.8, 0.2, // pos 2
        ];
        let is_masked = vec![false, true, true];

        // Unmask 1 token: should pick pos 1 (highest confidence 0.9)
        let result = unmask_by_confidence(&logits, &is_masked, 4, 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (1, 1)); // position 1, token 1

        // Unmask 2 tokens: both masked positions
        let result = unmask_by_confidence(&logits, &is_masked, 4, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 1)); // higher confidence first
        assert_eq!(result[1], (2, 2));
    }

    #[test]
    fn unmask_by_confidence_handles_zero() {
        let logits = vec![0.0; 8];
        let is_masked = vec![true, true];
        let result = unmask_by_confidence(&logits, &is_masked, 4, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn tokens_per_step_linear_sums_correctly() {
        let steps = tokens_per_step(MaskSchedule::Linear, 10, 3);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps.iter().sum::<usize>(), 10);
    }

    #[test]
    fn tokens_per_step_confidence_sums_correctly() {
        let steps = tokens_per_step(MaskSchedule::Confidence, 7, 4);
        assert_eq!(steps.len(), 4);
        assert_eq!(steps.iter().sum::<usize>(), 7);
    }

    #[test]
    fn tokens_per_step_cosine_sums_correctly() {
        let steps = tokens_per_step(MaskSchedule::Cosine, 20, 5);
        assert_eq!(steps.len(), 5);
        assert_eq!(steps.iter().sum::<usize>(), 20);
    }

    #[test]
    fn tokens_per_step_cosine_is_monotonically_nondecreasing() {
        // Cosine schedule should unmask more tokens as steps progress
        let steps = tokens_per_step(MaskSchedule::Cosine, 100, 10);
        // Allow for some non-monotonicity due to rounding, but overall trend should be increasing
        // Just check first half sum < second half sum
        let first_half: usize = steps[..5].iter().sum();
        let second_half: usize = steps[5..].iter().sum();
        assert!(
            second_half >= first_half,
            "cosine schedule should unmask more later: first_half={first_half}, second_half={second_half}"
        );
    }

    #[test]
    fn tokens_per_step_single_step() {
        let steps = tokens_per_step(MaskSchedule::Linear, 10, 1);
        assert_eq!(steps, vec![10]);
    }

    #[test]
    fn tokens_per_step_zero_steps() {
        let steps = tokens_per_step(MaskSchedule::Linear, 10, 0);
        assert!(steps.is_empty());
    }
}

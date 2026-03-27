use rand::Rng;

/// Trait for token sampling strategies.
///
/// Implementations receive a mutable logits slice and return a sampled token index.
/// Samplers may modify the logits in-place (e.g., applying temperature, masking).
pub trait Sampler: Send + Sync {
    /// Sample a single token index from the logits distribution.
    ///
    /// `logits` is a mutable `[vocab_size]` slice that samplers may modify in-place.
    /// `rng` provides randomness for stochastic samplers.
    fn sample(&self, logits: &mut [f32], rng: &mut dyn RngAdapter) -> u32;

    fn name(&self) -> &str;
}

/// Trait adapter so we can pass `&mut dyn Rng`-like objects through trait objects.
pub trait RngAdapter {
    fn next_f32(&mut self) -> f32;
    fn next_u32_range(&mut self, range: std::ops::Range<u32>) -> u32;
}

/// Blanket implementation for any `rand::Rng`.
impl<R: Rng> RngAdapter for R {
    fn next_f32(&mut self) -> f32 {
        self.gen::<f32>()
    }
    fn next_u32_range(&mut self, range: std::ops::Range<u32>) -> u32 {
        self.gen_range(range)
    }
}

// ── Greedy ──────────────────────────────────────────────────────────

/// Argmax sampling: always picks the highest-logit token. Deterministic.
pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&self, logits: &mut [f32], _rng: &mut dyn RngAdapter) -> u32 {
        argmax(logits)
    }

    fn name(&self) -> &str {
        "greedy"
    }
}

// ── Temperature ─────────────────────────────────────────────────────

/// Divide logits by temperature before softmax sampling.
/// Temperature=0 falls back to greedy (argmax).
pub struct TemperatureSampler {
    pub temperature: f32,
}

impl Sampler for TemperatureSampler {
    fn sample(&self, logits: &mut [f32], rng: &mut dyn RngAdapter) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let inv_t = 1.0 / self.temperature;
        for l in logits.iter_mut() {
            *l *= inv_t;
        }
        softmax_sample(logits, rng)
    }

    fn name(&self) -> &str {
        "temperature"
    }
}

// ── Top-K ───────────────────────────────────────────────────────────

/// Keep only the top K logits, zero the rest, then sample.
/// K=1 is equivalent to greedy.
pub struct TopKSampler {
    pub k: usize,
}

impl Sampler for TopKSampler {
    fn sample(&self, logits: &mut [f32], rng: &mut dyn RngAdapter) -> u32 {
        if self.k <= 1 {
            return argmax(logits);
        }
        top_k_filter(logits, self.k);
        softmax_sample(logits, rng)
    }

    fn name(&self) -> &str {
        "top_k"
    }
}

// ── Top-P (Nucleus) ─────────────────────────────────────────────────

/// Sort by probability, keep tokens until cumulative probability >= p.
/// p=0.0 is equivalent to greedy (only the top token).
pub struct TopPSampler {
    pub p: f32,
}

impl Sampler for TopPSampler {
    fn sample(&self, logits: &mut [f32], rng: &mut dyn RngAdapter) -> u32 {
        if self.p <= 0.0 {
            return argmax(logits);
        }
        top_p_filter(logits, self.p);
        softmax_sample(logits, rng)
    }

    fn name(&self) -> &str {
        "top_p"
    }
}

// ── Repetition Penalty ──────────────────────────────────────────────

/// Penalizes already-generated tokens by dividing their logits.
///
/// For positive logits: `logit /= penalty`
/// For negative logits: `logit *= penalty`
///
/// This always moves the logit closer to zero (less likely).
pub struct RepetitionPenalty {
    pub penalty: f32,
    pub generated_tokens: Vec<u32>,
}

impl RepetitionPenalty {
    /// Apply the repetition penalty to logits in-place.
    pub fn apply(&self, logits: &mut [f32]) {
        if (self.penalty - 1.0).abs() < f32::EPSILON {
            return;
        }
        for &tok in &self.generated_tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
    }
}

// ── Combined Sampler ────────────────────────────────────────────────

/// Chains: repetition penalty → temperature → top-k → top-p → sample.
pub struct CombinedSampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl CombinedSampler {
    /// Sample with full pipeline, given the list of previously generated tokens.
    pub fn sample_with_history(
        &self,
        logits: &mut [f32],
        generated_tokens: &[u32],
        rng: &mut dyn RngAdapter,
    ) -> u32 {
        // 1. Repetition penalty
        if (self.repetition_penalty - 1.0).abs() > f32::EPSILON {
            let rp = RepetitionPenalty {
                penalty: self.repetition_penalty,
                generated_tokens: generated_tokens.to_vec(),
            };
            rp.apply(logits);
        }

        // 2. Temperature
        if self.temperature <= 0.0 {
            return argmax(logits);
        }
        let inv_t = 1.0 / self.temperature;
        for l in logits.iter_mut() {
            *l *= inv_t;
        }

        // 3. Top-K
        if self.top_k > 0 && self.top_k < logits.len() {
            if self.top_k == 1 {
                return argmax(logits);
            }
            top_k_filter(logits, self.top_k);
        }

        // 4. Top-P
        if self.top_p > 0.0 && self.top_p < 1.0 {
            top_p_filter(logits, self.top_p);
        }

        // 5. Sample
        softmax_sample(logits, rng)
    }
}

impl Sampler for CombinedSampler {
    fn sample(&self, logits: &mut [f32], rng: &mut dyn RngAdapter) -> u32 {
        // Without history context, skip repetition penalty
        self.sample_with_history(logits, &[], rng)
    }

    fn name(&self) -> &str {
        "combined"
    }
}

// ── Utilities ───────────────────────────────────────────────────────

/// Return the index of the maximum value.
pub fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// In-place softmax, then sample from the resulting probability distribution.
pub fn softmax_sample(logits: &mut [f32], rng: &mut dyn RngAdapter) -> u32 {
    softmax_inplace(logits);
    categorical_sample(logits, rng)
}

/// In-place softmax: subtract max for numerical stability, then exp and normalize.
pub fn softmax_inplace(logits: &mut [f32]) {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max_val).exp();
        sum += *l;
    }
    if sum > 0.0 {
        for l in logits.iter_mut() {
            *l /= sum;
        }
    }
}

/// Sample from a probability distribution using inverse CDF.
fn categorical_sample(probs: &[f32], rng: &mut dyn RngAdapter) -> u32 {
    let u = rng.next_f32();
    let mut cumsum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if u < cumsum {
            return i as u32;
        }
    }
    // Fallback to last token (rounding errors)
    (probs.len() - 1) as u32
}

/// Zero out all logits except the top-k highest.
fn top_k_filter(logits: &mut [f32], k: usize) {
    let k = k.min(logits.len());
    // Find the k-th largest value
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let threshold = logits[indices[k - 1]];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Zero out tokens beyond the nucleus (cumulative probability >= p).
fn top_p_filter(logits: &mut [f32], p: f32) {
    // Convert to probabilities
    let mut probs: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    // Softmax first to get probabilities
    let max_val = probs
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, v) in probs.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for (_, v) in probs.iter_mut() {
            *v /= sum;
        }
    }

    // Sort descending by probability
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff
    let mut cumsum = 0.0f32;
    let mut keep = std::collections::HashSet::new();
    for (idx, prob) in &probs {
        keep.insert(*idx);
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }

    // Mask out tokens not in the nucleus
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn greedy_is_deterministic_100_runs() {
        let sampler = GreedySampler;
        let base_logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let mut rng = make_rng();

        let mut results = Vec::new();
        for _ in 0..100 {
            let mut logits = base_logits.clone();
            results.push(sampler.sample(&mut logits, &mut rng));
        }
        assert!(
            results.iter().all(|&r| r == results[0]),
            "Greedy must be deterministic"
        );
        assert_eq!(results[0], 3); // index of 0.9
    }

    #[test]
    fn temperature_zero_equals_greedy() {
        let temp = TemperatureSampler { temperature: 0.0 };
        let greedy = GreedySampler;
        let base_logits = vec![1.0, 3.0, 2.0, 5.0, 0.5];
        let mut rng = make_rng();

        for _ in 0..50 {
            let mut l1 = base_logits.clone();
            let mut l2 = base_logits.clone();
            assert_eq!(
                temp.sample(&mut l1, &mut rng),
                greedy.sample(&mut l2, &mut rng),
            );
        }
    }

    #[test]
    fn top_k_1_equals_greedy() {
        let topk = TopKSampler { k: 1 };
        let greedy = GreedySampler;
        let base_logits = vec![1.0, 3.0, 2.0, 5.0, 0.5];
        let mut rng = make_rng();

        for _ in 0..50 {
            let mut l1 = base_logits.clone();
            let mut l2 = base_logits.clone();
            assert_eq!(
                topk.sample(&mut l1, &mut rng),
                greedy.sample(&mut l2, &mut rng),
            );
        }
    }

    #[test]
    fn top_p_zero_equals_greedy() {
        let topp = TopPSampler { p: 0.0 };
        let greedy = GreedySampler;
        let base_logits = vec![1.0, 3.0, 2.0, 5.0, 0.5];
        let mut rng = make_rng();

        for _ in 0..50 {
            let mut l1 = base_logits.clone();
            let mut l2 = base_logits.clone();
            assert_eq!(
                topp.sample(&mut l1, &mut rng),
                greedy.sample(&mut l2, &mut rng),
            );
        }
    }

    #[test]
    fn temperature_1_has_entropy() {
        let sampler = TemperatureSampler { temperature: 1.0 };
        // Logits that give roughly equal probabilities
        let base_logits = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let mut rng = make_rng();

        let mut counts = [0u32; 5];
        for _ in 0..1000 {
            let mut logits = base_logits.clone();
            let token = sampler.sample(&mut logits, &mut rng);
            counts[token as usize] += 1;
        }
        // With uniform logits, each should get ~200 samples. Check all > 50.
        for (i, &c) in counts.iter().enumerate() {
            assert!(
                c > 50,
                "Token {i} only sampled {c} times out of 1000, expected ~200"
            );
        }
    }

    #[test]
    fn repetition_penalty_lowers_repeated_probability() {
        let mut logits = vec![2.0, 3.0, 1.0, 4.0, 0.5];
        let original = logits.clone();

        let rp = RepetitionPenalty {
            penalty: 2.0,
            generated_tokens: vec![1, 3], // penalize tokens 1 and 3
        };
        rp.apply(&mut logits);

        // Positive logits should be divided by penalty
        assert!((logits[1] - original[1] / 2.0).abs() < f32::EPSILON);
        assert!((logits[3] - original[3] / 2.0).abs() < f32::EPSILON);
        // Non-penalized tokens unchanged
        assert_eq!(logits[0], original[0]);
        assert_eq!(logits[2], original[2]);
        assert_eq!(logits[4], original[4]);
    }

    #[test]
    fn repetition_penalty_negative_logits() {
        let mut logits = vec![-2.0, 3.0];
        let rp = RepetitionPenalty {
            penalty: 2.0,
            generated_tokens: vec![0], // penalize negative logit
        };
        rp.apply(&mut logits);
        // Negative logit should be multiplied (made more negative)
        assert!((logits[0] - (-4.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn softmax_produces_valid_distribution() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax_inplace(&mut logits);

        // All non-negative
        assert!(
            logits.iter().all(|&p| p >= 0.0),
            "Probabilities must be non-negative"
        );
        // Sum to 1
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities must sum to 1, got {sum}"
        );
    }

    #[test]
    fn combined_sampler_applies_all_stages() {
        let sampler = CombinedSampler {
            temperature: 1.0,
            top_k: 3,
            top_p: 0.9,
            repetition_penalty: 1.5,
        };
        let mut rng = make_rng();

        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 0.5];
        let generated = vec![1u32]; // penalize token 1 (the highest)
        let token = sampler.sample_with_history(&mut logits, &generated, &mut rng);
        assert!(token < 5);
    }

    #[test]
    fn top_k_filter_preserves_top_values() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        top_k_filter(&mut logits, 3);
        // Top 3 are indices 1(5.0), 4(4.0), 2(3.0)
        assert!(logits[1].is_finite());
        assert!(logits[4].is_finite());
        assert!(logits[2].is_finite());
        // The rest should be -inf
        assert!(logits[0] == f32::NEG_INFINITY);
        assert!(logits[3] == f32::NEG_INFINITY);
    }

    #[test]
    fn argmax_returns_correct_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 5.0]), 2);
    }
}

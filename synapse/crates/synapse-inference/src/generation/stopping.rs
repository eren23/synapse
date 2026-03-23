/// Conditions that terminate token generation.
pub enum StopCondition {
    /// Stop when a specific EOS token is generated.
    EosToken(u32),
    /// Stop after generating this many tokens (excludes prompt).
    MaxLength(usize),
    /// Stop when the generated token sequence contains any of these token-ID subsequences.
    StopSequences(Vec<Vec<u32>>),
}

/// Evaluates a set of stop conditions against the current generation state.
pub struct StopChecker {
    conditions: Vec<StopCondition>,
}

impl StopChecker {
    pub fn new(conditions: Vec<StopCondition>) -> Self {
        Self { conditions }
    }

    /// Returns `true` if any stop condition is met.
    ///
    /// - `last_token`: the most recently generated token
    /// - `generated_tokens`: all tokens generated so far (excluding prompt)
    /// - `num_generated`: count of generated tokens
    pub fn should_stop(
        &self,
        last_token: u32,
        generated_tokens: &[u32],
        num_generated: usize,
    ) -> bool {
        for cond in &self.conditions {
            match cond {
                StopCondition::EosToken(eos) => {
                    if last_token == *eos {
                        return true;
                    }
                }
                StopCondition::MaxLength(max) => {
                    if num_generated >= *max {
                        return true;
                    }
                }
                StopCondition::StopSequences(sequences) => {
                    for seq in sequences {
                        if generated_tokens.len() >= seq.len()
                            && &generated_tokens[generated_tokens.len() - seq.len()..] == &seq[..]
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eos_stops_generation() {
        let checker = StopChecker::new(vec![StopCondition::EosToken(2)]);
        assert!(!checker.should_stop(1, &[1], 1));
        assert!(checker.should_stop(2, &[1, 2], 2));
    }

    #[test]
    fn max_length_stops_generation() {
        let checker = StopChecker::new(vec![StopCondition::MaxLength(3)]);
        assert!(!checker.should_stop(1, &[1, 2], 2));
        assert!(checker.should_stop(3, &[1, 2, 3], 3));
        assert!(checker.should_stop(4, &[1, 2, 3, 4], 4));
    }

    #[test]
    fn stop_sequence_stops_generation() {
        let checker = StopChecker::new(vec![StopCondition::StopSequences(vec![vec![5, 6, 7]])]);
        assert!(!checker.should_stop(5, &[1, 2, 5], 3));
        assert!(!checker.should_stop(6, &[1, 2, 5, 6], 4));
        assert!(checker.should_stop(7, &[1, 2, 5, 6, 7], 5));
    }

    #[test]
    fn multiple_conditions_any_triggers() {
        let checker = StopChecker::new(vec![
            StopCondition::EosToken(99),
            StopCondition::MaxLength(5),
        ]);
        // EOS triggers first
        assert!(checker.should_stop(99, &[99], 1));
        // Max length triggers
        assert!(checker.should_stop(10, &[1, 2, 3, 4, 5], 5));
    }

    #[test]
    fn no_conditions_never_stops() {
        let checker = StopChecker::new(vec![]);
        assert!(!checker.should_stop(1, &[1, 2, 3], 3));
    }
}

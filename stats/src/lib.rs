//! Bayesian statistics for proportion inference.
//!
//! # Mathematical Foundations
//!
//! ## Mode 1: Single proportion credible interval
//!
//! Given n successes in m trials, we use a uniform (non-informative) prior:
//!   Prior: p ~ Beta(1, 1)  (equivalent to Uniform[0,1])
//!
//! The Beta distribution is conjugate to the Binomial likelihood. The posterior is:
//!   Posterior: p | data ~ Beta(n + 1, m - n + 1)
//!
//! Source: Gelman et al., "Bayesian Data Analysis" 3rd ed., Chapter 2.
//! Source: DeGroot & Schervish, "Probability and Statistics" 4th ed., Section 7.3.
//!
//! The posterior mean (point estimate) is (n+1)/(m+2).
//!
//! The 100*(1-α)% credible interval is:
//!   [Beta_inv(α/2; n+1, m-n+1), Beta_inv(1-α/2; n+1, m-n+1)]
//!
//! ## Mode 2: Two-sample same-p Bayesian model comparison
//!
//! Given (n1, m1) and (n2, m2), compare:
//!   H_same: both samples share a single p ~ Uniform[0,1]
//!   H_diff: independent p1 ~ Uniform[0,1], p2 ~ Uniform[0,1]
//!
//! Equal model priors: P(H_same) = P(H_diff) = 0.5
//!
//! ### Marginal likelihood under H_same
//!
//! P(data | H_same) = ∫₀¹ C(m1,n1) p^n1 (1-p)^(m1-n1) · C(m2,n2) p^n2 (1-p)^(m2-n2) dp
//!                  = C(m1,n1) C(m2,n2) · B(n1+n2+1, m1+m2-n1-n2+1)
//!
//! where B(a, b) = Γ(a)Γ(b)/Γ(a+b) is the Beta function.
//!
//! ### Marginal likelihood under H_diff
//!
//! P(data | H_diff) = C(m1,n1) B(n1+1, m1-n1+1) · C(m2,n2) B(n2+1, m2-n2+1)
//!
//! ### Posterior probability for H_same
//!
//! The binomial coefficients cancel in the ratio, giving:
//!
//!   P(H_same | data) = L_same / (L_same + L_diff)
//!
//! where:
//!   L_same = B(n1+n2+1, m1+m2-n1-n2+1)
//!   L_diff = B(n1+1, m1-n1+1) · B(n2+1, m2-n2+1)
//!
//! Source: MacKay, "Information Theory, Inference, and Learning Algorithms", Chapter 28.
//! Source: Kass & Raftery, "Bayes Factors", JASA 1995, 90(430):773-795.
//! Source: Lee, "Bayesian Statistics: An Introduction" 4th ed.
//!
//! ## Mode 3: Compare two trials (independent posteriors)
//!
//! Given (n1, m1) and (n2, m2), fit independent Beta posteriors:
//!   p1 | data ~ Beta(n1+1, m1-n1+1)
//!   p2 | data ~ Beta(n2+1, m2-n2+1)
//!
//! Closed-form results (Gelman et al. BDA 3rd ed., Ch. 2):
//!   E[p_i] = (n_i+1)/(m_i+2)       (posterior mean)
//!   E[p1 - p2] = E[p1] - E[p2]     (linearity of expectation)
//!
//! P(p1 > p2) via Monte Carlo (Robert & Casella, "Monte Carlo Statistical Methods"
//! 2nd ed., Ch. 3): draw N=100_000 paired samples (s1, s2) from the two posteriors,
//! estimate P(p1 > p2) ≈ #{s1 > s2} / N.
//!
//! 95% CI on the difference: sort the N sampled differences and take the 2.5th and
//! 97.5th percentiles.

use statrs::function::beta::ln_beta;
use statrs::distribution::Beta as StatrsBeta;
use thiserror::Error;
use rand::distributions::Distribution;

/// Errors from invalid inputs.
#[derive(Debug, Error, PartialEq)]
pub enum StatsError {
    #[error("trials m must be > 0, got {0}")]
    ZeroTrials(u64),
    #[error("successes n ({n}) must not exceed trials m ({m})")]
    ExceedsTrials { n: u64, m: u64 },
    #[error("confidence level must be in (0, 1), got {0}")]
    InvalidConfidence(f64),
    #[error("Beta function evaluation failed: {0}")]
    BetaError(String),
    #[error("invalid rate: {0}")]
    InvalidRate(String),
}

/// Result of a single-proportion credible interval computation.
#[derive(Debug, Clone, PartialEq)]
pub struct CredibleInterval {
    /// Posterior mean = (n+1)/(m+2). This is NOT the MLE (n/m).
    pub point_estimate: f64,
    /// Lower bound of the credible interval.
    pub lower: f64,
    /// Upper bound of the credible interval.
    pub upper: f64,
    /// The confidence level used (e.g. 0.95 for 95%).
    pub confidence: f64,
    /// Beta posterior parameters (a, b) = (n+1, m-n+1).
    pub alpha: f64,
    pub beta_param: f64,
}

/// Result of a two-sample same-p comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct SamePResult {
    /// Posterior probability that both samples share a single p.
    pub prob_same: f64,
    /// Posterior probability that the samples come from different ps.
    pub prob_diff: f64,
    /// The log Bayes factor: ln(L_same / L_diff).
    /// Positive favours H_same, negative favours H_diff.
    pub log_bayes_factor: f64,
}

/// Result of a two-trial comparison using independent Beta posteriors.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareTwoTrialsResult {
    /// Posterior mean for trial 1: (n1+1)/(m1+2).
    pub mean1: f64,
    /// Posterior mean for trial 2: (n2+1)/(m2+2).
    pub mean2: f64,
    /// Expected difference: E[p1 - p2] = mean1 - mean2 (closed form).
    pub mean_diff: f64,
    /// P(p1 > p2) estimated via Monte Carlo (N=100_000 samples).
    pub prob_p1_gt_p2: f64,
    /// Lower bound of 95% credible interval on p1 - p2 (MC percentile).
    pub ci_diff_lower: f64,
    /// Upper bound of 95% credible interval on p1 - p2 (MC percentile).
    pub ci_diff_upper: f64,
}

/// Parse a rate string into a probability in [0, 1].
///
/// Accepted formats:
/// - Decimal in [0, 1]: `"0.72"` → 0.72
/// - Percentage with suffix: `"72%"` or `"71.5%"` → divide by 100
/// - Bare number > 1: `"72"` → treated as percentage, divide by 100
///
/// Rounding (used by callers to compute n = round(rate * m)) follows
/// `f64::round()`, which is round-half-away-from-zero.
pub fn parse_rate(s: &str) -> Result<f64, StatsError> {
    let s = s.trim();
    let (num_str, is_pct) = if let Some(stripped) = s.strip_suffix('%') {
        (stripped.trim(), true)
    } else {
        (s, false)
    };

    let val: f64 = num_str
        .parse()
        .map_err(|_| StatsError::InvalidRate(format!("cannot parse {s:?} as a number")))?;

    let rate = if is_pct || val > 1.0 {
        val / 100.0
    } else {
        val
    };

    if !(0.0..=1.0).contains(&rate) {
        return Err(StatsError::InvalidRate(format!(
            "rate must be in [0, 1] (or [0, 100] as percentage), got {s:?}"
        )));
    }
    Ok(rate)
}

/// Compute the Bayesian credible interval for a proportion.
///
/// # Arguments
/// - `n`: number of successes (0 ≤ n ≤ m)
/// - `m`: number of trials (m ≥ 1)
/// - `confidence`: credible mass, e.g. 0.95 for 95% (must be in (0, 1))
///
/// # Method
/// Prior: Beta(1,1) (uniform). Posterior: Beta(n+1, m-n+1).
/// Credible interval: equal-tailed, using quantiles at (1-confidence)/2 and 1-(1-confidence)/2.
pub fn credible_interval(n: u64, m: u64, confidence: f64) -> Result<CredibleInterval, StatsError> {
    if m == 0 {
        return Err(StatsError::ZeroTrials(m));
    }
    if n > m {
        return Err(StatsError::ExceedsTrials { n, m });
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(StatsError::InvalidConfidence(confidence));
    }

    // Posterior parameters: Beta(a, b)
    let a = (n as f64) + 1.0;
    let b = (m - n) as f64 + 1.0;

    let point_estimate = a / (a + b); // posterior mean

    let alpha_tail = (1.0 - confidence) / 2.0;

    let lower = beta_quantile(alpha_tail, a, b)?;
    let upper = beta_quantile(1.0 - alpha_tail, a, b)?;

    Ok(CredibleInterval {
        point_estimate,
        lower,
        upper,
        confidence,
        alpha: a,
        beta_param: b,
    })
}

/// Compute the posterior probability that two samples share a single p.
///
/// # Arguments
/// - `n1`, `m1`: successes and trials in sample 1
/// - `n2`, `m2`: successes and trials in sample 2
///
/// # Method
/// Bayesian model comparison with equal priors P(H_same) = P(H_diff) = 0.5.
/// Marginal likelihoods computed via Beta function integrals (analytic).
/// Binomial coefficients cancel in the likelihood ratio.
pub fn same_p_probability(
    n1: u64,
    m1: u64,
    n2: u64,
    m2: u64,
) -> Result<SamePResult, StatsError> {
    if m1 == 0 {
        return Err(StatsError::ZeroTrials(m1));
    }
    if m2 == 0 {
        return Err(StatsError::ZeroTrials(m2));
    }
    if n1 > m1 {
        return Err(StatsError::ExceedsTrials { n: n1, m: m1 });
    }
    if n2 > m2 {
        return Err(StatsError::ExceedsTrials { n: n2, m: m2 });
    }

    // Log marginal likelihood under H_same:
    // ln L_same = ln B(n1+n2+1, m1+m2-n1-n2+1)
    let ln_l_same = ln_beta(
        (n1 + n2) as f64 + 1.0,
        (m1 + m2 - n1 - n2) as f64 + 1.0,
    );

    // Log marginal likelihood under H_diff:
    // ln L_diff = ln B(n1+1, m1-n1+1) + ln B(n2+1, m2-n2+1)
    let ln_l_diff = ln_beta((n1 as f64) + 1.0, (m1 - n1) as f64 + 1.0)
        + ln_beta((n2 as f64) + 1.0, (m2 - n2) as f64 + 1.0);

    let log_bayes_factor = ln_l_same - ln_l_diff;

    // Posterior probabilities (equal priors cancel):
    //   P(H_same | data) = L_same / (L_same + L_diff)
    // In log-space to avoid overflow:
    //   = exp(ln_l_same) / (exp(ln_l_same) + exp(ln_l_diff))
    //   = 1 / (1 + exp(ln_l_diff - ln_l_same))
    let prob_same = if log_bayes_factor > 500.0 {
        // L_same >> L_diff
        1.0
    } else if log_bayes_factor < -500.0 {
        // L_same << L_diff
        0.0
    } else {
        1.0 / (1.0 + (ln_l_diff - ln_l_same).exp())
    };

    let prob_diff = 1.0 - prob_same;

    Ok(SamePResult {
        prob_same,
        prob_diff,
        log_bayes_factor,
    })
}

/// Compare two independent trials using separate Beta posteriors.
///
/// # Arguments
/// - `n1`, `m1`: successes and trials in trial 1
/// - `n2`, `m2`: successes and trials in trial 2
///
/// # Method
/// Independent uniform priors → Beta(n+1, m-n+1) posteriors.
/// Closed-form means and mean difference (Gelman et al. BDA 3rd ed., Ch. 2).
/// P(p1 > p2) via Monte Carlo: N=100_000 paired Beta samples; count s1 > s2.
/// 95% CI on p1-p2 via MC percentiles (2.5th and 97.5th).
/// Source: Robert & Casella, "Monte Carlo Statistical Methods" 2nd ed., Ch. 3.
pub fn compare_two_trials(
    n1: u64,
    m1: u64,
    n2: u64,
    m2: u64,
) -> Result<CompareTwoTrialsResult, StatsError> {
    if m1 == 0 {
        return Err(StatsError::ZeroTrials(m1));
    }
    if m2 == 0 {
        return Err(StatsError::ZeroTrials(m2));
    }
    if n1 > m1 {
        return Err(StatsError::ExceedsTrials { n: n1, m: m1 });
    }
    if n2 > m2 {
        return Err(StatsError::ExceedsTrials { n: n2, m: m2 });
    }

    let a1 = n1 as f64 + 1.0;
    let b1 = (m1 - n1) as f64 + 1.0;
    let a2 = n2 as f64 + 1.0;
    let b2 = (m2 - n2) as f64 + 1.0;

    let mean1 = a1 / (a1 + b1);
    let mean2 = a2 / (a2 + b2);
    let mean_diff = mean1 - mean2;

    let beta1 = StatrsBeta::new(a1, b1).map_err(|e| StatsError::BetaError(e.to_string()))?;
    let beta2 = StatrsBeta::new(a2, b2).map_err(|e| StatsError::BetaError(e.to_string()))?;
    let mut rng = rand::thread_rng();

    const N: usize = 100_000;
    let mut count_gt = 0usize;
    let mut diffs = Vec::with_capacity(N);

    for _ in 0..N {
        let s1 = beta1.sample(&mut rng);
        let s2 = beta2.sample(&mut rng);
        let d = s1 - s2;
        diffs.push(d);
        if s1 > s2 {
            count_gt += 1;
        }
    }

    let prob_p1_gt_p2 = count_gt as f64 / N as f64;

    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo_idx = ((0.025 * N as f64) as usize).min(N - 1);
    let hi_idx = ((0.975 * N as f64) as usize).min(N - 1);
    let ci_diff_lower = diffs[lo_idx];
    let ci_diff_upper = diffs[hi_idx];

    Ok(CompareTwoTrialsResult {
        mean1,
        mean2,
        mean_diff,
        prob_p1_gt_p2,
        ci_diff_lower,
        ci_diff_upper,
    })
}

/// Compute the quantile (inverse CDF) of the Beta(a, b) distribution at probability p.
///
/// Uses bisection on the CDF (from statrs) for reliable convergence and accuracy.
/// The bisection terminates when the interval is narrower than 1e-12.
///
/// See Press et al., "Numerical Recipes" §6.4 for context on Beta quantile algorithms.
fn beta_quantile(p: f64, a: f64, b: f64) -> Result<f64, StatsError> {
    use statrs::distribution::{Beta, ContinuousCDF};

    let dist = Beta::new(a, b).map_err(|e| StatsError::BetaError(e.to_string()))?;

    // Bisection: find x in [0,1] such that CDF(x) == p.
    // 64 iterations gives precision ~2^{-64} ≈ 5e-20, far beyond f64's ~1e-15.
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if dist.cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn assert_approx(a: f64, b: f64, tol: f64, msg: &str) {
        assert!(
            (a - b).abs() < tol,
            "{msg}: got {a}, expected {b}, diff = {}",
            (a - b).abs()
        );
    }

    // ── Mode 1: credible interval ───────────────────────────────────────────

    /// Point estimate should be posterior mean = (n+1)/(m+2).
    #[test]
    fn test_point_estimate() {
        let ci = credible_interval(3, 10, 0.95).unwrap();
        assert_approx(ci.point_estimate, 4.0 / 12.0, EPS, "point estimate");
    }

    /// Verify symmetry: n successes and (m-n) failures should mirror each other.
    #[test]
    fn test_credible_interval_symmetry() {
        let ci_n = credible_interval(3, 10, 0.95).unwrap();
        let ci_flip = credible_interval(7, 10, 0.95).unwrap();
        assert_approx(
            ci_n.lower,
            1.0 - ci_flip.upper,
            EPS,
            "symmetric lower",
        );
        assert_approx(
            ci_n.upper,
            1.0 - ci_flip.lower,
            EPS,
            "symmetric upper",
        );
    }

    /// The credible interval must contain the point estimate.
    #[test]
    fn test_interval_contains_point_estimate() {
        for &(n, m) in &[(0, 1), (1, 1), (0, 10), (5, 10), (10, 10), (3, 7)] {
            let ci = credible_interval(n, m, 0.95).unwrap();
            assert!(
                ci.lower <= ci.point_estimate && ci.point_estimate <= ci.upper,
                "n={n}, m={m}: [{lower}, {upper}] does not contain {pe}",
                lower = ci.lower,
                upper = ci.upper,
                pe = ci.point_estimate,
            );
        }
    }

    /// Interval bounds must be in [0, 1].
    #[test]
    fn test_interval_bounds_in_unit() {
        for &(n, m) in &[(0, 1), (1, 1), (0, 100), (100, 100), (50, 100)] {
            let ci = credible_interval(n, m, 0.95).unwrap();
            assert!(
                ci.lower >= 0.0 && ci.upper <= 1.0,
                "n={n}, m={m}: bounds out of [0,1]"
            );
        }
    }

    /// Edge case: zero successes.
    #[test]
    fn test_zero_successes() {
        let ci = credible_interval(0, 10, 0.95).unwrap();
        assert!(ci.lower >= 0.0);
        assert!(ci.upper > 0.0);
        assert!(ci.upper < 0.5, "upper={}", ci.upper);
    }

    /// Edge case: all successes.
    #[test]
    fn test_all_successes() {
        let ci = credible_interval(10, 10, 0.95).unwrap();
        assert!(ci.upper <= 1.0);
        assert!(ci.lower < 1.0);
        assert!(ci.lower > 0.5, "lower={}", ci.lower);
    }

    /// Edge case: single trial.
    #[test]
    fn test_single_trial() {
        let ci0 = credible_interval(0, 1, 0.95).unwrap();
        let ci1 = credible_interval(1, 1, 0.95).unwrap();
        assert!(ci0.point_estimate < 0.5);
        assert!(ci1.point_estimate > 0.5);
    }

    /// Wider confidence → wider interval.
    #[test]
    fn test_wider_confidence_wider_interval() {
        let ci90 = credible_interval(5, 20, 0.90).unwrap();
        let ci95 = credible_interval(5, 20, 0.95).unwrap();
        let ci99 = credible_interval(5, 20, 0.99).unwrap();
        assert!(
            ci90.lower > ci95.lower && ci95.lower > ci99.lower,
            "lower bounds should decrease with wider confidence"
        );
        assert!(
            ci90.upper < ci95.upper && ci95.upper < ci99.upper,
            "upper bounds should increase with wider confidence"
        );
    }

    /// Known value: Beta(6, 6) 95% CI.
    /// n=5, m=10 → posterior Beta(6, 6).
    /// Mean = 6/12 = 0.5.
    ///
    /// Hand-verified by computing I_x(6,6) via the binomial sum formula:
    ///   I_x(6,6) = sum_{k=6}^{11} C(11,k) x^k (1-x)^(11-k)
    /// At x=0.234, this sum ≈ 0.02512 (close to 0.025), confirming the 2.5th
    /// percentile is near 0.234. Statrs bisection converges to ≈ 0.23379.
    /// By symmetry the 97.5th percentile is 1 − 0.23379 ≈ 0.76621.
    #[test]
    fn test_known_value_symmetric() {
        let ci = credible_interval(5, 10, 0.95).unwrap();
        assert_approx(ci.point_estimate, 0.5, EPS, "mean");
        // 2.5th and 97.5th percentiles of Beta(6,6), hand-verified via binomial sum
        assert_approx(ci.lower, 0.23379, 5e-4, "lower 95% CI for Beta(6,6)");
        assert_approx(ci.upper, 0.76621, 5e-4, "upper 95% CI for Beta(6,6)");
        // Verify the CDF at the bounds is ≈ 0.025 / 0.975
        assert!(ci.lower > 0.22 && ci.lower < 0.25, "lower in expected range");
        assert!(ci.upper > 0.75 && ci.upper < 0.78, "upper in expected range");
    }

    /// Known value: Beta(2, 11) 95% CI.
    /// n=1, m=11 → posterior Beta(2, 11).
    /// Mean = 2/13 ≈ 0.15385.
    ///
    /// Hand-verified: I_x(2,11) = 1 - (1-x)^12 - 12x(1-x)^11.
    /// At x=0.02086: (1-x)^12 ≈ 0.7766, 12*x*(1-x)^11 ≈ 0.1986, sum ≈ 0.9752,
    /// so CDF ≈ 0.0248 ≈ 0.025. ✓
    #[test]
    fn test_known_value_skewed() {
        let ci = credible_interval(1, 11, 0.95).unwrap();
        assert_approx(ci.point_estimate, 2.0 / 13.0, EPS, "mean");
        // 2.5th percentile of Beta(2,11), hand-verified via CDF formula
        assert_approx(ci.lower, 0.02086, 1e-4, "lower 95% CI for Beta(2,11)");
        assert!(ci.lower > 0.01 && ci.lower < 0.04, "lower in expected range");
        assert!(ci.upper > 0.35 && ci.upper < 0.55, "upper in expected range");
    }

    /// Large m: interval should be narrow.
    #[test]
    fn test_large_m_narrow_interval() {
        let ci = credible_interval(500, 1000, 0.95).unwrap();
        let width = ci.upper - ci.lower;
        assert!(width < 0.065, "width={width}");
        assert_approx(ci.point_estimate, 501.0 / 1002.0, EPS, "mean");
    }

    // Error cases
    #[test]
    fn test_zero_trials_error() {
        assert_eq!(
            credible_interval(0, 0, 0.95),
            Err(StatsError::ZeroTrials(0))
        );
    }

    #[test]
    fn test_exceeds_trials_error() {
        assert_eq!(
            credible_interval(6, 5, 0.95),
            Err(StatsError::ExceedsTrials { n: 6, m: 5 })
        );
    }

    #[test]
    fn test_invalid_confidence_error() {
        assert_eq!(
            credible_interval(5, 10, 0.0),
            Err(StatsError::InvalidConfidence(0.0))
        );
        assert_eq!(
            credible_interval(5, 10, 1.0),
            Err(StatsError::InvalidConfidence(1.0))
        );
    }

    // ── Mode 2: same-p comparison ──────────────────────────────────────────

    /// Identical data → strong evidence for same p.
    #[test]
    fn test_identical_data_favours_same() {
        let r = same_p_probability(5, 10, 5, 10).unwrap();
        assert!(
            r.prob_same > 0.5,
            "identical data should favour H_same, got {prob}",
            prob = r.prob_same
        );
    }

    /// Very different proportions → strong evidence for different ps.
    #[test]
    fn test_very_different_proportions_favour_diff() {
        // Sample 1: 0/100, sample 2: 100/100
        let r = same_p_probability(0, 100, 100, 100).unwrap();
        assert!(
            r.prob_same < 0.01,
            "wildly different data should strongly favour H_diff, got {prob}",
            prob = r.prob_same
        );
    }

    /// prob_same + prob_diff = 1.
    #[test]
    fn test_probabilities_sum_to_one() {
        for &(n1, m1, n2, m2) in &[
            (3, 10, 7, 10),
            (0, 5, 5, 5),
            (100, 200, 50, 100),
            (1, 1, 0, 1),
        ] {
            let r = same_p_probability(n1, m1, n2, m2).unwrap();
            assert_approx(r.prob_same + r.prob_diff, 1.0, EPS, "sum = 1");
        }
    }

    /// Probabilities in [0, 1].
    #[test]
    fn test_probabilities_in_unit_interval() {
        for &(n1, m1, n2, m2) in &[
            (0, 1, 0, 1),
            (1, 1, 1, 1),
            (0, 100, 100, 100),
            (50, 100, 50, 100),
        ] {
            let r = same_p_probability(n1, m1, n2, m2).unwrap();
            assert!(
                (0.0..=1.0).contains(&r.prob_same),
                "prob_same out of [0,1]: {prob}",
                prob = r.prob_same
            );
            assert!(
                (0.0..=1.0).contains(&r.prob_diff),
                "prob_diff out of [0,1]: {prob}",
                prob = r.prob_diff
            );
        }
    }

    /// Symmetry: swapping samples should give the same result.
    #[test]
    fn test_same_p_symmetry() {
        let r1 = same_p_probability(3, 10, 7, 20).unwrap();
        let r2 = same_p_probability(7, 20, 3, 10).unwrap();
        assert_approx(r1.prob_same, r2.prob_same, EPS, "symmetric prob_same");
    }

    /// Known value: computed for n1=n2=5, m1=m2=10.
    ///
    /// L_same = B(n1+n2+1, m1+m2-n1-n2+1) = B(11, 11)
    /// L_diff = B(n1+1, m1-n1+1) * B(n2+1, m2-n2+1) = B(6,6)^2
    ///
    /// ln_beta(11, 11) = lgamma(11) + lgamma(11) - lgamma(22) ≈ -15.17131
    /// ln_beta(6, 6)   = lgamma(6)  + lgamma(6)  - lgamma(12) ≈  -7.92732
    /// ln_l_diff = 2 * -7.92732 ≈ -15.85465
    ///
    /// log_bf = -15.17131 - (-15.85465) ≈ 0.68333
    /// prob_same = 1 / (1 + exp(-0.68333)) ≈ 0.66448
    ///
    /// (Cross-checked with Python math.lgamma)
    #[test]
    fn test_known_value_same_p_equal_proportions() {
        let r = same_p_probability(5, 10, 5, 10).unwrap();
        assert_approx(r.log_bayes_factor, 0.68333497, 1e-6, "log BF");
        assert_approx(r.prob_same, 0.66448262, 1e-6, "prob_same");
    }

    /// Edge case: n=0 for both samples.
    #[test]
    fn test_both_zero_successes() {
        let r = same_p_probability(0, 10, 0, 10).unwrap();
        // Both 0/10 → strong evidence for same p
        assert!(r.prob_same > 0.5, "prob_same={}", r.prob_same);
    }

    /// Error: zero trials.
    #[test]
    fn test_same_p_zero_trials_error() {
        assert!(same_p_probability(0, 0, 5, 10).is_err());
        assert!(same_p_probability(5, 10, 0, 0).is_err());
    }

    /// Error: exceeds trials.
    #[test]
    fn test_same_p_exceeds_trials_error() {
        assert!(same_p_probability(11, 10, 5, 10).is_err());
        assert!(same_p_probability(5, 10, 11, 10).is_err());
    }

    // ── parse_rate ─────────────────────────────────────────────────────────

    /// Decimal form: "0.72" → 0.72.
    #[test]
    fn test_parse_rate_decimal() {
        assert_approx(parse_rate("0.72").unwrap(), 0.72, EPS, "0.72");
    }

    /// Bare percentage number: "72" → 0.72.
    #[test]
    fn test_parse_rate_bare_number() {
        assert_approx(parse_rate("72").unwrap(), 0.72, EPS, "72");
    }

    /// Explicit percent suffix: "72%" → 0.72.
    #[test]
    fn test_parse_rate_percent_suffix() {
        assert_approx(parse_rate("72%").unwrap(), 0.72, EPS, "72%");
    }

    /// Fractional percentage: "71.5%" rounds to 0.715.
    #[test]
    fn test_parse_rate_fractional_percent() {
        assert_approx(parse_rate("71.5%").unwrap(), 0.715, EPS, "71.5%");
    }

    /// Zero rate: "0%" → 0.0.
    #[test]
    fn test_parse_rate_zero_percent() {
        assert_approx(parse_rate("0%").unwrap(), 0.0, EPS, "0%");
        assert_approx(parse_rate("0").unwrap(), 0.0, EPS, "0");
    }

    /// Full rate: "100%" → 1.0.
    #[test]
    fn test_parse_rate_hundred_percent() {
        assert_approx(parse_rate("100%").unwrap(), 1.0, EPS, "100%");
        assert_approx(parse_rate("100").unwrap(), 1.0, EPS, "100");
        assert_approx(parse_rate("1.0").unwrap(), 1.0, EPS, "1.0");
    }

    /// Whitespace is trimmed.
    #[test]
    fn test_parse_rate_whitespace() {
        assert_approx(parse_rate("  72%  ").unwrap(), 0.72, EPS, "trimmed");
    }

    /// Bad input: cannot be parsed as number.
    #[test]
    fn test_parse_rate_bad_input() {
        assert!(parse_rate("abc").is_err());
        assert!(parse_rate("").is_err());
        assert!(parse_rate("%").is_err());
    }

    /// Out-of-range input.
    #[test]
    fn test_parse_rate_out_of_range() {
        assert!(parse_rate("101%").is_err());
        assert!(parse_rate("-1%").is_err());
        assert!(parse_rate("-0.1").is_err());
    }

    // ── n = round(rate * m) rounding ────────────────────────────────────────

    /// Verify round-half-away-from-zero: 0.72 * 100 = 72.0 exactly.
    #[test]
    fn test_rate_to_n_exact() {
        let rate = parse_rate("0.72").unwrap();
        let n = (rate * 100_f64).round() as u64;
        assert_eq!(n, 72);
    }

    /// 71.5% * 100 = 71.5 → rounds to 72 (round-half-away-from-zero).
    #[test]
    fn test_rate_to_n_half_up() {
        let rate = parse_rate("71.5%").unwrap();
        let n = (rate * 100_f64).round() as u64;
        assert_eq!(n, 72, "71.5% of 100 should round to 72");
    }

    // ── compare_two_trials ─────────────────────────────────────────────────

    /// Closed-form mean difference: hand-compute and verify.
    /// n1=5, m1=10: mean1 = 6/12 = 0.5
    /// n2=3, m2=10: mean2 = 4/12 = 1/3
    /// mean_diff = 0.5 - 1/3 = 1/6 ≈ 0.16667
    #[test]
    fn test_compare_mean_diff_closed_form() {
        let r = compare_two_trials(5, 10, 3, 10).unwrap();
        assert_approx(r.mean1, 6.0 / 12.0, EPS, "mean1");
        assert_approx(r.mean2, 4.0 / 12.0, EPS, "mean2");
        assert_approx(r.mean_diff, 1.0 / 6.0, EPS, "mean_diff");
    }

    /// When both trials are identical, mean_diff should be 0.
    #[test]
    fn test_compare_identical_trials_mean_diff_zero() {
        let r = compare_two_trials(5, 10, 5, 10).unwrap();
        assert_approx(r.mean_diff, 0.0, EPS, "mean_diff == 0 for identical trials");
    }

    /// Symmetry: P(p1 > p2) for identical posteriors ≈ 0.5 (within MC error).
    /// Both trials: n=10, m=10 → Beta(11, 1), highly concentrated near 1.0.
    /// Identical → P(p1 > p2) = 0.5 by symmetry.
    #[test]
    fn test_compare_prob_symmetry_identical() {
        let r = compare_two_trials(10, 10, 10, 10).unwrap();
        // Allow ±0.02 for Monte Carlo variance at N=100_000
        assert!(
            (r.prob_p1_gt_p2 - 0.5).abs() < 0.02,
            "P(p1>p2) for identical trials should be ~0.5, got {}",
            r.prob_p1_gt_p2
        );
    }

    /// P(p1 > p2) near 1.0 when trial 1 is all successes and trial 2 is all failures.
    /// n1=10,m1=10 → Beta(11,1) ≈ concentrated near 1.
    /// n2=0,m2=10 → Beta(1,11) ≈ concentrated near 0.
    #[test]
    fn test_compare_prob_extreme_near_one() {
        let r = compare_two_trials(10, 10, 0, 10).unwrap();
        assert!(
            r.prob_p1_gt_p2 > 0.99,
            "P(p1>p2) should be very close to 1.0, got {}",
            r.prob_p1_gt_p2
        );
    }

    /// P(p1 > p2) and P(p2 > p1) should sum to ~1.0 (ignoring P(p1=p2)≈0).
    #[test]
    fn test_compare_prob_complement() {
        let r1 = compare_two_trials(7, 10, 3, 10).unwrap();
        let r2 = compare_two_trials(3, 10, 7, 10).unwrap();
        // P(p1>p2) + P(p2>p1) ≈ 1 (ties are measure-zero for continuous distributions)
        assert!(
            (r1.prob_p1_gt_p2 + r2.prob_p1_gt_p2 - 1.0).abs() < 0.01,
            "complementary probs should sum to ~1"
        );
    }

    /// Error cases.
    #[test]
    fn test_compare_zero_trials_error() {
        assert!(compare_two_trials(0, 0, 5, 10).is_err());
        assert!(compare_two_trials(5, 10, 0, 0).is_err());
    }

    #[test]
    fn test_compare_exceeds_trials_error() {
        assert!(compare_two_trials(11, 10, 5, 10).is_err());
        assert!(compare_two_trials(5, 10, 11, 10).is_err());
    }
}

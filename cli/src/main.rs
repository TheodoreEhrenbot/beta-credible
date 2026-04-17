use beta_credible_stats::{credible_interval, hierarchical_bayes_compare, parse_rate, same_p_probability};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "beta-credible",
    about = "Bayesian credible intervals and same-p comparison for proportions",
    long_about = "
beta-credible computes Bayesian posterior statistics for proportions using a
uniform (Beta(1,1)) prior conjugate to the Binomial likelihood.

Mode 1 (interval): given n successes in m trials, reports the posterior mean
and a credible interval at the requested confidence level.

Mode 2 (same-p): given two samples (n1/m1) and (n2/m2), reports the posterior
probability that they share a single underlying proportion p, versus coming from
two different proportions.
"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute a Bayesian credible interval for a proportion.
    ///
    /// Prior: Uniform = Beta(1,1). Posterior: Beta(n+1, m-n+1).
    Interval {
        /// Number of successes (0 ≤ n ≤ m)
        #[arg(short = 'n', long)]
        successes: u64,

        /// Number of trials (m ≥ 1)
        #[arg(short = 'm', long)]
        trials: u64,

        /// Credible mass (e.g. 0.95 for 95%). Default: 0.95.
        #[arg(short, long, default_value = "0.95")]
        confidence: f64,
    },
    /// Compute the posterior probability that two samples share a single p.
    ///
    /// Prior: P(H_same) = P(H_diff) = 0.5, each p ~ Uniform[0,1].
    SameP {
        /// Successes in sample 1
        #[arg(long)]
        n1: u64,

        /// Trials in sample 1
        #[arg(long)]
        m1: u64,

        /// Successes in sample 2
        #[arg(long)]
        n2: u64,

        /// Trials in sample 2
        #[arg(long)]
        m2: u64,
    },

    /// Hierarchical Bayesian comparison: one control vs K intervention arms.
    ///
    /// Model: θ_control ~ Beta(1,1); μ ~ Uniform(0,1), κ ~ Exponential(1);
    /// θ_k | μ,κ ~ Beta(μκ,(1−μ)κ); likelihoods Binomial.
    /// Inference via Metropolis-within-Gibbs MCMC.
    Hierarchical {
        /// Control successes (s_0)
        #[arg(long)]
        s0: u64,

        /// Control trials (n_0)
        #[arg(long)]
        n0: u64,

        /// Intervention successes (repeat once per arm, e.g. --sk 30 --sk 45)
        #[arg(long = "sk", required = true)]
        sk: Vec<u64>,

        /// Intervention trials (repeat once per arm, must match --sk count)
        #[arg(long = "nk", required = true)]
        nk: Vec<u64>,

        /// Intervention success rates as percentages (alternative to --sk/--nk;
        /// use with --mk; e.g. --rate 30% --rate 45%)
        #[arg(long = "rate", conflicts_with_all = ["sk", "nk"])]
        rate: Vec<String>,

        /// Intervention trial counts when using --rate (repeat once per arm)
        #[arg(long = "mk", conflicts_with_all = ["sk", "nk"])]
        mk: Vec<u64>,

        /// Total MCMC iterations [default: 50000]
        #[arg(long, default_value = "50000")]
        iterations: usize,

        /// Burn-in iterations to discard [default: 10000]
        #[arg(long, default_value = "10000")]
        burnin: usize,

        /// Thinning factor [default: 4]
        #[arg(long, default_value = "4")]
        thin: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Interval {
            successes,
            trials,
            confidence,
        } => {
            match credible_interval(successes, trials, confidence) {
                Ok(ci) => {
                    let pct = (confidence * 100.0).round() as u64;
                    println!("Bayesian credible interval (uniform prior)");
                    println!("  Input:          {successes}/{trials} successes");
                    println!("  Posterior:      Beta({a}, {b})", a = ci.alpha, b = ci.beta_param);
                    println!("  Point estimate: {pe:.6}  (posterior mean)", pe = ci.point_estimate);
                    println!(
                        "  {pct}% CI:        [{lower:.6}, {upper:.6}]",
                        lower = ci.lower,
                        upper = ci.upper
                    );
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
        Commands::SameP { n1, m1, n2, m2 } => {
            match same_p_probability(n1, m1, n2, m2) {
                Ok(r) => {
                    println!("Bayesian same-p comparison (uniform priors, equal model priors)");
                    println!("  Sample 1:        {n1}/{m1} successes");
                    println!("  Sample 2:        {n2}/{m2} successes");
                    println!("  P(same p | data): {ps:.6}", ps = r.prob_same);
                    println!("  P(diff p | data): {pd:.6}", pd = r.prob_diff);
                    println!("  Log Bayes factor (same/diff): {lbf:.4}", lbf = r.log_bayes_factor);
                    let interpretation = if r.log_bayes_factor > 3.0 {
                        "Strong evidence for same p"
                    } else if r.log_bayes_factor > 1.1 {
                        "Substantial evidence for same p"
                    } else if r.log_bayes_factor > 0.0 {
                        "Weak evidence for same p"
                    } else if r.log_bayes_factor > -1.1 {
                        "Weak evidence for different ps"
                    } else if r.log_bayes_factor > -3.0 {
                        "Substantial evidence for different ps"
                    } else {
                        "Strong evidence for different ps"
                    };
                    println!("  Interpretation:   {interpretation}");
                    println!("  (Kass & Raftery 1995 scale: |log BF| > 3 is strong)");
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Hierarchical { s0, n0, sk, nk, rate, mk, iterations, burnin, thin } => {
            // Build interventions vec from either (sk,nk) or (rate,mk)
            let interventions: Vec<(u64, u64)> = if !rate.is_empty() {
                if rate.len() != mk.len() {
                    eprintln!("Error: --rate and --mk must have the same number of values");
                    std::process::exit(1);
                }
                rate.iter().zip(mk.iter()).map(|(r_str, &m)| {
                    let p = match parse_rate(r_str) {
                        Ok(v) => v,
                        Err(e) => {
                            eprintln!("Error parsing rate {r_str:?}: {e}");
                            std::process::exit(1);
                        }
                    };
                    let s = (p * m as f64).round() as u64;
                    (s, m)
                }).collect()
            } else {
                if sk.len() != nk.len() {
                    eprintln!("Error: --sk and --nk must have the same number of values");
                    std::process::exit(1);
                }
                sk.iter().zip(nk.iter()).map(|(&s, &n)| (s, n)).collect()
            };

            match hierarchical_bayes_compare(s0, n0, &interventions, iterations, burnin, thin) {
                Ok(r) => {
                    println!("Hierarchical Bayesian comparison (Metropolis-within-Gibbs MCMC)");
                    println!("  Control:       {s0}/{n0} → posterior mean {cm:.6}", cm = r.control_mean);
                    println!("  MCMC:          {iterations} iter / {burnin} burn-in / thin {thin} → {} samples", r.n_samples);
                    println!();
                    println!("  Group hyperparameters:");
                    println!("    μ (mean):      {mm:.6}  [{mlo:.6}, {mhi:.6}] 95% HDI",
                        mm = r.mu_mean, mlo = r.mu_hdi_lower, mhi = r.mu_hdi_upper);
                    println!("    κ (concentr.): {km:.4}  [{klo:.4}, {khi:.4}] 95% HDI",
                        km = r.kappa_mean, klo = r.kappa_hdi_lower, khi = r.kappa_hdi_upper);
                    println!();
                    println!("  Intervention results (δ_k = θ_k − θ_control):");
                    println!("  {:>4}  {:>7}  {:>8}  {:>8}  {:>22}  {:>7}  {:>8}  {:>4}",
                        "#", "Naive", "Post.θ_k", "δ mean", "95% HDI on δ", "P(δ>0)", "Shrink", "Sig");
                    println!("  {}", "-".repeat(80));
                    for iv in &r.interventions {
                        let shrink_str = match iv.shrinkage {
                            Some(s) => format!("{:+.0}%", s * 100.0),
                            None => "n/a".to_string(),
                        };
                        let sig_str = if iv.significant { "✓" } else { "" };
                        println!("  {:>4}  {:>7.4}  {:>8.6}  {:+8.6}  [{:+9.6}, {:+9.6}]  {:>7.4}  {:>8}  {:>4}",
                            iv.index + 1,
                            iv.naive_rate,
                            iv.posterior_mean_theta,
                            iv.mean_delta,
                            iv.hdi_lower,
                            iv.hdi_upper,
                            iv.prob_delta_gt_zero,
                            shrink_str,
                            sig_str,
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
}

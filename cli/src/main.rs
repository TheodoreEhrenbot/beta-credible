use beta_credible_stats::{credible_interval, same_p_probability};
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
    }
}

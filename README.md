# beta-credible

Bayesian proportion inference: credible intervals and two-sample same-*p* comparison.

**Web app**: https://theodoreehrenbot.github.io/beta-credible/

---

## Modes

### Mode 1: Credible interval for a single proportion

Given *n* successes in *m* trials, compute a Bayesian credible interval.

**Prior**: Uniform = Beta(1, 1).  
**Posterior**: Beta(*n*+1, *m*−*n*+1) — conjugate update from Binomial likelihood.  
**Point estimate**: posterior mean = (*n*+1)/(*m*+2).  
**Credible interval**: equal-tailed, using the posterior Beta quantiles.

### Mode 2: Two-sample same-*p* comparison

Given two samples (*n*₁, *m*₁) and (*n*₂, *m*₂), compute the posterior probability that both come from a single shared proportion *p*.

**Model prior**: P(H_same) = P(H_diff) = 0.5.  
**H_same**: single *p* ~ Uniform[0,1].  
**H_diff**: independent *p*₁, *p*₂ each ~ Uniform[0,1].

Marginal likelihoods (uniform priors on *p*):

> L_same = B(*n*₁+*n*₂+1, *m*₁+*m*₂−*n*₁−*n*₂+1)
>
> L_diff = B(*n*₁+1, *m*₁−*n*₁+1) × B(*n*₂+1, *m*₂−*n*₂+1)

Binomial coefficients cancel in the ratio. The posterior:

> P(H_same | data) = L_same / (L_same + L_diff)

Interpretation follows the Kass & Raftery (1995) Bayes factor scale: |log BF| > 3 is "strong" evidence.

---

## Math references

- Gelman et al., *Bayesian Data Analysis* 3rd ed., §2.4 (conjugate Beta-Binomial)
- DeGroot & Schervish, *Probability and Statistics* 4th ed., §7.3
- MacKay, *Information Theory, Inference, and Learning Algorithms*, Ch. 28 (marginal likelihood)
- Kass & Raftery, "Bayes Factors", *JASA* 1995, 90(430):773–795

---

## CLI usage

```
cargo install beta-credible
```

```
# 95% credible interval: 5 successes in 10 trials
beta-credible interval --successes 5 --trials 10

# Custom confidence level
beta-credible interval -n 3 -m 20 --confidence 0.99

# Compare two proportions
beta-credible same-p --n1 5 --m1 10 --n2 7 --m2 10
```

Example output:

```
Bayesian credible interval (uniform prior)
  Input:          5/10 successes
  Posterior:      Beta(6, 6)
  Point estimate: 0.500000  (posterior mean)
  95% CI:        [0.233794, 0.766206]
```

```
Bayesian same-p comparison (uniform priors, equal model priors)
  Sample 1:        5/10 successes
  Sample 2:        7/10 successes
  P(same p | data): 0.523648
  P(diff p | data): 0.476352
  Log Bayes factor: 0.0945  (same vs diff)
  Interpretation:   Weak evidence for same p
  (Kass & Raftery 1995: |log BF| > 3 is strong evidence)
```

---

## Project structure

```
beta-credible/
├── stats/    # Pure Rust stats library (no UI, no I/O)
├── cli/      # Command-line binary
└── web/      # Yew + WASM frontend (GitHub Pages)
```

## Build

```bash
# Run tests
cargo test --workspace --exclude beta-credible-web

# Build CLI
cargo build -p beta-credible

# Build WASM frontend (requires trunk and wasm32-unknown-unknown target)
rustup target add wasm32-unknown-unknown
cargo install trunk
cd web && trunk build
```

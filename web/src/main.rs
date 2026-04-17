use beta_credible_stats::{
    compare_two_trials, credible_interval, parse_rate, same_p_probability,
};
use yew::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum Tab {
    Interval,
    SameP,
    Compare,
}

#[function_component(App)]
fn app() -> Html {
    let tab = use_state(|| Tab::Interval);

    // ── Interval state ───────────────────────────────────────────────────────
    let n_str = use_state(|| "5".to_string());
    let m_str = use_state(|| "10".to_string());
    let conf_str = use_state(|| "0.95".to_string());
    let interval_result: UseStateHandle<Option<String>> = use_state(|| None);

    // ── Interval: rate-based alternate input ─────────────────────────────────
    let m_rate_str = use_state(|| "100".to_string());
    let rate_str = use_state(|| "72%".to_string());
    let rate_conf_str = use_state(|| "0.95".to_string());
    let rate_result: UseStateHandle<Option<String>> = use_state(|| None);

    // ── SameP state ──────────────────────────────────────────────────────────
    let n1_str = use_state(|| "5".to_string());
    let m1_str = use_state(|| "10".to_string());
    let n2_str = use_state(|| "7".to_string());
    let m2_str = use_state(|| "10".to_string());
    let samep_result: UseStateHandle<Option<String>> = use_state(|| None);

    // ── Compare state ────────────────────────────────────────────────────────
    let cn1_str = use_state(|| "7".to_string());
    let cm1_str = use_state(|| "10".to_string());
    let cn2_str = use_state(|| "3".to_string());
    let cm2_str = use_state(|| "10".to_string());
    let compare_result: UseStateHandle<Option<String>> = use_state(|| None);

    // ── Compare: rate-based alternate input ──────────────────────────────────
    let compare_rate_mode = use_state(|| false);
    let cm1_rate_str = use_state(|| "100".to_string());
    let crate1_str = use_state(|| "70%".to_string());
    let cm2_rate_str = use_state(|| "100".to_string());
    let crate2_str = use_state(|| "30%".to_string());
    let compare_rate_result: UseStateHandle<Option<String>> = use_state(|| None);

    // ── Tab click ────────────────────────────────────────────────────────────
    let on_tab_interval = {
        let tab = tab.clone();
        Callback::from(move |_| tab.set(Tab::Interval))
    };
    let on_tab_samep = {
        let tab = tab.clone();
        Callback::from(move |_| tab.set(Tab::SameP))
    };
    let on_tab_compare = {
        let tab = tab.clone();
        Callback::from(move |_| tab.set(Tab::Compare))
    };

    // ── Interval compute (n + m direct entry) ────────────────────────────────
    let on_interval_compute = {
        let n_str = n_str.clone();
        let m_str = m_str.clone();
        let conf_str = conf_str.clone();
        let result = interval_result.clone();
        Callback::from(move |_: MouseEvent| {
            let n: u64 = match n_str.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    result.set(Some("Error: successes must be a non-negative integer".to_string()));
                    return;
                }
            };
            let m: u64 = match m_str.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    result.set(Some("Error: trials must be a positive integer".to_string()));
                    return;
                }
            };
            let conf: f64 = match conf_str.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    result.set(Some("Error: confidence must be a number between 0 and 1 (e.g. 0.95)".to_string()));
                    return;
                }
            };
            match credible_interval(n, m, conf) {
                Ok(ci) => {
                    let pct = (conf * 100.0).round() as u64;
                    let text = format!(
                        "Bayesian credible interval (uniform prior)\n\
                         Input:          {n}/{m} successes\n\
                         Posterior:      Beta({a:.0}, {b:.0})\n\
                         Point estimate: {pe:.6}  (posterior mean)\n\
                         {pct}% CI:        [{lo:.6}, {hi:.6}]",
                        a = ci.alpha,
                        b = ci.beta_param,
                        pe = ci.point_estimate,
                        lo = ci.lower,
                        hi = ci.upper,
                    );
                    result.set(Some(text));
                }
                Err(e) => result.set(Some(format!("Error: {e}"))),
            }
        })
    };

    // ── Interval compute (total + rate) ──────────────────────────────────────
    let on_rate_compute = {
        let m_rate_str = m_rate_str.clone();
        let rate_str = rate_str.clone();
        let rate_conf_str = rate_conf_str.clone();
        let result = rate_result.clone();
        Callback::from(move |_: MouseEvent| {
            let m: u64 = match m_rate_str.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    result.set(Some("Error: trials must be a positive integer".to_string()));
                    return;
                }
            };
            let rate = match parse_rate(&rate_str) {
                Ok(v) => v,
                Err(e) => {
                    result.set(Some(format!("Error: {e}")));
                    return;
                }
            };
            let conf: f64 = match rate_conf_str.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    result.set(Some("Error: confidence must be a number between 0 and 1 (e.g. 0.95)".to_string()));
                    return;
                }
            };
            // n = round(rate * m), round-half-away-from-zero
            let n = (rate * m as f64).round() as u64;
            match credible_interval(n, m, conf) {
                Ok(ci) => {
                    let pct = (conf * 100.0).round() as u64;
                    let text = format!(
                        "Bayesian credible interval (uniform prior)\n\
                         Input:          rate={rate_pct:.4}% of {m} trials → n={n}\n\
                         Posterior:      Beta({a:.0}, {b:.0})\n\
                         Point estimate: {pe:.6}  (posterior mean)\n\
                         {pct}% CI:        [{lo:.6}, {hi:.6}]",
                        rate_pct = rate * 100.0,
                        a = ci.alpha,
                        b = ci.beta_param,
                        pe = ci.point_estimate,
                        lo = ci.lower,
                        hi = ci.upper,
                    );
                    result.set(Some(text));
                }
                Err(e) => result.set(Some(format!("Error: {e}"))),
            }
        })
    };

    // ── SameP compute ────────────────────────────────────────────────────────
    let on_samep_compute = {
        let n1_str = n1_str.clone();
        let m1_str = m1_str.clone();
        let n2_str = n2_str.clone();
        let m2_str = m2_str.clone();
        let result = samep_result.clone();
        Callback::from(move |_: MouseEvent| {
            let parse_u64 = |s: &str, name: &str| -> Result<u64, String> {
                s.trim()
                    .parse::<u64>()
                    .map_err(|_| format!("Error: {name} must be a non-negative integer"))
            };
            let n1 = match parse_u64(&n1_str, "n1") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            let m1 = match parse_u64(&m1_str, "m1") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            let n2 = match parse_u64(&n2_str, "n2") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            let m2 = match parse_u64(&m2_str, "m2") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            match same_p_probability(n1, m1, n2, m2) {
                Ok(r) => {
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
                    let text = format!(
                        "Bayesian same-p comparison (equal model priors)\n\
                         Sample 1:        {n1}/{m1} successes\n\
                         Sample 2:        {n2}/{m2} successes\n\
                         P(same p | data): {ps:.6}\n\
                         P(diff p | data): {pd:.6}\n\
                         Log Bayes factor: {lbf:.4}  (same vs diff)\n\
                         Interpretation:   {interpretation}\n\
                         (Kass & Raftery 1995: |log BF| > 3 is strong evidence)",
                        ps = r.prob_same,
                        pd = r.prob_diff,
                        lbf = r.log_bayes_factor,
                    );
                    result.set(Some(text));
                }
                Err(e) => result.set(Some(format!("Error: {e}"))),
            }
        })
    };

    // ── Compare compute ──────────────────────────────────────────────────────
    let on_compare_compute = {
        let cn1_str = cn1_str.clone();
        let cm1_str = cm1_str.clone();
        let cn2_str = cn2_str.clone();
        let cm2_str = cm2_str.clone();
        let result = compare_result.clone();
        Callback::from(move |_: MouseEvent| {
            let parse_u64 = |s: &str, name: &str| -> Result<u64, String> {
                s.trim()
                    .parse::<u64>()
                    .map_err(|_| format!("Error: {name} must be a non-negative integer"))
            };
            let n1 = match parse_u64(&cn1_str, "n1") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            let m1 = match parse_u64(&cm1_str, "m1") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            let n2 = match parse_u64(&cn2_str, "n2") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            let m2 = match parse_u64(&cm2_str, "m2") {
                Ok(v) => v,
                Err(e) => { result.set(Some(e)); return; }
            };
            match compare_two_trials(n1, m1, n2, m2) {
                Ok(r) => {
                    let text = format!(
                        "Compare two trials (independent uniform priors)\n\
                         Trial 1:         {n1}/{m1} successes  →  posterior mean {pe1:.6}\n\
                         Trial 2:         {n2}/{m2} successes  →  posterior mean {pe2:.6}\n\
                         Mean difference: E[p1 − p2] = {diff:.6}\n\
                         P(p1 > p2):      {pgt:.4}  (Monte Carlo, N=100,000)\n\
                         95% CI on diff:  [{cilo:.6}, {cihi:.6}]  (MC percentiles)\n\
                         (Gelman et al. BDA 3rd ed. Ch. 2; Robert & Casella MCM 2nd ed. Ch. 3)",
                        pe1 = r.mean1,
                        pe2 = r.mean2,
                        diff = r.mean_diff,
                        pgt = r.prob_p1_gt_p2,
                        cilo = r.ci_diff_lower,
                        cihi = r.ci_diff_upper,
                    );
                    result.set(Some(text));
                }
                Err(e) => result.set(Some(format!("Error: {e}"))),
            }
        })
    };

    // ── Compare: rate compute ────────────────────────────────────────────────
    let on_compare_rate_compute = {
        let cm1_rate_str = cm1_rate_str.clone();
        let crate1_str = crate1_str.clone();
        let cm2_rate_str = cm2_rate_str.clone();
        let crate2_str = crate2_str.clone();
        let result = compare_rate_result.clone();
        Callback::from(move |_: MouseEvent| {
            let m1: u64 = match cm1_rate_str.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    result.set(Some("Error: trial 1 trials must be a positive integer".to_string()));
                    return;
                }
            };
            let rate1 = match parse_rate(&crate1_str) {
                Ok(v) => v,
                Err(e) => { result.set(Some(format!("Error: {e}"))); return; }
            };
            let m2: u64 = match cm2_rate_str.trim().parse() {
                Ok(v) => v,
                Err(_) => {
                    result.set(Some("Error: trial 2 trials must be a positive integer".to_string()));
                    return;
                }
            };
            let rate2 = match parse_rate(&crate2_str) {
                Ok(v) => v,
                Err(e) => { result.set(Some(format!("Error: {e}"))); return; }
            };
            let n1 = (rate1 * m1 as f64).round() as u64;
            let n2 = (rate2 * m2 as f64).round() as u64;
            match compare_two_trials(n1, m1, n2, m2) {
                Ok(r) => {
                    let text = format!(
                        "Compare two trials (independent uniform priors)\n\
                         Trial 1:         rate={r1_pct:.4}% of {m1} trials → n1={n1}  →  posterior mean {pe1:.6}\n\
                         Trial 2:         rate={r2_pct:.4}% of {m2} trials → n2={n2}  →  posterior mean {pe2:.6}\n\
                         Mean difference: E[p1 − p2] = {diff:.6}\n\
                         P(p1 > p2):      {pgt:.4}  (Monte Carlo, N=100,000)\n\
                         95% CI on diff:  [{cilo:.6}, {cihi:.6}]  (MC percentiles)\n\
                         (Gelman et al. BDA 3rd ed. Ch. 2; Robert & Casella MCM 2nd ed. Ch. 3)",
                        r1_pct = rate1 * 100.0,
                        r2_pct = rate2 * 100.0,
                        pe1 = r.mean1,
                        pe2 = r.mean2,
                        diff = r.mean_diff,
                        pgt = r.prob_p1_gt_p2,
                        cilo = r.ci_diff_lower,
                        cihi = r.ci_diff_upper,
                    );
                    result.set(Some(text));
                }
                Err(e) => result.set(Some(format!("Error: {e}"))),
            }
        })
    };

    // ── Compare: toggle input mode ───────────────────────────────────────────
    let on_compare_toggle_nm = {
        let compare_rate_mode = compare_rate_mode.clone();
        Callback::from(move |_| compare_rate_mode.set(false))
    };
    let on_compare_toggle_rate = {
        let compare_rate_mode = compare_rate_mode.clone();
        Callback::from(move |_| compare_rate_mode.set(true))
    };

    // ── Input handlers ────────────────────────────────────────────────────────
    macro_rules! on_input {
        ($state:expr) => {{
            let state = $state.clone();
            Callback::from(move |e: InputEvent| {
                use yew::TargetCast;
                use web_sys::HtmlInputElement;
                let input: HtmlInputElement = e.target_unchecked_into();
                state.set(input.value());
            })
        }};
    }

    // ── Render ────────────────────────────────────────────────────────────────
    html! {
        <>
        <h1>{"beta-credible"}</h1>
        <p class="desc">
            {"Bayesian proportion inference. "}
            <a href="https://github.com/TheodoreEhrenbot/beta-credible">{"Source & CLI on GitHub"}</a>
            {". Prior: Uniform = Beta(1,1)."}
        </p>

        <div class="tabs">
            <div
                class={if *tab == Tab::Interval { "tab active" } else { "tab" }}
                onclick={on_tab_interval}
            >
                {"Credible Interval"}
            </div>
            <div
                class={if *tab == Tab::SameP { "tab active" } else { "tab" }}
                onclick={on_tab_samep}
            >
                {"Same-p Comparison"}
            </div>
            <div
                class={if *tab == Tab::Compare { "tab active" } else { "tab" }}
                onclick={on_tab_compare}
            >
                {"Compare Two Trials"}
            </div>
        </div>

        { if *tab == Tab::Interval { html! {
            <>
            <h2>{"Mode 1: Single-proportion credible interval"}</h2>
            <p class="desc">
                {"Given n successes in m trials, compute the posterior credible interval. "}
                {"Posterior: Beta(n+1, m−n+1). Point estimate = posterior mean = (n+1)/(m+2)."}
            </p>

            <h3 class="input-mode-label">{"Enter successes and trials directly"}</h3>
            <div class="form-group">
                <div class="field">
                    <label for="n">{"Successes (n)"}</label>
                    <input id="n" type="number" min="0" value={(*n_str).clone()} oninput={on_input!(n_str)} />
                </div>
                <div class="field">
                    <label for="m">{"Trials (m)"}</label>
                    <input id="m" type="number" min="1" value={(*m_str).clone()} oninput={on_input!(m_str)} />
                </div>
                <div class="field">
                    <label for="conf">{"Confidence (e.g. 0.95)"}</label>
                    <input id="conf" type="text" value={(*conf_str).clone()} oninput={on_input!(conf_str)} />
                </div>
                <button onclick={on_interval_compute}>{"Compute"}</button>
            </div>
            { if let Some(text) = (*interval_result).clone() {
                let is_err = text.starts_with("Error");
                html! {
                    <div class={if is_err { "result error" } else { "result" }}>{ text }</div>
                }
            } else { html!{} } }

            <h3 class="input-mode-label">{"Or: enter total trials and a rate"}</h3>
            <p class="desc">
                {"Enter a rate as a decimal (e.g. "}
                <code>{"0.72"}</code>
                {") or percentage (e.g. "}
                <code>{"72"}</code>
                {" or "}
                <code>{"72%"}</code>
                {"). n = round(rate × m) is computed for you."}
            </p>
            <div class="form-group">
                <div class="field">
                    <label for="m_rate">{"Trials (m)"}</label>
                    <input id="m_rate" type="number" min="1" value={(*m_rate_str).clone()} oninput={on_input!(m_rate_str)} />
                </div>
                <div class="field">
                    <label for="rate">{"Rate (e.g. '72%' or '0.72')"}</label>
                    <input id="rate" type="text" value={(*rate_str).clone()} oninput={on_input!(rate_str)} />
                </div>
                <div class="field">
                    <label for="rate_conf">{"Confidence (e.g. 0.95)"}</label>
                    <input id="rate_conf" type="text" value={(*rate_conf_str).clone()} oninput={on_input!(rate_conf_str)} />
                </div>
                <button onclick={on_rate_compute}>{"Compute"}</button>
            </div>
            { if let Some(text) = (*rate_result).clone() {
                let is_err = text.starts_with("Error");
                html! {
                    <div class={if is_err { "result error" } else { "result" }}>{ text }</div>
                }
            } else { html!{} } }
            </>
        }} else if *tab == Tab::SameP { html! {
            <>
            <div class="deprecation-notice">
                {"⚠ "}
                <strong>{"Deprecated."}</strong>
                {" Use the new 'Compare Two Trials' tab instead. \
                  This Bayes-factor comparison will be removed in a future update."}
            </div>
            <h2>{"Mode 2: Two-sample same-p comparison"}</h2>
            <p class="desc">
                {"Given two samples, what is the posterior probability they share a single underlying proportion p? "}
                {"Prior: P(H_same) = P(H_diff) = 0.5; each p ~ Uniform[0,1]."}
            </p>
            <div class="form-group">
                <div class="field">
                    <label for="n1">{"Sample 1 successes (n1)"}</label>
                    <input id="n1" type="number" min="0" value={(*n1_str).clone()} oninput={on_input!(n1_str)} />
                </div>
                <div class="field">
                    <label for="m1">{"Sample 1 trials (m1)"}</label>
                    <input id="m1" type="number" min="1" value={(*m1_str).clone()} oninput={on_input!(m1_str)} />
                </div>
                <div class="field">
                    <label for="n2">{"Sample 2 successes (n2)"}</label>
                    <input id="n2" type="number" min="0" value={(*n2_str).clone()} oninput={on_input!(n2_str)} />
                </div>
                <div class="field">
                    <label for="m2">{"Sample 2 trials (m2)"}</label>
                    <input id="m2" type="number" min="1" value={(*m2_str).clone()} oninput={on_input!(m2_str)} />
                </div>
                <button onclick={on_samep_compute}>{"Compute"}</button>
            </div>
            { if let Some(text) = (*samep_result).clone() {
                let is_err = text.starts_with("Error");
                html! {
                    <div class={if is_err { "result error" } else { "result" }}>{ text }</div>
                }
            } else { html!{} } }
            </>
        }} else { html! {
            <>
            <h2>{"Mode 3: Compare two trials"}</h2>
            <p class="desc">
                {"Given two independent trials, estimate the difference between their underlying proportions. "}
                {"Posteriors: Beta(n+1, m−n+1) for each. "}
                {"P(p1 > p2) computed via Monte Carlo (N=100,000 samples)."}
            </p>

            <div class="input-mode-toggle">
                <button
                    class={if !*compare_rate_mode { "toggle-btn active" } else { "toggle-btn" }}
                    onclick={on_compare_toggle_nm}
                >{"n/m mode"}</button>
                <button
                    class={if *compare_rate_mode { "toggle-btn active" } else { "toggle-btn" }}
                    onclick={on_compare_toggle_rate}
                >{"rate mode"}</button>
            </div>

            { if !*compare_rate_mode { html! {
                <>
                <div class="form-group">
                    <div class="field">
                        <label for="cn1">{"Trial 1 successes (n1)"}</label>
                        <input id="cn1" type="number" min="0" value={(*cn1_str).clone()} oninput={on_input!(cn1_str)} />
                    </div>
                    <div class="field">
                        <label for="cm1">{"Trial 1 trials (m1)"}</label>
                        <input id="cm1" type="number" min="1" value={(*cm1_str).clone()} oninput={on_input!(cm1_str)} />
                    </div>
                    <div class="field">
                        <label for="cn2">{"Trial 2 successes (n2)"}</label>
                        <input id="cn2" type="number" min="0" value={(*cn2_str).clone()} oninput={on_input!(cn2_str)} />
                    </div>
                    <div class="field">
                        <label for="cm2">{"Trial 2 trials (m2)"}</label>
                        <input id="cm2" type="number" min="1" value={(*cm2_str).clone()} oninput={on_input!(cm2_str)} />
                    </div>
                    <button onclick={on_compare_compute}>{"Compute"}</button>
                </div>
                { if let Some(text) = (*compare_result).clone() {
                    let is_err = text.starts_with("Error");
                    html! {
                        <div class={if is_err { "result error" } else { "result" }}>{ text }</div>
                    }
                } else { html!{} } }
                </>
            }} else { html! {
                <>
                <div class="form-group">
                    <div class="field">
                        <label for="cm1_rate">{"Trial 1 trials (m1)"}</label>
                        <input id="cm1_rate" type="number" min="1" value={(*cm1_rate_str).clone()} oninput={on_input!(cm1_rate_str)} />
                    </div>
                    <div class="field">
                        <label for="crate1">{"Trial 1 rate"}</label>
                        <input id="crate1" type="text" style="width: 80px" value={(*crate1_str).clone()} oninput={on_input!(crate1_str)} />
                    </div>
                    <div class="field">
                        <label for="cm2_rate">{"Trial 2 trials (m2)"}</label>
                        <input id="cm2_rate" type="number" min="1" value={(*cm2_rate_str).clone()} oninput={on_input!(cm2_rate_str)} />
                    </div>
                    <div class="field">
                        <label for="crate2">{"Trial 2 rate"}</label>
                        <input id="crate2" type="text" style="width: 80px" value={(*crate2_str).clone()} oninput={on_input!(crate2_str)} />
                    </div>
                    <button onclick={on_compare_rate_compute}>{"Compute"}</button>
                </div>
                { if let Some(text) = (*compare_rate_result).clone() {
                    let is_err = text.starts_with("Error");
                    html! {
                        <div class={if is_err { "result error" } else { "result" }}>{ text }</div>
                    }
                } else { html!{} } }
                </>
            }} }
            </>
        }}}
        </>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}

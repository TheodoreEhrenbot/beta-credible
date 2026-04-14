use beta_credible_stats::{credible_interval, same_p_probability};
use yew::prelude::*;

#[derive(Debug, Clone, PartialEq)]
enum Tab {
    Interval,
    SameP,
}

#[function_component(App)]
fn app() -> Html {
    let tab = use_state(|| Tab::Interval);

    // ── Interval state ───────────────────────────────────────────────────────
    let n_str = use_state(|| "5".to_string());
    let m_str = use_state(|| "10".to_string());
    let conf_str = use_state(|| "0.95".to_string());
    let interval_result: UseStateHandle<Option<String>> = use_state(|| None);

    // ── SameP state ──────────────────────────────────────────────────────────
    let n1_str = use_state(|| "5".to_string());
    let m1_str = use_state(|| "10".to_string());
    let n2_str = use_state(|| "7".to_string());
    let m2_str = use_state(|| "10".to_string());
    let samep_result: UseStateHandle<Option<String>> = use_state(|| None);

    // ── Tab click ────────────────────────────────────────────────────────────
    let on_tab_interval = {
        let tab = tab.clone();
        Callback::from(move |_| tab.set(Tab::Interval))
    };
    let on_tab_samep = {
        let tab = tab.clone();
        Callback::from(move |_| tab.set(Tab::SameP))
    };

    // ── Interval compute ─────────────────────────────────────────────────────
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
        </div>

        { if *tab == Tab::Interval { html! {
            <>
            <h2>{"Mode 1: Single-proportion credible interval"}</h2>
            <p class="desc">
                {"Given n successes in m trials, compute the posterior credible interval. "}
                {"Posterior: Beta(n+1, m−n+1). Point estimate = posterior mean = (n+1)/(m+2)."}
            </p>
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
            </>
        }} else { html! {
            <>
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
        }}}
        </>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}

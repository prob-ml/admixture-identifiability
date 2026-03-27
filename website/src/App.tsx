import { type CSSProperties } from "react";

/* ------------------------------------------------------------------ */
/*  Paths to result images (served from public/)                       */
/* ------------------------------------------------------------------ */
const CASE = "case_v_binary_np95_pihat95";
const img = (name: string) => `/results/${CASE}/${name}`;

/* ------------------------------------------------------------------ */
/*  Reusable style helpers                                             */
/* ------------------------------------------------------------------ */
const container: CSSProperties = {
  maxWidth: 860,
  margin: "0 auto",
  padding: "2rem 1.5rem 4rem",
};

const section: CSSProperties = {
  marginTop: "3rem",
};

const h1Style: CSSProperties = {
  fontSize: "2.2rem",
  fontWeight: 700,
  lineHeight: 1.2,
  marginBottom: "0.5rem",
};

const h2Style: CSSProperties = {
  fontSize: "1.5rem",
  fontWeight: 600,
  marginBottom: "0.75rem",
  borderBottom: "2px solid var(--accent)",
  paddingBottom: "0.25rem",
  display: "inline-block",
};

const h3Style: CSSProperties = {
  fontSize: "1.15rem",
  fontWeight: 600,
  marginTop: "1.5rem",
  marginBottom: "0.5rem",
};

const pStyle: CSSProperties = {
  marginBottom: "1rem",
};

const figStyle: CSSProperties = {
  margin: "1.5rem 0",
  textAlign: "center",
};

const imgStyle: CSSProperties = {
  maxWidth: "100%",
  borderRadius: 6,
  border: "1px solid var(--border)",
  boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
};

const captionStyle: CSSProperties = {
  fontSize: "0.85rem",
  color: "var(--muted)",
  marginTop: "0.5rem",
};

const tableWrap: CSSProperties = {
  overflowX: "auto",
  margin: "1rem 0",
};

const tableStyle: CSSProperties = {
  borderCollapse: "collapse",
  width: "100%",
  fontSize: "0.95rem",
};

const thStyle: CSSProperties = {
  textAlign: "left",
  padding: "0.5rem 0.75rem",
  borderBottom: "2px solid var(--border)",
  fontWeight: 600,
};

const tdStyle: CSSProperties = {
  padding: "0.5rem 0.75rem",
  borderBottom: "1px solid var(--border)",
};

const eqBlock: CSSProperties = {
  background: "var(--code-bg)",
  borderRadius: 6,
  padding: "0.75rem 1rem",
  fontFamily: "var(--font-mono)",
  fontSize: "0.9rem",
  overflowX: "auto",
  margin: "1rem 0",
};

const gridTwo: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "1.5rem",
};

/* ------------------------------------------------------------------ */
/*  Components                                                         */
/* ------------------------------------------------------------------ */

function Figure({
  src,
  alt,
  caption,
}: {
  src: string;
  alt: string;
  caption: string;
}) {
  return (
    <figure style={figStyle}>
      <img src={src} alt={alt} style={imgStyle} />
      <figcaption style={captionStyle}>{caption}</figcaption>
    </figure>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */
export default function App() {
  return (
    <div style={container}>
      {/* ---- Header ---- */}
      <header>
        <h1 style={h1Style}>Admixture Identifiability</h1>
        <p style={{ ...pStyle, fontSize: "1.1rem", color: "var(--muted)" }}>
          Recovering a latent shape distribution from noisy additive
          observations using rectified flow matching.
        </p>
      </header>

      {/* ---- The Setup ---- */}
      <section style={section}>
        <h2 style={h2Style}>The Setup: Gaussian Bump Model</h2>

        <p style={pStyle}>
          We observe a one-dimensional signal <code>X</code> that is the sum of
          many shifted "bumps." Each bump is a Gaussian-shaped pulse whose
          amplitude and width are drawn from an unknown distribution{" "}
          <strong>π</strong>. A large fraction of the bumps are{" "}
          <em>null</em> — contributing exactly zero — so the signal is sparse.
        </p>

        <h3 style={h3Style}>Shape function (V-binary gating)</h3>
        <p style={pStyle}>
          Each latent mark has a binary gate <code>V ∈ &#123;0, 1&#125;</code>{" "}
          and parameters <code>(ρ, log σ)</code> ∈ ℝ². The shape on the
          integer grid &#123;−L, …, L&#125; is:
        </p>
        <div style={eqBlock}>
          f(x) = V · ρ · exp(−x² / (2 · exp(2·log σ)))
        </div>
        <p style={pStyle}>
          When <code>V=0</code> (null mark), the contribution to X is exactly
          zero and (ρ, log σ) = (0, 0) by convention. When <code>V=1</code>{" "}
          (active mark), ρ is used directly as the amplitude. This clean
          separation avoids the problem where the flow matching loss gets
          inflated by trying to predict arbitrary null-mark parameters (e.g.
          ρ ≈ −10) that don't affect X at all.
        </p>

        <h3 style={h3Style}>Generative process</h3>
        <p style={pStyle}>
          Given parameters (L, T, π), a single observation X ∈ ℝᵀ⁻²ᴸ⁺¹ is
          generated by:
        </p>
        <ol style={{ ...pStyle, paddingLeft: "1.5rem" }}>
          <li>
            For each position t ∈ &#123;0, …, T&#125;, sample (Vₜ, Uₜ) =
            (Vₜ, ρₜ, log σₜ) ∼ π.
          </li>
          <li>
            For each visible position t ∈ &#123;L, …, T−L&#125;, compute Xₜ =
            Σ Vₜ · ρ_τ · exp(−(t−τ)² / (2·exp(2·log σ_τ))).
          </li>
        </ol>

        <h3 style={h3Style}>Goal</h3>
        <p style={pStyle}>
          <strong>Estimate π from samples of X alone.</strong> The latent
          (V, U) marks are never observed. In particular, we want to recover
          the null probability — the fraction of marks where V=0.
        </p>
      </section>

      {/* ---- What We Do ---- */}
      <section style={section}>
        <h2 style={h2Style}>What We Do: Two-Model Rectified Flow Matching</h2>

        <p style={pStyle}>
          We train <strong>two</strong> networks simultaneously:
        </p>
        <ol style={{ ...pStyle, paddingLeft: "1.5rem" }}>
          <li>
            <strong>V-classifier</strong> — an MLP that predicts the binary gate
            V from X (binary cross-entropy loss).
          </li>
          <li>
            <strong>U-flow</strong> — a conditional rectified flow matching
            network that maps noise to <code>U ∈ ℝ⁽ᵀ⁺¹⁾ˣ²</code> given (X, V).
            The flow matching loss is <em>masked</em> so that only active (V=1)
            positions contribute — null positions are trivially zero and don't
            waste model capacity.
          </li>
        </ol>

        <h3 style={h3Style}>Phase I — Warm-up</h3>
        <p style={pStyle}>
          Train both networks on (X, V, U) triples sampled from an initial guess
          π̂. Each gradient step uses <em>freshly simulated</em> data.
        </p>

        <h3 style={h3Style}>Phase II — Iterative refinement (bootstrap)</h3>
        <ol style={{ ...pStyle, paddingLeft: "1.5rem" }}>
          <li>
            Draw X samples from the <strong>true</strong> model (L, T, π).
          </li>
          <li>
            Predict V̂ using the V-classifier (stop-gradient).
          </li>
          <li>
            Infer Û using the U-flow conditioned on V̂ (stop-gradient).
            Null positions (V̂=0) get U=(0, 0).
          </li>
          <li>
            Define π̂ as the empirical distribution over (V̂, Û).
          </li>
          <li>
            Train both models on fresh (X, V, U) drawn from this updated π̂.
          </li>
        </ol>
        <p style={pStyle}>
          Phase II restarts both optimisers from scratch — fresh cosine learning
          rate schedules and fresh Adam moments. An <strong>oracle</strong> pair
          of models is trained in parallel on ground-truth (X, V, U) from the
          true π as a performance reference.
        </p>

        <h3 style={h3Style}>Key insight: binary V separates null classification
          from mark prediction</h3>
        <p style={pStyle}>
          The previous approach used <code>softplus(ρ)</code> with null marks at
          ρ = −10.  This led to inflated flow matching losses because the model
          wasted capacity trying to predict arbitrary (ρ, log σ) values in the
          null regime — parameters that don't affect X at all. The binary V
          cleanly separates the "is this mark null?" question (handled by the
          V-classifier) from the "what are its parameters?" question (handled by
          the masked U-flow).
        </p>
      </section>

      {/* ---- What We Find ---- */}
      <section style={section}>
        <h2 style={h2Style}>What We Find</h2>

        <h3 style={h3Style}>
          Case study: <code>case_v_binary_np95_pihat95</code>
        </h3>

        <div style={tableWrap}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Parameter</th>
                <th style={thStyle}>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={tdStyle}>Support window L</td>
                <td style={tdStyle}>3</td>
              </tr>
              <tr>
                <td style={tdStyle}>Visible window T</td>
                <td style={tdStyle}>20</td>
              </tr>
              <tr>
                <td style={tdStyle}>True null probability</td>
                <td style={tdStyle}>0.95</td>
              </tr>
              <tr>
                <td style={tdStyle}>Initial guess null probability</td>
                <td style={tdStyle}>0.95</td>
              </tr>
              <tr>
                <td style={tdStyle}>Phase I steps</td>
                <td style={tdStyle}>10,000</td>
              </tr>
              <tr>
                <td style={tdStyle}>Phase II steps</td>
                <td style={tdStyle}>150,000</td>
              </tr>
              <tr>
                <td style={tdStyle}>Architecture</td>
                <td style={tdStyle}>V-classifier + masked U-flow</td>
              </tr>
              <tr>
                <td style={tdStyle}>Seed</td>
                <td style={tdStyle}>42</td>
              </tr>
            </tbody>
          </table>
        </div>

        <p style={pStyle}>
          The binary-V two-model architecture cleanly separates null
          classification from mark parameter prediction, enabling the flow
          matching loss to focus exclusively on learning the distribution of
          active marks.
        </p>
      </section>

      {/* ---- Training Loss Curves ---- */}
      <section style={section}>
        <h2 style={h2Style}>Training Loss Curves</h2>

        <p style={pStyle}>
          We evaluate the masked U-flow loss on a fixed batch of 512
          ground-truth <code>(X, V, U)</code> triples throughout training.
          This "GT eval loss" measures how well the flow network predicts the
          velocity field for <em>active</em> (V=1) marks — the true test of
          whether the model is learning the correct posterior.
        </p>

        <p style={pStyle}>
          In <strong>Phase I</strong>, we plot the U-flow training loss,
          GT eval U-flow loss, and V-classifier loss. All losses decrease
          together, confirming that training on the proxy distribution also
          improves performance on real data.
        </p>

        <p style={pStyle}>
          In <strong>Phase II</strong>, we compare two pairs of models evaluated
          on the same fixed GT data. The <em>bootstrap models</em> (initialised
          from Phase I) are trained using the bootstrap procedure — they never
          see ground-truth V or U. The <em>oracle models</em> are initialised
          from scratch and trained on fresh ground-truth{" "}
          <code>(X, V, U)</code> triples at each step.
        </p>

        <p style={pStyle}>
          The y-axis is clipped so that both curves are visible — using{" "}
          <code>min(oracle_max, bootstrap_max)</code> as the upper limit to
          avoid showing the very large early losses from the randomly
          initialised oracle.
        </p>

        <Figure
          src={img("loss_curves.png")}
          alt="Flow matching loss curves"
          caption="Loss curves evaluated on 512 fixed ground-truth (X, V, U) triples. Left: Phase I shows U-flow training loss, GT eval loss, and V-classifier loss. Right: Phase II compares the bootstrap model (orange) against an oracle model trained from scratch (green dashed)."
        />
      </section>

      {/* ---- Diagnostic Plots ---- */}
      <section style={section}>
        <h2 style={h2Style}>Diagnostic Plots</h2>

        <h3 style={h3Style}>Observation data</h3>
        <div style={gridTwo}>
          <Figure
            src={img("xdata_true.png")}
            alt="True X observations"
            caption="X samples from the true model (π with 95% null marks)"
          />
          <Figure
            src={img("xdata_pihat.png")}
            alt="Initial guess X observations"
            caption="X samples from the initial guess π̂"
          />
        </div>

        <h3 style={h3Style}>Posterior recovery</h3>
        <Figure
          src={img("posterior_vs_truth.png")}
          alt="Posterior vs truth scatter"
          caption="Ground-truth π samples vs aggregate posterior (V, U) inferred by the two-model architecture"
        />

        <h3 style={h3Style}>ρ marginal distribution</h3>
        <Figure
          src={img("rho_marginal.png")}
          alt="Rho marginal histogram"
          caption="Marginal histogram of ρ for V=1 (active) marks: true distribution vs flow-inferred posterior"
        />

        <h3 style={h3Style}>ρ survival function</h3>
        <Figure
          src={img("rho_survival.png")}
          alt="Rho survival function"
          caption="Survival function P(ρ > t) for V=1 marks: true vs inferred"
        />

        <h3 style={h3Style}>Latent (V, U) comparison</h3>
        <Figure
          src={img("true_vs_inferred_U.png")}
          alt="True vs inferred (V, U)"
          caption="True latent (V, U) vs inferred (V, U) for a batch of observations"
        />
      </section>

      {/* ---- Footer ---- */}
      <footer
        style={{
          marginTop: "4rem",
          padding: "1.5rem 0",
          borderTop: "1px solid var(--border)",
          fontSize: "0.85rem",
          color: "var(--muted)",
          textAlign: "center",
        }}
      >
        <p>
          Source:{" "}
          <a href="https://github.com/prob-ml/admixture-identifiability">
            prob-ml/admixture-identifiability
          </a>
        </p>
      </footer>
    </div>
  );
}

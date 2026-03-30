import { type CSSProperties } from "react";
import { Link, useParams } from "react-router-dom";
import { CASES } from "./cases";

/* ------------------------------------------------------------------ */
/*  Reusable style helpers                                             */
/* ------------------------------------------------------------------ */
const container: CSSProperties = {
  maxWidth: 860,
  margin: "0 auto",
  padding: "2rem 1.5rem 4rem",
};

const section: CSSProperties = { marginTop: "3rem" };

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

const pStyle: CSSProperties = { marginBottom: "1rem" };

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

const tableWrap: CSSProperties = { overflowX: "auto", margin: "1rem 0" };

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
/*  RunPage                                                            */
/* ------------------------------------------------------------------ */
export default function RunPage() {
  const { slug } = useParams<{ slug: string }>();
  const run = CASES.find((c) => c.slug === slug);

  if (!run) {
    return (
      <div style={container}>
        <h1 style={h1Style}>Run not found</h1>
        <p style={pStyle}>
          No case study with slug <code>{slug}</code>.
        </p>
        <Link to="/">← Back to main page</Link>
      </div>
    );
  }

  const img = (name: string) => `/results/${run.id}/${name}`;

  return (
    <div style={container}>
      {/* ---- Navigation ---- */}
      <nav style={{ marginBottom: "1rem", fontSize: "0.9rem" }}>
        <Link to="/">← Back to main page</Link>
      </nav>

      {/* ---- Header ---- */}
      <header>
        <h1 style={h1Style}>Run: {run.label}</h1>
        <p style={{ ...pStyle, fontSize: "1.05rem", color: "var(--muted)" }}>
          Case <code>{run.id}</code>
        </p>
      </header>

      {/* ---- Initial pihat configuration ---- */}
      <section style={section}>
        <h2 style={h2Style}>Initial π̂ Configuration</h2>

        <p style={pStyle}>
          The initial guess <strong>π̂</strong> used for Phase I warm-up:
        </p>

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
                <td style={tdStyle}>P̂(V=0) — initial null probability</td>
                <td style={tdStyle}><strong>{run.nullProbInit}</strong></td>
              </tr>
              <tr>
                <td style={tdStyle}>True P(V=0)</td>
                <td style={tdStyle}>{run.nullProbTrue}</td>
              </tr>
              <tr>
                <td style={tdStyle}>Number of mixture components</td>
                <td style={tdStyle}>{run.pihatComponents.length}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h3 style={h3Style}>Distribution on U | V=1</h3>
        <p style={pStyle}>
          The initial guess for the mark parameters (ρ, log σ) of active marks:
        </p>
        <div style={tableWrap}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Component</th>
                <th style={thStyle}>Weight</th>
                <th style={thStyle}>Mean (ρ, log σ)</th>
                <th style={thStyle}>Std (ρ, log σ)</th>
              </tr>
            </thead>
            <tbody>
              {run.pihatComponents.map((c, i) => (
                <tr key={i}>
                  <td style={tdStyle}>{i + 1}</td>
                  <td style={tdStyle}>{c.weight}</td>
                  <td style={tdStyle}>
                    <code>({c.mean[0]}, {c.mean[1]})</code>
                  </td>
                  <td style={tdStyle}>
                    <code>({c.std[0]}, {c.std[1]})</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={eqBlock}>
          π̂: P(V=0) = {run.nullProbInit}, &nbsp; U | V=1 ∼{" "}
          {run.pihatComponents.length === 1
            ? `𝒩(μ=(${run.pihatComponents[0].mean.join(", ")}), σ=(${run.pihatComponents[0].std.join(", ")}))`
            : "mixture of Gaussians (see table above)"}
        </div>
      </section>

      {/* ---- Run parameters ---- */}
      <section style={section}>
        <h2 style={h2Style}>Run Parameters</h2>

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
                <td style={tdStyle}>{run.L}</td>
              </tr>
              <tr>
                <td style={tdStyle}>Visible window T</td>
                <td style={tdStyle}>{run.T}</td>
              </tr>
              <tr>
                <td style={tdStyle}>Phase I steps</td>
                <td style={tdStyle}>{run.phaseISteps.toLocaleString()}</td>
              </tr>
              <tr>
                <td style={tdStyle}>Phase II steps</td>
                <td style={tdStyle}>{run.phaseIISteps.toLocaleString()}</td>
              </tr>
              <tr>
                <td style={tdStyle}>Architecture</td>
                <td style={tdStyle}>V-flow + masked U-flow</td>
              </tr>
              <tr>
                <td style={tdStyle}>Seed</td>
                <td style={tdStyle}>{run.seed}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* ---- Training Loss Curves ---- */}
      <section style={section}>
        <h2 style={h2Style}>Training Loss Curves</h2>

        <p style={pStyle}>
          We evaluate the masked U-flow loss on a fixed batch of 512
          ground-truth <code>(X, V, U)</code> triples throughout training.
          This "GT eval loss" measures how well the flow network predicts the
          velocity field for <em>active</em> (V=1) marks.
        </p>

        <Figure
          src={img("loss_curves.png")}
          alt="Flow matching loss curves"
          caption="Loss curves evaluated on 512 fixed ground-truth (X, V, U) triples. Left: Phase I shows U-flow training loss, GT eval loss, and V-flow loss. Right: Phase II compares the bootstrap model (orange) against an oracle model trained from scratch (green dashed)."
        />
      </section>

      {/* ---- P(V=0) Over Time ---- */}
      <section style={section}>
        <h2 style={h2Style}>P(V=0) Estimate Over Time</h2>

        <p style={pStyle}>
          The marginal estimate of P(V=0) from the bootstrap V-flow during
          Phase II. Since we have the explicit distribution on V, we can
          directly track how the estimated null probability converges toward
          the true value ({run.nullProbTrue}) from the initial guess
          ({run.nullProbInit}).
        </p>

        <Figure
          src={img("pv0_over_time.png")}
          alt="P(V=0) estimate over training"
          caption={`Bootstrap P̂(V=0) during Phase II training. True P(V=0) = ${run.nullProbTrue}, initial guess = ${run.nullProbInit}.`}
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
            caption={`X samples from the true model (π with ${100 * run.nullProbTrue}% null marks)`}
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
          caption="Ground-truth π samples vs aggregate posterior (V, U) inferred by the two-flow architecture"
        />

        <h3 style={h3Style}>ρ marginal distribution</h3>
        <Figure
          src={img("rho_marginal.png")}
          alt="Rho marginal histogram"
          caption="Marginal histogram of ρ for V=1 (active) marks: true distribution vs flow-inferred posterior"
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

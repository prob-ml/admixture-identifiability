/* ------------------------------------------------------------------ */
/*  Case study metadata                                                */
/* ------------------------------------------------------------------ */

export interface PihatComponent {
  weight: number;
  mean: [number, number]; // [rho, log_sigma]
  std: [number, number];
}

export interface CaseStudy {
  /** Directory name under public/results/ */
  id: string;
  /** Short display label */
  label: string;
  /** URL-safe slug used in routes */
  slug: string;
  /** Support window */
  L: number;
  /** Visible window */
  T: number;
  /** True null probability P(V=0) */
  nullProbTrue: number;
  /** Initial guess null probability P(V=0) */
  nullProbInit: number;
  /** Pihat mixture components for U | V=1 */
  pihatComponents: PihatComponent[];
  /** Phase I training steps */
  phaseISteps: number;
  /** Phase II training steps */
  phaseIISteps: number;
  /** Random seed */
  seed: number;
}

export const CASES: CaseStudy[] = [
  {
    id: "case_v_flow_masked_np95_pihat95",
    label: "P̂(V=0) = 0.95",
    slug: "np95-pihat95",
    L: 3,
    T: 20,
    nullProbTrue: 0.95,
    nullProbInit: 0.95,
    pihatComponents: [
      { weight: 1.0, mean: [1.0, 0.0], std: [1.0, 1.0] },
    ],
    phaseISteps: 10_000,
    phaseIISteps: 150_000,
    seed: 42,
  },
  {
    id: "case_v_flow_masked_np95_pihat80",
    label: "P̂(V=0) = 0.80",
    slug: "np95-pihat80",
    L: 3,
    T: 20,
    nullProbTrue: 0.95,
    nullProbInit: 0.80,
    pihatComponents: [
      { weight: 1.0, mean: [1.5, -0.25], std: [0.8, 0.8] },
    ],
    phaseISteps: 10_000,
    phaseIISteps: 150_000,
    seed: 42,
  },
];

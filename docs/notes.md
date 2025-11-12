| Aspect                      | Plain SVGD                                                                                | SVGD + ADMM                                                                                                              |
| :-------------------------- | :---------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------- |
| **Constraint enforcement**  | Must encode physics in the likelihood; forward solves appear only through loss gradients. | ADMM enforces constraints explicitly via auxiliary variables (U,E,\Lambda); stabilizes ill-conditioned inverse problems. |
| **Numerical stability**     | Gradients depend directly on (A(m)^{-1}); can blow up near resonances or singularities.   | ADMM decouples (A(m)) and (U) updates, regularizing the gradient flow.                                                   |
| **Interpretability**        | Purely statistical; no explicit “data–physics consistency.”                               | Physically interpretable: each iteration enforces PDE consistency and posterior consistency.                             |
| **Parallelization**         | Particle updates only; may need repeated PDE solves per gradient.                         | ADMM lets you reuse PDE solvers and decouple variables across particles efficiently.                                     |
| **Convergence region**      | Sensitive to stiffness in the physical model.                                             | Broader stable region due to augmented Lagrangian penalty term.                                                          |
| **Posterior approximation** | Good if gradients are accurate.                                                           | Often better-conditioned posterior sampling for physics-constrained inverse problems.                                    |


Next steps:
 - start making the pure svgd script modular, separating the core svgd functionalitirs
 - slowly incorporate the svgd admm idea, building on the above modular implementation
 - how to check the new code? are there hyperparameters that it will revert to pure svgd?

| Parameter                      | Role                                                             | Limit → Effect                                                                                                                |
| :----------------------------- | :--------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| **μ (ADMM penalty)**           | Enforces constraint (A(m)U = B).                                 | **μ → 0:** constraint enforcement vanishes → the term (\frac{\mu}{2}|A(m)U-B|^2) disappears, so the dual update does nothing. |
| **Dual variable Λ**            | Stores constraint momentum.                                      | **Λ frozen at 0:** no ADMM correction, identical to standard SVGD likelihood gradient.                                        |
| **E (auxiliary variable)**     | Residual accumulation variable.                                  | **E fixed = 0:** disables residual correction → reverts to direct PDE solve (as in standard SVGD).                            |
| **Number of ADMM inner loops** | Controls how tightly constraints are solved per outer iteration. | **N_ADMM = 0 or 1:** effectively no constraint enforcement, so you only have the SVGD gradient update.                        |

| Stage               | Configuration                            | What to Observe                                                                         |
| :------------------ | :--------------------------------------- | :-------------------------------------------------------------------------------------- |
| ① Baseline          | μ = 0, Λ = 0, E = 0                      | Should match pure SVGD exactly.                                                         |
| ② Light coupling    | small μ (e.g., 1e−3), Λ initialized at 0 | Constraint violation should begin decreasing, minor difference from SVGD.               |
| ③ Moderate coupling | μ = 1e−1, Λ updated each step            | Convergence slower but residuals smaller, particles more stable.                        |
| ④ Strong coupling   | large μ (e.g., 10)                       | Constraints almost perfectly enforced, step size in m may need reduction for stability. |


For each iteration (and optionally per particle), log:

||A(m)U - B||₂ (constraint residual)

||Λ||₂ (dual norm)

||E||₂ (residual accumulator norm)

||φ(m)||₂ (SVGD field magnitude)

KSD (kernelized Stein discrepancy)

ΔL_aug (change in augmented Lagrangian)

posterior mean/variance

If ΔL_aug and KSD both → 0, and constraint residuals flatten, you can confidently say the method converged.

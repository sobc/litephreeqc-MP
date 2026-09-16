# Numerical Methods & Geochemical Formulation

## 1. Thermodynamic Fundamentals of Speciation

Geochemical equilibrium in aqueous systems is governed by the **Law of Mass Action** combined with fundamental conservation principles (mass and charge conservation).

### 1.1 Master and Secondary Species
The chemical system is spanned by $E$ master species (e.g., $\text{Ca}^{+2}$, $\text{CO}_3^{-2}$, $\text{SO}_4^{-2}$, $\text{H}^+$). Each secondary aqueous species $i \in \{1, \dots, S\}$ is formed via a reversible reaction involving master species:

$$\sum_{e=1}^E \nu_{ie} M_e \rightleftharpoons A_i$$

Applying the Law of Mass Action yields:

$$K_i = \frac{a_i}{\prod_{e=1}^E a_e^{\nu_{ie}}} \implies \ln m_i = \ln K_i + \sum_{e=1}^E \nu_{ie} \ln a_e - \ln \gamma_i$$

Where:
- $m_i$: Molality of secondary species $i$ (mol/kgw)
- $a_e$: Thermodynamic activity of master species $e$
- $\gamma_i$: Activity coefficient ($a_i = \gamma_i \cdot m_i$)
- $K_i$: Equilibrium formation constant

---

## 2. Activity Coefficient Modeling (Debye-Hückel)

Aqueous activity coefficients are evaluated using the extended Debye-Hückel (WATEQ / Truesdell-Jones) formulation ([`include/device_types.hpp`](file:///mnt/nfsshare/home/mluebke/phreeqc3/cpp_sycl/include/device_types.hpp)):

$$\log_{10} \gamma_i = -\frac{A \cdot z_i^2 \sqrt{I}}{1 + B \cdot a_i^0 \sqrt{I}} + b_i \cdot I$$

For uncharged species ($z_i = 0$), $\gamma_i = 1.0$ (or Setchénow relationship). Ionic strength $I$ is computed from all charged dissolved species:

$$I = \frac{1}{2} \sum_{i=1}^S z_i^2 \cdot m_i$$

Parameters at $25^\circ\text{C}$: $A \approx 0.5085$, $B \approx 0.3281 \times 10^8$.

---

## 3. Residual Equations & Equilibrium System

The non-linear algebraic system for a cell with $E$ elements and $P$ equilibrium minerals has dimension $E + P$:

```mermaid
graph TD
    subgraph NR_System ["Residual System R(x) = 0"]
        R_Mass["Element Mass Balances (e = 1 .. E-1)<br/>R_e = sum_i(nu_ie * m_i) + sum_p(alpha_pe * m_min,p) - T_e"]
        R_H["Charge or pH Balance (e = H+)<br/>Floating pH: R_charge = sum_i(z_i * m_i) - Delta_charge<br/>Fixed pH: R_pH = ln a_H - ln a_H,target"]
        R_Min["Mineral Equilibrium Constraints (p = 1 .. P)<br/>R_min,p = 1e-3 * (SI_p - SI_target,p)"]
    end
```

### 3.1 Primary Unknowns $x$
- $\ln a_e$ for $e = 1, \dots, E$ (Natural logs of master activities)
- $m_{\text{min}, p}$ for $p = 1, \dots, P$ (Moles of equilibrium mineral phases)

### 3.2 Residual Formulations $R(x)$
1. **Element Mass Balance** ($e \ne \text{H}^+$):
   $$R_e = \sum_{i=1}^S \nu_{ie} m_i + \sum_{p=1}^P \alpha_{pe} m_{\text{min}, p} - T_e = 0$$
   ($T_e$: Prescribed total element molality, $\alpha_{pe}$: Stoichiometric coefficient of element $e$ in mineral $p$).
2. **Charge Balance at Floating pH** ($e = \text{H}^+$):
   $$R_{\text{charge}} = \sum_{i=1}^S z_i m_i - \Delta_{\text{charge}} = 0$$
   ($\Delta_{\text{charge}}$: Target charge imbalance determined during initial equilibration).
3. **Fixed pH Mode** (e.g. Stage 1 speciation):
   $$R_{\text{pH}} = \ln a_{\text{H}^+} - \ln a_{\text{target}} = 0$$
4. **Mineral Equilibrium Constraints**:
   $$SI_p = \log_{10} IAP_p - \log_{10} K_{sp, p} = \sum_{e=1}^E \alpha_{pe} \frac{\ln a_e}{\ln 10} - \log_{10} K_{sp, p}$$
   - If $m_{\text{min}, p} > 0$ or `force_equality == true`:
     $$R_{E + p} = 10^{-3} \cdot (SI_p - SI_{\text{target}, p}) = 0$$
   - If undersaturated and mineral is exhausted: Row is deactivated ($R_{E+p} = 0, \Delta m_{\text{min}, p} = 0$).

---

## 4. Analytical Jacobian Formulation

The Jacobian matrix $J \in \mathbb{R}^{(E+P) \times (E+P)}$ with entries $J_{jk} = \frac{\partial R_j}{\partial x_k}$ is constructed analytically directly in thread registers:

1. **Element-Activity Block** ($j \in [0, E-1], k \in [0, E-1]$):
   $$\frac{\partial R_j}{\partial \ln a_k} = \sum_{i=1}^S \nu_{ji} \nu_{ki} m_i$$
2. **Charge Balance Row** ($j = \text{H}^+, k \in [0, E-1]$):
   $$\frac{\partial R_{\text{charge}}}{\partial \ln a_k} = \sum_{i=1}^S z_i \nu_{ki} m_i$$
3. **Element-Mineral Coupling** ($j \in [0, E-1], p \in [0, P-1]$):
   $$\frac{\partial R_j}{\partial m_{\text{min}, p}} = \alpha_{pj}$$
4. **Mineral-Activity Block** ($p \in [0, P-1], k \in [0, E-1]$):
   $$\frac{\partial R_{E+p}}{\partial \ln a_k} = 10^{-3} \cdot \frac{\alpha_{pk}}{\ln 10}$$

The linear system $J \cdot \Delta x = -R$ is inverted using partial pivoting.

### 4.1 Zero-Mass Element Handling (`has_mass[e]`)
In multi-dimensional reactive transport simulations, certain elements may be entirely absent in specific cells (e.g., pristine groundwater before tracer/mineral fronts arrive, such as $\text{Cl}^- = 0$ or $\text{Mg}^{+2} = 0$). Standard Newton-Raphson solvers construct zero rows and columns in the Jacobian when $T_e = 0$ and $m_{\text{min}, p} = 0$, leading to singular matrices and divergence.

To prevent singularity, the solver dynamically tracks element availability:
- An element $e$ has mass (`has_mass[e] == true`) if $e = \text{H}^+$, or $T_e > 10^{-16}$, or any present/forced mineral contains $e$.
- If `has_mass[e] == false`:
  - Its master activity is pinned to $\ln a_e = -40 \ln 10$.
  - Secondary species containing element $e$ are assigned zero molality ($m_i = 0$).
  - Its residual is set to zero ($R_e = 0$).
  - The Jacobian is decoupled by setting the diagonal entry $J_{ee} = 1.0$ and off-diagonals $J_{ek} = J_{ke} = 0.0$.
  - Update steps $\Delta \ln a_e$ are skipped.

This ensures the linear system remains strictly non-singular and well-conditioned without requiring dynamic reallocation or cell-by-cell matrix reconfiguration.

### 4.2 Step Relaxation & Convergence
To prevent divergence due to exponential terms, update steps are clamped:
$$\Delta \ln a_e = \text{clamp}(\Delta \ln a_e, -2.302, +2.302)$$
This bounds activity changes to at most one order of magnitude per Newton iteration.

Convergence criterion: $\|R\|_2 < 10^{-11}$.

---

## 5. Two-Stage Initial Equilibration Workflow

In realistic geochemical modeling scenarios, water analyses (e.g., pH = 7.0, $\text{Ca} = 1\,\text{mM}$, $\text{CO}_3 = 2\,\text{mM}$, $\text{SO}_4 = 3\,\text{mM}$) rarely satisfy electrical neutrality perfectly. PHREEQC accommodates this by establishing a permanent background charge imbalance during initialization.

The C++ backend reproduces this via a robust two-stage process ([`src/parser.cpp`](file:///mnt/nfsshare/home/mluebke/poet-MP/ext/litephreeqc-MP/src/parser.cpp)):

```mermaid
graph TD
    A["SystemInput (Raw input concentrations & initial pH)"] --> B["Stage 1: Speciation at FIXED pH (7.0, ignoring minerals P=0)"]
    B --> C["Compute background charge imbalance<br/>Delta_charge = sum_i(z_i * m_i)"]
    C --> D["Add mineral inventories into total element stocks<br/>T_e = T_e + alpha_pe * moles_min"]
    D --> E["Stage 2: Mineral equilibration at FLOATING pH<br/>Constraint: R_charge = sum_i(z_i * m_i) - Delta_charge == 0"]
    E --> F["Warm-start converged master activities<br/>state.master_activities[e, cell] = ln_act_stage2"]
    F --> G["Ready for parallel time-stepping kinetics"]
```

By **warm-starting** all grid cells with the converged activities from Stage 2, subsequent kinetic time steps typically converge in fewer than 3 Newton-Raphson iterations.

---

## 6. 2nd-Order Runge-Kutta (Heun) Adaptive Sub-stepping for Kinetics

Coupled reactive transport requires robust, accurate integration of kinetic mineral reactions over outer time steps $\Delta t$. Because mineral dissolution and precipitation rates non-linearly depend on saturation indices ($SR$), 1st-order explicit Euler integration can accumulate numerical drift across reaction fronts.

To achieve $\mathcal{O}(h^2)$ second-order temporal accuracy and exact error control with zero dynamic memory allocations on device/GPU, `solver_backend.cpp` implements an **embedded 2nd-Order Runge-Kutta (Heun's Predictor-Corrector) method** with physical rate-limiting and local truncation error estimation:

### 6.1 Two-Stage Runge-Kutta (Heun) Execution Cycle

```mermaid
flowchart TD
    Start["Start Sub-step (current h <= t_rem)"] --> S1_Eq["Stage 1: Solve Equilibrium Speciation at t_n<br/>(Yields a_1, SI_1, and SR_1)"]
    S1_Eq --> S1_Rate["Stage 1: Evaluate Rates k_1 via JIT<br/>k_1 = dynamic_rate(ctx_1)"]
    S1_Rate --> ActiveCheck{"Any active reaction?<br/>max |k_1| > 1e-15"}
    ActiveCheck -- No --> QuickAccept["Accept step instantly!<br/>h = min(1.5 * h, 1e30)"]
    ActiveCheck -- Yes --> Bounds1["Check Dissolution/Precipitation Bounds<br/>Reject if overshoot (exact h_deplete)"]
    Bounds1 --> S2_Pred["Stage 2: Form Predictor State<br/>T_pred = T_n + sum(nu * k_1)<br/>m_pred = m_n - k_1"]
    S2_Pred --> S2_Eq["Stage 2: Solve Equilibrium at Predictor State<br/>(Yields a_2, SI_2, and SR_2)"]
    S2_Eq --> S2_Rate["Stage 2: Evaluate Rates k_2 at Predictor<br/>k_2 = dynamic_rate(ctx_2)"]
    S2_Rate --> RK_Comb["Compute RK2 Heun Average & Error:<br/>r_RK2 = 0.5 * (k_1 + k_2)<br/>err = 0.5 * |k_1 - k_2|"]
    RK_Comb --> TolCheck{"Local Error <= tol (1e-8 mol)?"}
    TolCheck -- No --> RejRK["Reject step!<br/>h_retry = h * 0.8 * sqrt(tol / err)"]
    TolCheck -- Yes --> FinalSolve["Solve Final Corrector Equilibrium<br/>(Warm-started from Stage 2)"]
    FinalSolve --> Commit["Accept step, t_rem -= h<br/>h_next = min(h * 0.9 * sqrt(tol/err), max_allowed_h)"]
```

### 6.2 Mathematical Formulation & Error Estimation
1. **Stage 1 (Predictor Derivative $k_1$)**:
   $$\mathbf{k}_1 = \mathbf{R}(t_n, \mathbf{y}_n) \cdot h$$
2. **Predictor State $\tilde{\mathbf{y}}$**:
   $$\tilde{T}_e = T_e^n + \sum_{k} \nu_{ke} k_{1, k}, \quad \tilde{m}_k = \max(m_k^n - k_{1, k}, 0)$$
3. **Stage 2 (Corrector Derivative $k_2$)**:
   $$\mathbf{k}_2 = \mathbf{R}(t_n + h, \tilde{\mathbf{y}}) \cdot h$$
4. **Heun Combination**:
   $$r_{k, \text{RK2}} = \frac{1}{2} \left( k_{1, k} + k_{2, k} \right)$$
5. **Embedded Local Truncation Error**:
   $$\varepsilon_k = \left| r_{k, \text{Euler}} - r_{k, \text{RK2}} \right| = \frac{1}{2} \left| k_{1, k} - k_{2, k} \right|$$
   Tolerance: $\text{tol} = 10^{-8}\,\text{mol}$ (matching standard PHREEQC).
   - If $\max_k \varepsilon_k > \text{tol}$: Step is rejected and retried with $h_{\text{retry}} = h \cdot 0.8 \sqrt{\frac{\text{tol}}{\varepsilon_{\max}}}$.
   - If $\max_k \varepsilon_k \le \text{tol}$: Step is accepted and next step size adapts via $h_{\text{next}} = h \cdot 0.9 \sqrt{\frac{\text{tol}}{\varepsilon_{\max}}}$.

### 6.3 Physical Boundary Controls
- **Dissolution Overshoot Rejection**: For $m_k > 10^{-8}\,\text{mol}$, if $k_{1, k} > m_k \times 1.05$, the step is rejected and retried with $h_{\text{exact}} = h \cdot \frac{m_k}{k_{1, k}}$, ensuring exact depletion without premature exhaustion.
- **Precipitation Capacity Control**: For $SR > 1.0$, $x_{\max} = \min_{e} \frac{T_e}{\nu_{ke}} (1 - 1/SR)$. If $|k_1| > 1.25 x_{\max}$, step is rejected and scaled down.
- **Quiescent Cell Bypass**: If all kinetic rates $|k_{1, k}| < 10^{-15}$, Stage 2 is completely bypassed, allowing equilibrium cells to double step sizes without extra speciation calls.

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

### 4.1 Step Relaxation & Convergence
To prevent divergence due to exponential terms, update steps are clamped:
$$\Delta \ln a_e = \text{clamp}(\Delta \ln a_e, -2.302, +2.302)$$
This bounds activity changes to at most one order of magnitude per Newton iteration.

Convergence criterion: $\|R\|_2 < 10^{-11}$.

---

## 5. Two-Stage Initial Equilibration Workflow

In realistic geochemical modeling scenarios, water analyses (e.g., pH = 7.0, $\text{Ca} = 1\,\text{mM}$, $\text{CO}_3 = 2\,\text{mM}$, $\text{SO}_4 = 3\,\text{mM}$) rarely satisfy electrical neutrality perfectly. PHREEQC accommodates this by establishing a permanent background charge imbalance during initialization.

The C++ backend reproduces this via a robust two-stage process ([`src/parser.cpp`](file:///mnt/nfsshare/home/mluebke/phreeqc3/cpp_sycl/src/parser.cpp)):

```mermaid
graph TD
    A["SystemInput (Raw input concentrations & initial pH)"] --> B["Stage 1: Speciation at FIXED pH (7.0)"]
    B --> C["Compute background charge imbalance<br/>Delta_charge = sum_i(z_i * m_i)"]
    C --> D["Add mineral inventories into total element stocks<br/>T_e = T_e + alpha_pe * moles_min"]
    D --> E["Stage 2: Mineral equilibration at FLOATING pH<br/>Constraint: R_charge = sum_i(z_i * m_i) - Delta_charge == 0"]
    E --> F["Warm-start converged master activities<br/>state.master_activities[e, cell] = ln_act_stage2"]
    F --> G["Ready for parallel time-stepping kinetics"]
```

By **warm-starting** all grid cells with the converged activities from Stage 2, subsequent kinetic time steps typically converge in fewer than 3 Newton-Raphson iterations.

---

## 6. Adaptive Time Sub-stepping for Kinetics

Kinetic time steps $\Delta t$ are solved using an adaptive sub-stepping integrator ([`src/solver_backend.cpp`](file:///mnt/nfsshare/home/mluebke/phreeqc3/cpp_sycl/src/solver_backend.cpp)):

1. Initialize sub-step size $h = \frac{\Delta t}{10}$.
2. Solve equilibrium speciation before rate evaluation $\rightarrow$ yields activities $a$, saturation indices $SI$, and saturation ratios $SR$.
3. Evaluate kinetic rate $r$ via JIT-specialized function $\rightarrow$ reacted mass $\Delta m = r \cdot h$.
4. Formulate trial concentrations:
   $$T_e^{\text{trial}} = T_e + \nu_{\text{kin}, e} \cdot \Delta m$$
5. Solve equilibrium speciation with $T_e^{\text{trial}}$:
   - **Success**: Accept step: $t_{\text{rem}} \leftarrow t_{\text{rem}} - h$, increase step size $h \leftarrow \min(t_{\text{rem}}, 1.5 \cdot h)$.
   - **Failure**: Reject step: roll back state, shrink step size $h \leftarrow \frac{h}{2}$.

# SANS-Magnetic-Correlations

Python and Ubermag tool to model and simulate the **magnetic structure factor** $\mathfrak{S}_{\alpha\beta}(\mathbf{q})$ for Small-Angle Neutron Scattering (SANS) of magnetic nanoparticle systems.

The script computes the magnetic correlation functions for a pair of interacting nanoparticles under different interparticle exchange couplings via micromagnetic simulations. It then reconstructs the expected scattering intensities for both **unpolarized SANS** and **polarized SANS (Polaris $\pm$)**, evaluating the uncorrelated (NC) limit, the highly correlated limit, and the intermediate regimes using an Ornstein-Zernike correlation length $\xi$.

---

## Visuals

<div style="text-align: center;">
  <img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/3c4aef00-e550-470e-ab76-e3c5c6c060a3" />
</div>

**Figure 1:** 2D scattering patterns in the detector plane at $H = 20\,\mathrm{mT}$ and $H = 3\,\mathrm{T}$ for Polaris $\pm$ and unpolarized SANS. The interparticle exchange is varied ($A_{\mathrm{inter}} = -A_{\mathrm{bulk}}, 0, +A_{\mathrm{bulk}}$).

<div style="text-align: center;">
  <img width="4453" height="1592" alt="fig_polaris_radial_20mT_vs_3T" src="https://github.com/user-attachments/assets/529f0383-80c4-4edc-a60d-0f1494e8f19f" />
</div>

**Figure 2:** Polaris $\pm$ and unpolarized intensity at $q \in [0.2,\,3.0]\,\mathrm{nm}^{-1}$, evaluated at $H = 20\,\mathrm{mT}$ (left) and $H = 3\,\mathrm{T}$ (right). The non-correlated contribution (NC) is shown at $5\,\mathrm{K}$ (dashed black) and $300\,\mathrm{K}$ (dotted red); the correlated part is shown for $A_{\mathrm{inter}} = -A_{\mathrm{bulk}}$ (blue), $A_{\mathrm{inter}} = 0$ (orange) and $A_{\mathrm{inter}} = +A_{\mathrm{bulk}}$ (green), for $D/\xi = 0$ (solid thick) and $D/\xi = 0.01,\,0.05,\,0.10$ (thin lines). In the high field conditions, the dipolar overlaps $A>0$.

<div style="text-align: center;">
  <img width="4463" height="1682" alt="fig_polaris_angular_20mT_vs_3T (1)" src="https://github.com/user-attachments/assets/c85c860c-43fe-43d9-8a1f-347fe80d1476" />
</div>

**Figure 3:** Angular Polaris $\pm$ intensity $I(\theta)$ integrated over $q \in [0.75,\,0.85]\,\mathrm{nm}^{-1}$, evaluated at $H = 20\,\mathrm{mT}$ (left) and $H = 3\,\mathrm{T}$ (right). The non-correlated contribution (NC) is shown at $5\,\mathrm{K}$ (dashed black) and $300\,\mathrm{K}$ (dotted red); the correlated part is shown for $A_{\mathrm{inter}} = -A_{\mathrm{bulk}}$ (blue), $A_{\mathrm{inter}} = 0$ (orange) and $A_{\mathrm{inter}} = +A_{\mathrm{bulk}}$ (green), for $D/\xi = 0$ (solid thick) and $D/\xi = 0.01,\,0.05,\,0.10$ (thin lines).

---

## Physics Model

In a SANS experiment, the magnetic scattering cross-section depends on the polarization state of the incident and scattered neutrons. The general expression involves the Fourier transform of the magnetization $\tilde{\mathbf{M}}(\mathbf{q})$ and the Halpern-Johnson vector $\tilde{\mathbf{Q}}_\perp = \hat{\mathbf{q}} \times (\hat{\mathbf{q}} \times \tilde{\mathbf{M}})$. 

The magnetic structure factor is defined as:

$$
\mathfrak{S}_{\alpha\beta}(\mathbf{q}) = \frac{1}{N} \sum_{i=1}^N \sum_{j=1}^N \tilde{F}_\alpha^{(i)}(\mathbf{q}) \tilde{F}_\beta^{*(j)}(\mathbf{q}) e^{i \mathbf{q} \cdot (\mathbf{r}_i - \mathbf{r}_j)}
$$

where $\tilde{F}_\alpha^{(i)}(\mathbf{q})$ is the normalized Fourier transform of the magnetization of particle $i$.

### 1. Unpolarized SANS

For an unpolarized incident beam, the magnetic intensity is obtained by taking the trace of the Halpern-Johnson matrix $B(\mathbf{q})$ and the magnetic structure factor:

$$
I_{\mathrm{unpol}}(\mathbf{q}) = N M_s^2 V_p^2 \mathrm{Tr}[B(\mathbf{q}) \cdot \mathfrak{S}(\mathbf{q})]
$$

where $B(\mathbf{q}) = \mathbb{I} - \hat{\mathbf{q}}\hat{\mathbf{q}}^T$ projects the magnetization onto the plane perpendicular to $\mathbf{q}$. The unpolarized intensity explicitly reads:

$$
I_{\mathrm{unpol}}(\mathbf{q}) \propto \mathfrak{S}_{xx} + \mathfrak{S}_{yy} \cos^2\theta + \mathfrak{S}_{zz} \sin^2\theta - 2\mathfrak{S}_{yz} \sin\theta \cos\theta
$$

### 2. Polarized SANS (Polaris $\pm$)

For polarized neutrons, the spin-dependent cross-sections resolve the magnetic scattering into **Non-Spin-Flip (NSF, $\pm\pm$)** and **Spin-Flip (SF, $\pm\mp$)** channels. The SF channel isolates the transversal magnetization components and is free of nuclear coherent scattering. 

The SF cross-section is given by:

$$
\frac{d\Sigma^{\pm\mp}}{d\Omega}(\mathbf{q}) = \frac{8\pi^3}{V} b_H^2 \bigg( |\tilde{M}_x|^2 + |\tilde{M}_y|^2 \cos^4\theta + |\tilde{M}_z|^2 \sin^2\theta \cos^2\theta - (\tilde{M}_y \tilde{M}_z^* + \tilde{M}_y^* \tilde{M}_z) \sin\theta \cos^3\theta \mp i\chi \bigg)
$$

where the chiral function $\chi(\mathbf{q})$ is:

$$
\chi(\mathbf{q}) = \left(\tilde{M}_x \tilde{M}_y^* - \tilde{M}_x^* \tilde{M}_y\right) \cos^2\theta - \left(\tilde{M}_x \tilde{M}_z^* - \tilde{M}_x^* \tilde{M}_z\right) \sin\theta \cos\theta
$$

In the script, the matrices $B_{\alpha\beta}$ for the Polaris channels are constructed explicitly and dotted with the correlation tensors to reconstruct the 2D and angular patterns.

### 3. Correlation Model

The code separates the structure factor into a self-correlation $C_{\alpha\beta}^{\mathrm{self}}(\mathbf{q})$ and a cross-correlation between particles in contact $C_{\alpha\beta}^{D}(\mathbf{q})$. An Ornstein-Zernike type exponential decay is used to interpolate between the uncorrelated limit ($\xi \to 0$) and the highly correlated limit ($D/\xi \to 0$):

$$
C_{\alpha\beta}(\mathbf{q},r) = C_{\alpha\beta}^{\infty}(\mathbf{q}) + e^{-\frac{r-D}{\xi}} \left[ C_{\alpha\beta}^D(\mathbf{q}) - C_{\alpha\beta}^{\infty}(\mathbf{q}) \right]
$$

Under this model, the structure factor for the uncorrelated (NC) system is:

$$
\mathfrak{S}_{\alpha\beta}^{\mathrm{NC}}(\mathbf{q}) = C_{\alpha\beta}^{\mathrm{self,NC}}(\mathbf{q}) + [S(\mathbf{q}) - 1]C_{\alpha\beta}^{\infty}(\mathbf{q})
$$

And for the highly correlated system:

$$
\mathfrak{S}_{\alpha\beta}^{C}(\mathbf{q}) \approx C_{\alpha\beta}^{\mathrm{self,C}}(\mathbf{q}) + [S(\mathbf{q}) - 1]C_{\alpha\beta}^{\infty}(\mathbf{q}) + \Delta C_{\alpha\beta}^{D,\infty}(\mathbf{q}) e^{\frac{D}{\xi}} [B(\mathbf{q},\xi) + S(\mathbf{q}) - 1]
$$

where $S(\mathbf{q})$ is the structural structure factor (assumed hard spheres), and $B(\mathbf{q},\xi)$ is the Fourier transform of the correlation function.

---

## Requirements

- Python 3.9+
- [Ubermag](https://ubermag.github.io/) (`oommfc`, `discretisedfield`, `micromagneticmodel`)
- OOMMF installed and available in `PATH`
- `numpy`, `scipy`, `matplotlib`, `tqdm`

```bash
pip install -r requirements.txt

#!/usr/bin/env python
# coding: utf-8
"""
Polarized and unpolarized SANS magnetic structure factor reconstruction.

This script simulates a pair of interacting magnetic nanoparticles using 
micromagnetic simulations (Ubermag/OOMMF) and calculates the self- and 
cross-correlations of their magnetization Fourier transforms. 

It then reconstructs the 2D, radial, and angular scattering intensities 
for both unpolarized SANS and polarized SANS (Polaris ±) channels, 
evaluating different interparticle exchange values and correlation lengths.
"""

import os
import glob
import shutil
import gc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
from scipy.special import spherical_jn
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import oommfc as mc
import discretisedfield as df
import micromagneticmodel as mm

# ============================================================
# 1. PHYSICAL AND MESH PARAMETERS
# ============================================================
n0     = 0.8388e-9
d      = 10 * n0
s      = 0 * n0
shell  = 0 * 1e-9
cell   = (n0/3, n0/3, n0/3)
rho    = 5170
MW     = 234.37
Na     = 6.022e23
n      = Na * rho * 1000 / MW
Ms     = 2.6 * mm.consts.muB * n
alpha  = 0.5
K_core = 1e4
K_shell= 5e6
H0     = 20e-3 / mm.consts.mu0
A_bulk = (mm.consts.kB * 10) / (12 * 0.24 * n0/3)

# Contact zone (0 to deactivate)
contact_range = 0 * 4 * n0 / 3
z_contact     = d

# Geometry: Pair on the +z axis
z_NP1_c = d/2
z_NP2_c = 3*d/2 + s

region = df.Region(p1=(-d/2, -d/2, 0), p2=(d/2, d/2, 2*d + s))
subregions = {
    'NP1': df.Region(p1=(-d/2, -d/2, 0),     p2=(d/2, d/2, d)),
    'NP2': df.Region(p1=(-d/2, -d/2, d + s), p2=(d/2, d/2, 2*d + s)),
}
if s > 0:
    subregions['spacer'] = df.Region(p1=(-d/2, -d/2, d), p2=(d/2, d/2, d + s))

mesh   = df.Mesh(region=region, cell=cell, subregions=subregions)
Ncells = len(mesh) / 2
norma  = (Ms * np.pi/6 * Ncells)**2

# ============================================================
# 2. GEOMETRY AND ANISOTROPY HELPERS
# ============================================================
def in_NP1(x, y, z):
    return x**2 + y**2 + (z - z_NP1_c)**2 < (d/2)**2

def in_NP2(x, y, z):
    return x**2 + y**2 + (z - z_NP2_c)**2 < (d/2)**2

def in_core_NP1(x, y, z):
    return x**2 + y**2 + (z - z_NP1_c)**2 < (d/2 - shell)**2

def in_core_NP2(x, y, z):
    return x**2 + y**2 + (z - z_NP2_c)**2 < (d/2 - shell)**2

def in_contact_zone(z):
    return contact_range > 0 and abs(z - z_contact) < contact_range

def norm_fun(pos):
    x, y, z = pos
    return Ms if (in_NP1(x, y, z) or in_NP2(x, y, z)) else 0

def K_fun(pos):
    x, y, z = pos
    if in_contact_zone(z) and (in_NP1(x, y, z) or in_NP2(x, y, z)):
        return K_core
    if in_core_NP1(x, y, z) or in_core_NP2(x, y, z):
        return K_core
    if in_NP1(x, y, z) or in_NP2(x, y, z):
        return K_shell
    return 0

def u_fun(pos, u1, u2, u_c):
    x, y, z = pos
    if in_contact_zone(z) and (in_NP1(x, y, z) or in_NP2(x, y, z)):
        return u_c
    if in_core_NP1(x, y, z) or in_core_NP2(x, y, z):
        if contact_range > 0:
            return u_c
        return u1 if in_core_NP1(x, y, z) else u2
    return np.random.uniform(-1, 1, size=3)

# ============================================================
# 3. GENERAL UTILITIES
# ============================================================
def random_direction():
    t = np.random.uniform(0, 2*np.pi)
    p = np.arccos(2*np.random.uniform(0, 1) - 1)
    return np.array([np.sin(p)*np.cos(t), np.sin(p)*np.sin(t), np.cos(p)])

def cleanup_temp_files(system_name):
    if os.path.exists(system_name):
        shutil.rmtree(system_name, ignore_errors=True)
    for ext in ['*.ovf','*.omf','*.log','*.out','*.err','*.pkl']:
        for f in glob.glob(ext):
            try: os.remove(f)
            except: pass
    for td in ['temp_oommf','__pycache__','oommf_temp']:
        if os.path.exists(td):
            shutil.rmtree(td, ignore_errors=True)

def rotate_to_field_frame(ky_lab, kz_lab, phi, C_dict, ky_p, kz_p):
    U, V = np.meshgrid(ky_p, kz_p, indexing='ij')
    qy_lab =  U*np.cos(phi) + V*np.sin(phi)
    qz_lab = -U*np.sin(phi) + V*np.cos(phi)
    pts = np.column_stack([qy_lab.ravel(), qz_lab.ravel()])
    out = {}
    for name, arr in C_dict.items():
        ri = RegularGridInterpolator((ky_lab, kz_lab), arr.real, bounds_error=False, fill_value=0.0)
        ii = RegularGridInterpolator((ky_lab, kz_lab), arr.imag, bounds_error=False, fill_value=0.0)
        out[name] = (ri(pts) + 1j*ii(pts)).reshape(U.shape)
    return out

# ============================================================
# 4. MICROMAGNETIC SIMULATION CORE
# ============================================================
C_NAMES = ['Cxx', 'Cxy', 'Cxz', 'Cyy', 'Cyz', 'Czz']

def simulate(A_intra, n_phi=15, config_per_phi=8, verbose_check=True):
    phi_values = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    ky_lab = kz_lab = None
    ky_p   = kz_p   = None
    C_self_out  = {k: None for k in C_NAMES}
    C_cross_out = {k: None for k in C_NAMES}
    n_phi_used  = 0
    sim_count   = 0

    for phi in tqdm(phi_values, desc=f"A/Abulk={A_intra/A_bulk:+.2f}", leave=True):
        H_field = (0.0, H0*np.sin(phi), H0*np.cos(phi))
        C_self_phi  = {k: None for k in C_NAMES}
        C_cross_phi = {k: None for k in C_NAMES}

        for _ in range(config_per_phi):
            u1  = random_direction()
            u2  = random_direction()
            u_c = random_direction()

            u = df.Field(mesh, nvdim=3, value=lambda pos: u_fun(pos, u1, u2, u_c))
            K = df.Field(mesh, nvdim=1, value=K_fun)

            system_name = f'temp_polaris_{sim_count}'; sim_count += 1
            system = mm.System(name=system_name)
            system.m = df.Field(mesh, nvdim=3, value=lambda pos: np.random.uniform(-1, 1, size=3), norm=norm_fun, valid="norm")

            A_dict = {"NP1": A_bulk, "NP2": A_bulk, "NP1:NP2": A_intra}
            if s > 0: A_dict["spacer"] = 0

            system.energy = (mm.UniaxialAnisotropy(K=K, u=u) + mm.Exchange(A=A_dict) + mm.Zeeman(H=H_field) + mm.Demag())
            md = mc.MinDriver(); md.drive(system, verbose=False)

            m1 = system.m["NP1"]; m2 = system.m["NP2"]
            region2 = df.Region(p1=(-d/2, -d/2, -d/2), p2=(d/2, d/2, d/2))
            mesh2   = df.Mesh(region=region2, cell=cell)
            c1 = np.array(m1.mesh.region.center)
            c2 = np.array(m2.mesh.region.center)
            field1 = df.Field(mesh2, nvdim=3, value=lambda pos: m1(tuple(np.array(pos) + c1)))
            field2 = df.Field(mesh2, nvdim=3, value=lambda pos: m2(tuple(np.array(pos) + c2)))

            pad_aux = 250
            pad_width = {"x": (pad_aux, pad_aux), "y": (pad_aux, pad_aux), "z": (pad_aux, pad_aux)}
            fft1 = field1.pad(pad_width, mode='constant').fftn().sel(k_x=0)
            fft2 = field2.pad(pad_width, mode='constant').fftn().sel(k_x=0)

            if ky_lab is None:
                ky_lab = np.asarray(fft1.mesh.cells.k_y)
                kz_lab = np.asarray(fft1.mesh.cells.k_z)
                ky_p   = ky_lab.copy()
                kz_p   = kz_lab.copy()

            if C_self_phi['Cxx'] is None:
                for k in C_NAMES:
                    C_self_phi[k]  = np.zeros((len(ky_lab), len(kz_lab)), dtype=complex)
                    C_cross_phi[k] = np.zeros((len(ky_lab), len(kz_lab)), dtype=complex)

            fx1 = fft1.ft_x.array[:,:,0]; fy1 = fft1.ft_y.array[:,:,0]; fz1 = fft1.ft_z.array[:,:,0]
            fx2 = fft2.ft_x.array[:,:,0]; fy2 = fft2.ft_y.array[:,:,0]; fz2 = fft2.ft_z.array[:,:,0]

            C_self_phi['Cxx'] += fx1*np.conj(fx1); C_self_phi['Cxy'] += fx1*np.conj(fy1); C_self_phi['Cxz'] += fx1*np.conj(fz1)
            C_self_phi['Cyy'] += fy1*np.conj(fy1); C_self_phi['Cyz'] += fy1*np.conj(fz1); C_self_phi['Czz'] += fz1*np.conj(fz1)
            
            C_cross_phi['Cxx'] += fx1*np.conj(fx2); C_cross_phi['Cxy'] += fx1*np.conj(fy2); C_cross_phi['Cxz'] += fx1*np.conj(fz2)
            C_cross_phi['Cyy'] += fy1*np.conj(fy2); C_cross_phi['Cyz'] += fy1*np.conj(fz2); C_cross_phi['Czz'] += fz1*np.conj(fz2)

            del system, md, fft1, fft2, field1, field2
            cleanup_temp_files(system_name); gc.collect()

        for k in C_NAMES:
            C_self_phi[k]  /= config_per_phi
            C_cross_phi[k] /= config_per_phi

        C_self_rot  = rotate_to_field_frame(ky_lab, kz_lab, phi, C_self_phi,  ky_p, kz_p)
        C_cross_rot = rotate_to_field_frame(ky_lab, kz_lab, phi, C_cross_phi, ky_p, kz_p)

        if C_self_out['Cxx'] is None:
            for k in C_NAMES:
                C_self_out[k]  = np.zeros_like(C_self_rot[k])
                C_cross_out[k] = np.zeros_like(C_cross_rot[k])

        for k in C_NAMES:
            C_self_out[k]  += C_self_rot[k]
            C_cross_out[k] += C_cross_rot[k]
        n_phi_used += 1

    for k in C_NAMES:
        C_self_out[k]  /= n_phi_used
        C_cross_out[k] /= n_phi_used

    return ky_p, kz_p, C_self_out, C_cross_out

# ============================================================
# 5. POST-PROCESSING AND PLOTTING
# ============================================================
def S_q_hardsphere(q, phi, RHS):
    A_ = 2*RHS*q
    a = (1+2*phi)**2/(1-phi)**4
    b = -6*phi*(1+phi/2)**2/(1-phi)**4
    g = phi*a/2
    t1 = a*(np.sin(A_)-A_*np.cos(A_))/A_**2
    t2 = b*(2*A_*np.sin(A_)+(2-A_**2)*np.cos(A_)-2)/A_**3
    t3 = g*(-A_**4*np.cos(A_)+4*((3*A_**2-6)*np.cos(A_)+(A_**3-6*A_)*np.sin(A_)+6))/A_**5
    return 1/(1+24*phi*(t1+t2+t3)/A_)

def P_sphere(q, R):
    qR = q * R
    with np.errstate(invalid='ignore', divide='ignore'):
        val = 9.0 * (spherical_jn(1, qR) / qR)**2
    return np.where(qR == 0, 1.0, val)

def langevin(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[small]  = x[small]/3.0 - x[small]**3/45.0
    out[~small] = 1.0/np.tanh(x[~small]) - 1.0/x[~small]
    return out

def radial_average(qx, qy, field, n_bins=500):
    q_flat = np.sqrt(qx**2 + qy**2).ravel()
    I_mean, edges, _ = binned_statistic(q_flat, field.ravel(), statistic='mean', bins=n_bins)
    return 0.5*(edges[1:] + edges[:-1]), I_mean

def angular_average(KY, KZ, field, deg_per_bin=2.0, q_min=None, q_max=None):
    q_mag = np.sqrt(KY**2 + KZ**2) * (2*np.pi)
    mask = np.ones_like(q_mag, dtype=bool)
    if q_min is not None: mask &= (q_mag >= q_min)
    if q_max is not None: mask &= (q_mag <= q_max)
    theta_flat = np.arctan2(KY, KZ)[mask].ravel() % (2*np.pi)
    I_flat     = field[mask].ravel()
    n_bins = int(round(360.0 / deg_per_bin))
    edges  = np.linspace(0.0, 2*np.pi, n_bins + 1)
    I_mean, _, _ = binned_statistic(theta_flat, I_flat, 'mean',  bins=edges)
    I_std,  _, _ = binned_statistic(theta_flat, I_flat, 'std',   bins=edges)
    N,      _, _ = binned_statistic(theta_flat, I_flat, 'count', bins=edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        err = I_std / np.sqrt(N)
    bad = (~np.isfinite(err)) | (N < 2)
    I_mean = np.where(np.isfinite(I_mean), I_mean, 0.0)
    err    = np.where(bad, np.nan, err)
    return np.degrees(0.5*(edges[1:] + edges[:-1])), I_mean, err

# ============================================================
# MAIN ROUTINE
# ============================================================
if __name__ == "__main__":
    
    # --- 1. Run Simulations ---
    A_values = [-A_bulk, 0.0, A_bulk]
    n_phi, config_per_phi = 15, 8
    results = {}
    
    for A_intra in A_values:
        ky_p, kz_p, C_self, C_cross = simulate(A_intra, n_phi=n_phi, config_per_phi=config_per_phi)
        results[A_intra] = dict(ky_p=ky_p, kz_p=kz_p, C_self=C_self, C_cross=C_cross)

    out_dir = "data_polaris_azimuthal_C_H0_lexggD"
    os.makedirs(out_dir, exist_ok=True)
    KY_p, KZ_p = np.meshgrid(ky_p, kz_p, indexing='ij')

    save_dict = {
        'KY_p': KY_p, 'KZ_p': KZ_p, 'A_values': np.array(A_values),
        'A_over_Abulk': np.array([a/A_bulk for a in A_values]), 'A_bulk': np.array(A_bulk),
        'd': np.array(d), 's': np.array(s), 'Ms': np.array(Ms), 'shell': np.array(shell),
        'K_core': np.array(K_core), 'K_shell': np.array(K_shell),
        'contact_range': np.array(contact_range), 'z_contact': np.array(z_contact),
        'z_NP1_c': np.array(z_NP1_c), 'z_NP2_c': np.array(z_NP2_c),
        'norma': np.array(norma), 'C_names': np.array(C_NAMES),
        'n_phi': np.array(n_phi), 'config_per_phi': np.array(config_per_phi),
    }
    for i, A_intra in enumerate(A_values):
        for cn in C_NAMES:
            save_dict[f'self_{i}_{cn}']  = results[A_intra]['C_self'][cn]
            save_dict[f'cross_{i}_{cn}'] = results[A_intra]['C_cross'][cn]

    np.savez(os.path.join(out_dir, "C_components.npz"), **save_dict)
    print(f"Saved to {out_dir}/C_components.npz")

    # --- 2. Load Data and Post-Process ---
    data = np.load("data_polaris_azimuthal_C_H0_lexggD/C_components.npz", allow_pickle=True)
    A_over_Abulk = data['A_over_Abulk']
    d            = float(data['d'])
    norma        = float(data['norma'])
    Ms_data      = float(data['Ms'])
    theta        = np.arctan2(KY_p, KZ_p)

    # Physical parameters for post-processing
    phi_pack = 0.4
    R_p      = d / 2.0
    V_p      = 4.0/3.0 * np.pi * R_p**3
    rho_num  = phi_pack / V_p
    D_part   = d

    mu0   = 4*np.pi*1e-7
    H_si  = 20e-3/mm.consts.mu0
    T     = 5.0
    kB    = 1.380649e-23
    x_lang = mu0 * V_p * Ms_data * H_si / (kB * T)
    
    q_2D = np.sqrt(KY_p**2 + KZ_p**2)*(2*np.pi); q_2D[q_2D == 0] = 1e-12
    Sq_2D = S_q_hardsphere(q_2D, phi_pack, R_p)
    Pq_2D = P_sphere(q_2D, R_p)

    L_x = float(langevin(x_lang))
    m_zz = 1.0 - 2.0*L_x/x_lang if abs(x_lang) >= 1e-6 else 1.0/3.0
    m_xx = (1.0 - m_zz)/2.0
    m_yy = m_xx
    m_mat   = np.diag([m_xx, m_yy, m_zz])
    m_outer = np.zeros((3,3)); m_outer[2,2] = L_x**2

    # Matrices B
    Bxx = 1.0; Bxy = np.cos(theta)**2; Bxz = -np.sin(theta)*np.cos(theta)
    Byy = np.cos(theta)**4; Byz = -np.sin(theta)*np.cos(theta)**3; Bzz = np.cos(theta)**2 * np.sin(theta)**2

    def build_polaris(C):
        return (Bxx*C['Cxx'].real + Byy*C['Cyy'].real + Bzz*C['Czz'].real + 2.0*Byz*C['Cyz'].real + 2.0*Bxz*C['Cxz'].imag + 2.0*Bxy*C['Cxy'].imag)

    def build_unpolarized(C):
        return (C['Cxx'].real + np.cos(theta)**2 * C['Cyy'].real + np.sin(theta)**2 * C['Czz'].real - 2.0*np.sin(theta)*np.cos(theta) * C['Cyz'].real)

    idx_map = {'Cxx': (0,0), 'Cxy': (0,1), 'Cxz': (0,2), 'Cyy': (1,1), 'Cyz': (1,2), 'Czz': (2,2)}
    def C_from_mat(M, pref): return {k: pref*M[i,j] for k,(i,j) in idx_map.items()}

    C_inf = C_from_mat(m_outer, Pq_2D)
    C_NC_self = C_from_mat(m_mat, Pq_2D)

    q_max = q_2D.max()
    q_1d  = np.linspace(q_max/10000.0, q_max, 10000)
    S_1d  = S_q_hardsphere(q_1d, phi_pack, R_p)
    dS    = np.gradient(S_1d, q_1d)
    d2S   = np.gradient(dS, q_1d)
    lapS_1d = d2S + 2.0/q_1d * dS
    lapS_2D = np.interp(q_2D, q_1d, lapS_1d)
    b_q_2D = (2.0/np.pi**2)*(1.0 - Sq_2D) + (4.0/(3.0*D_part**2))*lapS_2D

    def B_q_xi(xi): return (rho_num * 8.0*np.pi * xi**3 / (1.0 + (q_2D*xi)**2)**2 + (D_part/xi) * b_q_2D)

    def intensities_NC():
        return build_polaris(C_NC_self) + (Sq_2D - 1)*build_polaris(C_inf), build_unpolarized(C_NC_self) + (Sq_2D - 1)*build_unpolarized(C_inf)

    def intensities_inf(Cs, Cc):
        Cs_n = {k: Cs[k]/norma for k in C_NAMES}; Cc_n = {k: Cc[k]/norma for k in C_NAMES}
        return build_polaris(Cs_n) + (Sq_2D - 1)*build_polaris(Cc_n), build_unpolarized(Cs_n) + (Sq_2D - 1)*build_unpolarized(Cc_n)

    def intensities_inter(Cs, Cc, D_over_xi):
        xi = D_part / D_over_xi
        weight = np.exp(D_over_xi) * (B_q_xi(xi) + Sq_2D - 1)
        Cs_n = {k: Cs[k]/norma for k in C_NAMES}; Cc_n = {k: Cc[k]/norma for k in C_NAMES}
        dC_n = {k: Cc_n[k] - C_inf[k] for k in C_NAMES}
        return build_polaris(Cs_n) + (Sq_2D - 1)*build_polaris(C_inf) + weight*build_polaris(dC_n), build_unpolarized(Cs_n) + (Sq_2D - 1)*build_unpolarized(C_inf) + weight*build_unpolarized(dC_n)

    # --- 3. Generate Plots ---
    D_over_xi_list = [0.01, 0.05, 0.1]
    I_NC_p, I_NC_u = intensities_NC()

    # 2D Plots
    out_dir_2D = "figs_polaris_2D_H0_lexggD"; os.makedirs(out_dir_2D, exist_ok=True)
    for i, a_ratio in enumerate(A_over_Abulk):
        Cs = {k: data[f'self_{i}_{k}'] for k in C_NAMES}
        Cc = {k: data[f'cross_{i}_{k}'] for k in C_NAMES}
        I_p = build_polaris(Cs)/norma + (build_polaris(Cc)/norma)*(Sq_2D - 1)
        I_u = build_unpolarized(Cs)/norma + (build_unpolarized(Cc)/norma)*(Sq_2D - 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, I, title in zip(axes, [np.where(I_p > 0, I_p, np.nan), np.where(I_u > 0, I_u, np.nan)],
                                [r"Polaris $\pm$" + f"   $A/A_{{bulk}}$={a_ratio:+.0f}", f"SANS no pol.  $A/A_{{bulk}}$={a_ratio:+.0f}"]):
            im = ax.pcolormesh(KY_p*1e-9*(2*np.pi), KZ_p*1e-9*(2*np.pi), I, shading='auto', cmap='viridis', norm=LogNorm(vmin=1e-4, vmax=1))
            ax.set_xlabel(r"k$_y$ (nm$^{-1}$)"); ax.set_ylabel(r"k$_z$ (nm$^{-1}$)"); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_title(title)
            plt.colorbar(im, ax=ax, label=r'$I_\mathrm{norm}$')
        plt.tight_layout()
        tag = f"{a_ratio:+.0f}".replace('+', 'p')
        fig.savefig(os.path.join(out_dir_2D, f"fig2D_AoverAbulk_{tag}_H0_lexggD.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # Radial Plots
    out_dir_rad = "radial_H0_lexggD"; os.makedirs(out_dir_rad, exist_ok=True)
    fig, axes = plt.subplots(len(A_over_Abulk), 2, figsize=(12, 4*len(A_over_Abulk)), sharex=True)
    if len(A_over_Abulk) == 1: axes = axes[np.newaxis, :]
    for i, a_ratio in enumerate(A_over_Abulk):
        Cs = {k: data[f'self_{i}_{k}'] for k in C_NAMES}; Cc = {k: data[f'cross_{i}_{k}'] for k in C_NAMES}
        ax_p, ax_u = axes[i, 0], axes[i, 1]
        
        for I_plot, ax in [(I_NC_p, ax_p), (I_NC_u, ax_u)]:
            q_r, I_r = radial_average(KY_p, KZ_p, I_plot); ax.plot(q_r*1e-9*(2*np.pi), I_r, 'k--', lw=1, label='NC')
        
        I_p, I_u = intensities_inf(Cs, Cc)
        for I_plot, ax in [(I_p, ax_p), (I_u, ax_u)]:
            q_r, I_r = radial_average(KY_p, KZ_p, I_plot); ax.plot(q_r*1e-9*(2*np.pi), I_r, 'k-', lw=3, label=r'D/$\xi$=0')

        for j, Dxi in enumerate(D_over_xi_list):
            I_p, I_u = intensities_inter(Cs, Cc, Dxi)
            for I_plot, ax in [(I_p, ax_p), (I_u, ax_u)]:
                q_r, I_r = radial_average(KY_p, KZ_p, I_plot); ax.plot(q_r*1e-9*(2*np.pi), I_r, '-', color=f'C{j}', lw=1, label=f'D/ξ={Dxi}')
        
        for ax in [ax_p, ax_u]:
            ax.set_yscale('log'); ax.set_xlim(0.3, 3.0); ax.set_ylim(1e-7, 1e0); ax.grid(True); ax.legend(fontsize=8)
        ax_p.set_ylabel(f'$I(q)$   $A/A_b={a_ratio:+.0f}$'); ax_u.set_ylabel(f'$I(q)$   $A/A_b={a_ratio:+.0f}$')
    axes[0, 0].set_title(r'Polaris $\pm$'); axes[0, 1].set_title('Unpolarized')
    axes[-1, 0].set_xlabel(r'q (nm$^{-1}$)'); axes[-1, 1].set_xlabel(r'q (nm$^{-1}$)')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir_rad, "radial_comparativo.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Angular Plots
    out_dir_ang = "angular_H0_lexggD"; os.makedirs(out_dir_ang, exist_ok=True)
    q_min_ang = 0.65 / 1e-9 * (2*np.pi)
    fig, axes = plt.subplots(len(A_over_Abulk), 1, figsize=(8, 4*len(A_over_Abulk)), sharex=True)
    if len(A_over_Abulk) == 1: axes = [axes]
    for i, a_ratio in enumerate(A_over_Abulk):
        Cs = {k: data[f'self_{i}_{k}'] for k in C_NAMES}; Cc = {k: data[f'cross_{i}_{k}'] for k in C_NAMES}
        ax = axes[i]
        
        th, Im, Ie = angular_average(KY_p, KZ_p, I_NC_p, q_min=q_min_ang); ax.errorbar(th, Im, yerr=Ie, fmt='k--', lw=1, capsize=2, label='NC')
        I_p, _ = intensities_inf(Cs, Cc); th, Im, Ie = angular_average(KY_p, KZ_p, I_p, q_min=q_min_ang); ax.errorbar(th, Im, yerr=Ie, fmt='k-', lw=3, capsize=2, label=r'D/$\xi$=0')
        
        for j, Dxi in enumerate(D_over_xi_list):
            I_p, _ = intensities_inter(Cs, Cc, Dxi); th, Im, Ie = angular_average(KY_p, KZ_p, I_p, q_min=q_min_ang); ax.errorbar(th, Im, yerr=Ie, fmt='-', color=f'C{j}', lw=1, capsize=2, label=f'D/ξ={Dxi}')
        
        ax.set_ylabel(r'$\langle I(\theta)\rangle$'); ax.set_title(r'Polaris $\pm$   $A/A_b=$' + f'{a_ratio:+.0f}')
        ax.set_xlim(0, 360); ax.set_xticks(np.arange(0, 361, 45)); ax.set_yscale('log'); ax.set_ylim(1e-7, 1e-5); ax.grid(True, which='both', alpha=0.5); ax.legend(fontsize=8)
    axes[-1].set_xlabel(r"$\theta$ (deg)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir_ang, "angular_comparativo.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("All simulations and plots generated successfully.")

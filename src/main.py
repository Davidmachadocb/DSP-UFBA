import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import dsp
import os


# style for plot
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'stix', 
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.figsize': (14, 5),
    'figure.dpi': 300
})

if __name__ == "__main__":

    if not os.path.exists("build"):
        os.makedirs("build")

    #parameters
    fs = 10000              
    N_sym = 3000            
    delta_f_real = 200
    delta_nu = 0.5
    Ts = 1 / fs

    #phase Noise
    sigma_phase = np.sqrt(2 * np.pi * delta_nu * Ts)
    delta_theta = np.random.normal(loc=0.0, scale=sigma_phase, size=N_sym)
    theta_k = np.cumsum(delta_theta)
    
    #symbols
    ints = np.random.randint(0, 4, N_sym)
    phases = (np.pi/4) + (ints * np.pi/2)
    s = np.exp(1j * phases)

    #add stuff to the pure
    t = np.arange(N_sym) / fs
    freq_offset_phasor = np.exp(1j * 2 * np.pi * delta_f_real * t)
    
    y_received = s * np.exp(1j * theta_k) * freq_offset_phasor

    #add noise
    noise = (np.random.randn(N_sym) + 1j*np.random.randn(N_sym)) * 0.05
    y_received += noise

    #process
    z_freq_corrected, f_est = dsp.freq_recovery(y_received, fs)
    print(f"Freq Offset Real: {delta_f_real} Hz | Estimated: {f_est:.4f} Hz")
    z_final, theta_est = dsp.phase_recovery(z_freq_corrected)

    
    #PLOT
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    #raw
    ax1.scatter(y_received.real, y_received.imag, s=2, c='tab:red', alpha=0.5)
    ax1.set_title(f"1. Recebido (Raw)\nOffset $\Delta f \\approx {delta_f_real}$ Hz")
    ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-1.5, 1.5)

    #freq recovery
    ax2.scatter(z_freq_corrected.real, z_freq_corrected.imag, s=2, c='tab:orange', alpha=0.5)
    ax2.set_title(f"2. Pós-Rec. Frequência\nRemoveu rotação rápida")
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)

    #phase recovery
    ax3.scatter(z_final.real, z_final.imag, s=2, c='tab:green', alpha=0.5)
    ax3.set_title(f"3. Pós-Rec. Fase (Final)\nConstelação Limpa")
    ax3.set_xlim(-1.5, 1.5); ax3.set_ylim(-1.5, 1.5)

    #add grid lines
    for ax in [ax1, ax2, ax3]:
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_aspect('equal', 'box')

    plt.tight_layout()
    
    #save the figure
    filename = "build/full_recovery_cascade.png"
        
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()
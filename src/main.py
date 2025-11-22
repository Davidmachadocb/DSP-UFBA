import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'Times New Roman', 'serif'],
    'mathtext.fontset': 'stix', 
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.figsize': (10, 5),
    'figure.dpi': 300
})

def freq_recovery(y, fs):
    """
    Algoritmo de recuperação de portadora para QPSK (Método da 4ª Potência).
    
    Args:
        y (np.array): Sinal QPSK complexo recebido (com offset de frequência).
        fs (float): Taxa de amostragem.
        
    Returns:
        z (np.array)
        delta_f_est (float)
    """
    N = len(y)
    
    # 1. Elevar à 4ª potência para remover a modulação QPSK
    y_4 = y ** 4
    
    # 2. FFT para encontrar o pico espectral
    Y_4_spec = np.fft.fft(y_4, n=4*N)
    freqs = np.fft.fftfreq(len(Y_4_spec), d=1/fs)
    
    # 3. Encontrar o índice do máximo espectral
    idx_max = np.argmax(np.abs(Y_4_spec))
    f_peak = freqs[idx_max]
    
    # 4. Calcular o Delta f estimado (1/4 do pico)
    delta_f_est = f_peak / 4.0
    
    # 5. Remover o offset de frequência
    # Cria o vetor de tempo correspondente
    t = np.arange(N) / fs
    
    # Gera a exponencial complexa corretiva: e^(-j 2pi delta_f t)
    correction_phasor = np.exp(-1j * 2 * np.pi * delta_f_est * t)
    
    # Aplica a correção (Misturador na Fig 6.1)
    z = y * correction_phasor
    
    return z, delta_f_est

# parameters
fs = 10000          # sampling rate
N_sym = 1000        # number of symbols
delta_f_real = 150   # frequency offset

# exp( j(pi/4 + m[k]pi/2)  )
ints = np.random.randint(0, 4, N_sym)
phases = (np.pi/4) + (ints * np.pi/2)
s = np.exp(1j * phases)

#  y[k] = s[k] exp( jk2pi Df T_s  )
t = np.arange(N_sym) / fs
freq_offset = np.exp(1j * 2 * np.pi * delta_f_real * t)
y_received = s * freq_offset

# white noise
noise = (np.random.randn(N_sym) + 1j*np.random.randn(N_sym)) * 0.1
y_received += noise

z_corrected, f_est = freq_recovery(y_received, fs)

print(f"Offset Real: {delta_f_real} Hz")
print(f"Offset Estimado: {f_est:.4f} Hz")

# --- Plotagem ---
fig, (ax1, ax2) = plt.subplots(1, 2)

# Plot Antes (Sinal Recebido)
ax1.scatter(y_received.real, y_received.imag, s=5, c='tab:red', alpha=0.6)
ax1.set_title(f"Antes: Sinal Recebido\n(Offset $\Delta f = {delta_f_real}$ Hz)")
ax1.set_xlabel("In-Phase (I)")
ax1.set_ylabel("Quadrature (Q)")
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.axhline(0, color='black', linewidth=0.8)
ax1.axvline(0, color='black', linewidth=0.8)

# Plot Depois (Sinal Recuperado)
ax2.scatter(z_corrected.real, z_corrected.imag, s=5, c='tab:green', alpha=0.6)
ax2.set_title(f"Depois: Sinal Recuperado\n(Est. $\widehat{{\Delta f}} = {f_est:.2f}$ Hz)")
ax2.set_xlabel("In-Phase (I)")
# ax2.set_ylabel("Quadrature (Q)") # Opcional para limpar visual
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.axhline(0, color='black', linewidth=0.8)
ax2.axvline(0, color='black', linewidth=0.8)

plt.tight_layout()

plt.savefig("build/freq_rec")

plt.show()
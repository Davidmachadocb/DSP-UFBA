import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from optic.dsp.core import pnorm
import scipy as sp

def phase_recovery(y, N=20, M=4, Es=1.0, sigma_delta_theta=0.01, sigma_n=0.1):
    """
    Optimized Viterbi & Viterbi Phase Recovery using Vectorized Matrix Construction.
    """

    ## N amostrar no passado e N amostras no futuro
    L = 2 * N + 1 

    coeff_K = (Es**M) * (M**2) * (sigma_delta_theta**2) # peso pra variancia do ruido de fase
    coeff_I = (Es**(M-1)) * (M**2) * (sigma_n**2)       # peso pra variancia do AWGN

    # Vectorized K matrix construction
    idx = np.arange(-N, N + 1)
    I, J = np.meshgrid(idx, idx, indexing='ij')
    
    sign_match = (np.sign(I) == np.sign(J))
    nonzero = (I != 0)
    
    K = np.zeros((L, L))
    mask = sign_match & nonzero
    K[mask] = np.minimum(np.abs(I[mask]), np.abs(J[mask]))

    I_mat = np.eye(L)
    C = (coeff_K * K) + (coeff_I * I_mat)

    ones_vec = np.ones(L)
    
    C_inv = linalg.inv(C)
    w_ml = np.dot(ones_vec.T, C_inv).T

    y_M = y**M
    z_vec = np.convolve(y_M, w_ml, mode='same')
    
    theta_est = (1/M) * np.angle(z_vec) - (np.pi / M)
    theta_est_unwrapped = np.unwrap(theta_est) 
    
    z = y * np.exp(-1j * theta_est_unwrapped)
    
    return z, theta_est_unwrapped

def freq_recovery(y, fs):
    """
    Coarse Frequency Offset Estimation (FFT-based).
    """
    N = len(y)
    y_4 = y ** 4
    
    Y_4_spec = np.fft.fft(y_4, n=4*N)
    freqs = np.fft.fftfreq(len(Y_4_spec), d=1/fs)
    
    idx_max = np.argmax(np.abs(Y_4_spec))
    f_peak = freqs[idx_max]
    
    delta_f_est = f_peak / 4.0
    
    t = np.arange(N) / fs
    correction_phasor = np.exp(-1j * 2 * np.pi * delta_f_est * t)
    
    z = y * correction_phasor
    
    return z, delta_f_est
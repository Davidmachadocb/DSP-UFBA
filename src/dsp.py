import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from optic.dsp.core import pnorm
import scipy as sp

def phase_recovery(y, N=20, M=4, Es=1.0, sigma_delta_theta=0.01, sigma_n=0.1):
    """
    Optimized Viterbi & Viterbi Phase Recovery using Vectorized Matrix Construction.
    """
    L = 2 * N + 1
    coeff_K = (Es**M) * (M**2) * (sigma_delta_theta**2)
    coeff_I = (Es**(M-1)) * (M**2) * (sigma_n**2)

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
    
    try:
        w_ml = linalg.solve(C, ones_vec)
    except linalg.LinAlgError:
        w_ml = np.ones(L)

    w_ml = w_ml / np.sum(w_ml)

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

def mlFilterVV(Es, nModes, OSNRdB, delta_lw, Rs, N, M=4):
    """
    Calcula os coeficientes do filtro de máxima verossimilhança (ML) para o algoritmo 
    Viterbi & Viterbi, que depende da relação sinal-ruído e da magnitude do ruído de fase.

    Parameters
    ----------
    Es : float
        Energia dos símbolos.
    
    nModes : int
        Número de polarizações.

    OSNRdB : float
        OSNR do canal em dB.

    delta_lw : int
        Soma das larguras de linha do laser do oscilador local e transmissor.

    Rs : int
        Taxa de símbolos. [símbolos/segundo]

    N : int
        Número de símbolos passados e futuros na janela. O comprimento
        do filtro é então L = 2*N+1

    M : int, optional
        Ordem do esquema de modulação M-PSK. Defaults to 4.

    Returns
    -------
    np.array
        Coeficientes do filtro de máxima verossimilhança a ser usado em Viterbi & Viterbi.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms

        [2] E. Ip, J.M. Kahn, Feedforward carrier recovery for coherent optical communications. J.
            Lightwave Technol. 25(9), 2675–2692 (2007).
    """
    
    Ts = 1/Rs          # Período de símbolo
    L  = 2 * N + 1     # Comprimento do filtro
    Bref = 12.5e9      # Banda de referência [Hz]

    # dB para valor linear
    SNR = 10**(OSNRdB / 10) * (2 * Bref) / (nModes*Rs)
    
    # define a variância do ruído multiplicativo
    σ_deltaTheta = 2 * np.pi * delta_lw * Ts
    
    # define a variância do ruído aditivo
    σ_eta = Es / (2 * SNR)
    
    K = np.zeros((L, L))
    B = np.zeros((N + 1, N + 1))
    
    # Determina a matriz B de forma vetorizada evitando loop nested 
    # e overhead de loops explícitos
    index = np.arange(N + 1)
    B = np.minimum.outer(index, index)
    
    K[:N+1,:N+1] = np.rot90(B, 2)
    K[N:L,N:L] = B
    
    I = np.eye(L)
    
    # Obtém a matriz de covariância
    C = Es**M * M**2 * σ_deltaTheta * K + Es**(M - 1) * M**2 * σ_eta * I
    
    # Determina os coeficientes do filtro 
    h = np.linalg.inv(C) @ np.ones(L)
    
    return h/np.max(h)

def viterbiCPR(z, lw, Rs, OSNRdB, N, M=4):
    """
    Compensa o ruído de fase com o algoritmo Viterbi & Viterbi.
    
    Parameters
    ----------
    z : np.array
        Sinal normalizado em potência, no qual a recuperação de fase será realizada.
        
    lw : int
        Soma das larguras de linha do laser do oscilador local e transmissor.

    Rs : int
        Taxa de símbolos. [símbolos/segundo].
        
    OSNRdB : float
        OSNR do canal em dB.
        
    N : int
        Número de símbolos passados e futuros na janela. O comprimento
        do filtro é então L = 2*N+1.
        
    M : int, optional
        Ordem da potência, by default 4

    Returns
    -------
    tuple:
        sigRx (np.array): Constelação com referência de fase.
        phiPU (np.array): Estimativa do ruído de fase em cada modo.
    
    Referências
    -----------
        [1] Digital Coherent Optical Systems, Architecture and Algorithms
    """
    
    try:
        nModes = z.shape[1]
    except IndexError:
        z = z.reshape(len(z), 1)
        nModes = 1
    
    # comprimento do filtro
    L = 2 * N + 1

    Es = np.mean(np.abs(z)**2)

    # obtém os coeficientes do filtro de máxima verossimilhança
    h = mlFilterVV(Es, nModes, OSNRdB, lw, Rs, N)
    
    # estimativa de fase 
    phiTime = np.zeros(z.shape)
    
    for indPhase in range(nModes):
        
        # zero padding
        sigRx = np.pad(z[:, indPhase], (L//2, L//2), mode='constant')
        
        # calcula a matriz de convolução de comprimento L
        sigRx = convmtx(sigRx, L)
        
        # up-down flip
        sigRx = np.flipud(sigRx[:, L-1: -L+1])
        
        # obtém a estimativa de fase em cada modo 
        phiTime[:, indPhase] = np.angle(np.dot(h.T, sigRx**M)) / M - np.pi/M
    
    # phase unwrap
    phiPU = np.unwrap(phiTime, period=2*np.pi/M, axis=0)
    
    # compensa o ruído de fase
    z = pnorm(z * np.exp(-1j * phiPU))
    
    return z, phiPU


def convmtx(h, N):
    """
    Determina uma matriz de convolução a partir de um vetor de entrada 'h'

    Parameters
    ----------
    h : np.array
        Matriz de entrada, especificado como linha ou coluna.
    
    N : int
        Comprimento do vetor a ser convoluído, especificado como um número inteiro positivo.

    Returns
    -------
    H: np.array:
        Matriz de convolução H
    """

    H = sp.linalg.toeplitz(np.hstack((h, np.zeros(N-1))), np.zeros(N))
    return H.T
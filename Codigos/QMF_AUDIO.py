"""
===== COMPRESSÃO DE ÁUDIO POR WAVELETS - ANÁLISE DE SNR =====

Implementação didática para análise de compressão de sinais de áudio utilizando
a Transformada Wavelet Discreta (DWT) com limiarização e reconstrução parcial.

Este código foi desenvolvido como parte do minicurso 
"Introdução à Compressão e Processamento de Áudio com Wavelets",
ministrado pelo IEEE Signal Processing Society Student Chapter da 
Universidade de Brasília (IEEE SPS UnB) durante a Semana Universitária (SEMUNI) 2025.

O script é estruturado em módulos independentes, abordando conceitos fundamentais de:
- Decomposição e reconstrução de sinais por wavelets;
- Limiarização e compressão de coeficientes;
- Análise de SNR (Signal-to-Noise Ratio) em diferentes níveis de compressão;
- Visualização gráfica e exportação de áudio reconstruído.

Baseado em material de referência acadêmica sobre filtragem QMF e transformadas wavelet,
refatorado para fins didáticos, clareza e eficiência computacional.

Refatoração e organização: IEEE SPS UnB
"""


import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sounddevice as sd
from scipy.io.wavfile import write
from typing import Tuple, Optional


# ----- CARREGAMENTO E PRÉ-PROCESSAMENTO DE DADOS# ----- #
def load_audio_signal(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Carrega sinal de áudio de arquivo .mat.
    
    Args:
        filepath: Caminho do arquivo .mat
    
    Returns:
        Tupla (sinal, frequência_amostragem)
    """
    data = loadmat(filepath)
    
    signal = data['x'].squeeze()
    fs = int(data['fs'].squeeze())
    
    # Garante que o sinal seja 1D
    if signal.ndim > 1:
        signal = signal.flatten()
    
    return signal, fs

def print_signal_info(signal: np.ndarray, fs: int):
    """Imprime informações sobre o sinal carregado."""
    duration = len(signal) / fs
    
    print("\n" + "="*60)
    print("INFORMAÇÕES DO SINAL DE ÁUDIO")
    print("="*60)
    print(f"Frequência de amostragem: {fs} Hz")
    print(f"Número de amostras: {len(signal)}")
    print(f"Duração: {duration:.2f} segundos")
    print(f"Amplitude máxima: {np.max(np.abs(signal)):.4f}")
    print("="*60 + "\n")
# ------------------------------------------------------ #


# ----- DECOMPOSIÇÃO WAVELET ----- #
def decompose_wavelet(signal: np.ndarray, wavelet: str = 'db5', 
                      level: int = 4) -> Tuple[np.ndarray, list]:
    """
    Realiza decomposição wavelet do sinal.
    
    Args:
        signal: Sinal de entrada
        wavelet: Tipo de wavelet (ex: 'db5' para Daubechies 5)
        level: Número de níveis de decomposição
    
    Returns:
        Tupla (array_coeficientes, slices_coeficientes)
    """
    # Decomposição wavelet multinível
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Converte para array único para facilitar manipulação
    coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    return coeff_array, coeff_slices
# -------------------------------- #


# ----- COMPRESSÃO E LIMIARIZAÇÃO ----- #
def threshold_coefficients(coeff_array: np.ndarray, 
                          keep_percentage: float) -> np.ndarray:
    """
    Aplica limiarização mantendo apenas os maiores coeficientes.
    
    Args:
        coeff_array: Array de coeficientes wavelet
        keep_percentage: Porcentagem de coeficientes a manter (0-100)
    
    Returns:
        Array com coeficientes limiarizados (outros zerados)
    """
    num_total = coeff_array.size
    num_keep = int(num_total * keep_percentage / 100)
    
    # Define limiar baseado nos maiores coeficientes em magnitude
    threshold = np.partition(np.abs(coeff_array.flatten()), -num_keep)[-num_keep]
    
    # Zera coeficientes abaixo do limiar
    thresholded = np.where(np.abs(coeff_array) >= threshold, coeff_array, 0)
    
    return thresholded

def count_nonzero_coefficients(coeff_array: np.ndarray, 
                               tolerance: float = 1e-12) -> int:
    """
    Conta coeficientes não-zerados (acima da tolerância).
    
    Args:
        coeff_array: Array de coeficientes
        tolerance: Tolerância para considerar zero
    
    Returns:
        Número de coeficientes não-zerados
    """
    return np.sum(np.abs(coeff_array) >= tolerance)
# ------------------------------------- #


# ----- RECONSTRUÇÃO DE SINAIS ----- #
def reconstruct_signal(coeff_array: np.ndarray, coeff_slices: list,
                       wavelet: str, original_length: int) -> np.ndarray:
    """
    Reconstrói sinal a partir dos coeficientes wavelet.
    
    Args:
        coeff_array: Array de coeficientes
        coeff_slices: Estrutura de slices para reconstrução
        wavelet: Tipo de wavelet usado na decomposição
        original_length: Comprimento do sinal original
    
    Returns:
        Sinal reconstruído
    """
    # Converte array de volta para estrutura de coeficientes
    coeffs_reconstructed = pywt.array_to_coeffs(coeff_array, coeff_slices, 
                                                 output_format='wavedec')
    
    # Reconstrói o sinal
    signal_reconstructed = pywt.waverec(coeffs_reconstructed, wavelet)
    
    # Ajusta ao tamanho original (remove padding)
    signal_reconstructed = signal_reconstructed[:original_length]
    
    return signal_reconstructed
# ---------------------------------- # 


# ----- MÉTRICAS DE QUALIDADE ----- #
def calculate_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calcula SNR (Signal-to-Noise Ratio) em dB.
    
    Args:
        original: Sinal original
        reconstructed: Sinal reconstruído
    
    Returns:
        SNR em decibéis
    """
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def calculate_compression_ratio(total_coeffs: int, 
                               kept_coeffs: int) -> float:
    """
    Calcula taxa de compressão.
    
    Args:
        total_coeffs: Total de coeficientes
        kept_coeffs: Coeficientes mantidos
    
    Returns:
        Taxa de compressão
    """
    return total_coeffs / kept_coeffs if kept_coeffs > 0 else float('inf')
# --------------------------------- #


# ----- ANÁLISE DE COMPRESSÃO ----- #
def analyze_compression(signal: np.ndarray, coeff_array: np.ndarray,
                       coeff_slices: list, wavelet: str,
                       percentages: np.ndarray = None) -> Tuple[list, list]:
    """
    Analisa qualidade de compressão para diferentes porcentagens.
    
    Args:
        signal: Sinal original
        coeff_array: Array de coeficientes wavelet
        coeff_slices: Estrutura de slices
        wavelet: Tipo de wavelet
        percentages: Array de porcentagens a testar
    
    Returns:
        Tupla (lista_SNRs, lista_percentagens)
    """
    if percentages is None:
        percentages = np.arange(5, 105, 5)
    
    snr_values = []
    num_total = coeff_array.size
    
    print("\n" + "="*70)
    print("ANÁLISE DE COMPRESSÃO - SNR vs PORCENTAGEM DE COEFICIENTES")
    print("="*70)
    print(f"{'Mantidos (%)':<15} | {'SNR (dB)':<12} | {'Coef. Ativos':<20} | {'Compressão':<10}")
    print("-"*70)
    
    for percentage in percentages:
        # Aplica limiarização
        thresholded = threshold_coefficients(coeff_array, percentage)
        
        # Conta coeficientes não-zerados
        num_nonzero = count_nonzero_coefficients(thresholded)
        
        # Reconstrói sinal
        reconstructed = reconstruct_signal(thresholded, coeff_slices, 
                                          wavelet, len(signal))
        
        # Calcula SNR
        snr = calculate_snr(signal, reconstructed)
        snr_values.append(snr)
        
        # Calcula taxa de compressão
        compression = calculate_compression_ratio(num_total, num_nonzero)
        
        # Imprime resultados
        print(f"{percentage:>7.0f}%          | {snr:>10.2f} dB | "
              f"{num_nonzero}/{num_total:<10} | {compression:>6.1f}x")
    
    print("="*70 + "\n")
    
    return snr_values, percentages
# --------------------------------- #


# ----- VISUALIZAÇÃO ----- #
def plot_snr_analysis(percentages: np.ndarray, snr_values: list,
                     wavelet: str = 'db5'):
    """
    Plota gráfico de SNR vs Porcentagem de coeficientes mantidos.
    
    Args:
        percentages: Array de porcentagens testadas
        snr_values: Lista de valores de SNR correspondentes
        wavelet: Nome da wavelet usada
    """
    plt.figure(figsize=(12, 6))
    plt.plot(percentages, snr_values, marker='o', color='#2E86AB', 
             linewidth=2, markersize=6, label='SNR')
    
    # Linha de referência para 30 dB (boa qualidade)
    plt.axhline(y=30, color='red', linestyle='--', alpha=0.5, 
                label='Referência: 30 dB')
    
    plt.xlabel('Porcentagem de coeficientes mantidos (%)', fontsize=12)
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.title(f'Compressão por Wavelets ({wavelet.upper()}) - '
              f'SNR vs Porcentagem', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
# ------------------------ #


# ----- EXPORTAÇÃO E REPRODUÇÃO DE ÁUDIO ----- #
def save_reconstructed_audio(signal: np.ndarray, coeff_array: np.ndarray,
                            coeff_slices: list, wavelet: str,
                            fs: int, keep_percentage: float,
                            output_filename: str = 'audio_reconstruido.wav',
                            play_audio: bool = False) -> np.ndarray:
    """
    Reconstrói, normaliza, salva e opcionalmente reproduz áudio comprimido.
    
    Args:
        signal: Sinal original
        coeff_array: Array de coeficientes wavelet
        coeff_slices: Estrutura de slices
        wavelet: Tipo de wavelet
        fs: Frequência de amostragem
        keep_percentage: Porcentagem de coeficientes a manter
        output_filename: Nome do arquivo de saída
        play_audio: Se True, reproduz o áudio após salvar
    
    Returns:
        Sinal reconstruído e normalizado
    """
    # Limiarização
    thresholded = threshold_coefficients(coeff_array, keep_percentage)
    
    # Reconstrução
    reconstructed = reconstruct_signal(thresholded, coeff_slices, 
                                      wavelet, len(signal))
    
    # Normaliza para intervalo [-1, 1]
    normalized = reconstructed / np.max(np.abs(reconstructed))
    
    # Converte para formato WAV (16-bit PCM)
    wav_data = np.int16(normalized * 32767)
    
    # Salva arquivo
    write(output_filename, fs, wav_data)
    
    # Calcula métricas
    snr = calculate_snr(signal, reconstructed)
    num_total = coeff_array.size
    num_kept = count_nonzero_coefficients(thresholded)
    compression = calculate_compression_ratio(num_total, num_kept)
    
    print("\n" + "="*60)
    print("ÁUDIO SALVO COM SUCESSO")
    print("="*60)
    print(f"📁 Arquivo: {output_filename}")
    print(f"📊 Coeficientes mantidos: {keep_percentage}%")
    print(f"🗜️  Taxa de compressão: {compression:.1f}x")
    print(f"📈 SNR: {snr:.2f} dB")
    print("="*60 + "\n")
    
    # Reproduz se solicitado
    if play_audio:
        print(f"▶️  Reproduzindo áudio com {keep_percentage}% dos coeficientes...")
        sd.play(normalized, fs)
        sd.wait()
        print("✓ Reprodução concluída\n")
    
    return normalized
# -------------------------------------------- #


# ----- FUNÇÃO PRINCIPAL ----- #
def main():
    """Função principal de execução."""
    
    # CONFIGURAÇÃO
    AUDIO_FILE = 'exemplo_audio.mat'
    WAVELET = 'db5'
    DECOMPOSITION_LEVEL = 4
    PERCENTAGES = np.arange(5, 105, 5)
    
    # CARREGAMENTO 
    print("Carregando sinal de áudio...")
    signal, fs = load_audio_signal(AUDIO_FILE)
    print_signal_info(signal, fs)
    
    # DECOMPOSIÇÃO
    print(f"Realizando decomposição wavelet ({WAVELET.upper()}, "
          f"{DECOMPOSITION_LEVEL} níveis)...")
    coeff_array, coeff_slices = decompose_wavelet(signal, WAVELET, 
                                                   DECOMPOSITION_LEVEL)
    print(f"✓ Total de coeficientes: {coeff_array.size}\n")
    
    # ANÁLISE DE COMPRESSÃO 
    snr_values, percentages = analyze_compression(signal, coeff_array, 
                                                   coeff_slices, WAVELET,
                                                   PERCENTAGES)
    
    # ========== VISUALIZAÇÃO ==========
    plot_snr_analysis(percentages, snr_values, WAVELET)
    
    # EXPORTAÇÃO DE ÁUDIO
    EXPORT_PERCENTAGE = 25
    OUTPUT_FILE = f'audio{EXPORT_PERCENTAGE}.wav'
    
    save_reconstructed_audio(
        signal=signal,
        coeff_array=coeff_array,
        coeff_slices=coeff_slices,
        wavelet=WAVELET,
        fs=fs,
        keep_percentage=EXPORT_PERCENTAGE,
        output_filename=OUTPUT_FILE,
        play_audio=True  # Altere para False se não quiser reproduzir
    )

if __name__ == '__main__':
    main()
# ---------------------------- #
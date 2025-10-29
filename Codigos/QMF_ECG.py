"""
===== ANÁLISE MULTINÍVEL E MULTITAXA DE SINAIS =====

Implementação didática para decomposição e reconstrução de sinais utilizando filtros QMF.

Este código foi desenvolvido como parte do minicurso "Introdução ao Processamento de Sinais com Filtros QMF",
ministrado pelo IEEE Signal Processing Society Student Chapter da Universidade de Brasília (IEEE SPS UnB)
durante a Semana Universitária (SEMUNI) 2025.

O script apresenta uma estrutura modular e otimizada para fins educacionais, abordando conceitos de:
- Análise e síntese multibanda;
- Processamento multitaxa;
- Decomposição e reconstrução de sinais.

Baseado em material fornecido pelo professor orientador e adaptado para maior clareza e eficiência.

Refatoração e organização: IEEE SPS UnB
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from copy import deepcopy


# ----- OPERAÇÕES BÁSICAS DE PROCESSAMENTO DE SINAIS ----- #
def downsample(x, n=None, M=2):
    """
    Realiza downsampling de um sinal por fator M.
    
    Args:
        x: Sinal de entrada
        n: Índices do sinal (opcional)
        M: Fator de downsampling
    
    Returns:
        y: Sinal com downsampling
        n_: Novos índices
    """
    if n is None:
        n = np.arange(len(x))
    
    # Encontra o índice zero
    k = np.where(n == 0)[0][0]
    
    # Processa lado direito
    x_right = x[k:]
    y_right = x_right[::M]
    
    # Processa lado esquerdo
    x_left = x[:k+1][::-1]
    y_left = x_left[::M][::-1]
    y_left = y_left[:-1]
    
    # Combina resultados
    n_ = np.arange(-len(y_left), len(y_right))
    y = np.concatenate((y_left, y_right))
    
    return y, n_

def upsample(x, M=2, remove_leading_zeros=True):
    """
    Realiza upsampling de um sinal por fator M inserindo zeros.
    
    Args:
        x: Sinal de entrada
        M: Fator de upsampling
        remove_leading_zeros: Remove zeros finais se True
    
    Returns:
        y: Sinal com upsampling
    """
    y = np.zeros(len(x) * M)
    y[::M] = x
    
    if remove_leading_zeros:
        y = y[:-(M-1)]
    
    return y

def signal_sum(x, y):
    """
    Soma dois sinais de tamanhos diferentes, alinhando pelo início.
    
    Args:
        x, y: Sinais a serem somados
    
    Returns:
        Soma dos sinais
    """
    if len(y) > len(x):
        x, y = y, x
    
    ya = np.zeros_like(x)
    ya[:len(y)] = y
    
    return x + ya

def unite_signals(x_list):
    """
    Une uma lista de sinais em um único vetor concatenado.
    
    Args:
        x_list: Lista de sinais
    
    Returns:
        Sinal concatenado
    """
    N = sum(len(xk) for xk in x_list)
    y = np.zeros(N)
    n = 0
    
    for xk in x_list:
        y[n:n+len(xk)] = xk
        n += len(xk)
    
    return y
# -------------------------------------------------------- #


# ----- VALIDAÇÃO E OPERAÇÕES DE FILTROS QMF ----- #
def filter_change_z_signal(h):
    """Altera o sinal dos coeficientes ímpares (substitui z por -z)."""
    h_ = deepcopy(h)
    h_[1::2] = -h_[1::2]
    return h_

def locations_null_coefficients(v, tol):
    """Retorna índices dos coeficientes nulos dentro da tolerância."""
    return np.where(np.abs(v) <= tol)[0]

def qmf_filters_validator(h0, h1, g0, g1, tol=1e-6):
    """
    Valida se os filtros formam um banco QMF válido.
    
    Verifica:
    - Termo de aliasing: 1/2*(H0(-z)G0(z) + H1(-z)G1(z)) = 0
    - Termo LTI: 1/2*(H0(z)G0(z) + H1(z)G1(z)) = A*z^(-d)
    
    Args:
        h0, h1: Filtros de análise
        g0, g1: Filtros de síntese
        tol: Tolerância para verificação
    
    Returns:
        valid_filters: True se válido
        lti_term_valid: Validação do termo LTI
        aliasing_term_null: Validação do termo de aliasing
        A: Ganho do sistema
        d: Atraso do sistema
    """
    # Verificação do termo de aliasing
    h0_ = filter_change_z_signal(h0)
    h1_ = filter_change_z_signal(h1)
    aliasing_term = 0.5 * signal_sum(np.convolve(h0_, g0), np.convolve(h1_, g1))
    aliasing_term_null = len(locations_null_coefficients(aliasing_term, tol)) == len(aliasing_term)
    
    # Verificação do termo LTI
    lti_term = 0.5 * signal_sum(np.convolve(h0, g0), np.convolve(h1, g1))
    k_null = locations_null_coefficients(lti_term, tol)
    
    if len(k_null) == len(lti_term) - 1:
        lti_term_valid = True
        n = np.arange(len(lti_term))
        d = np.setdiff1d(n, k_null)[0]
        A = lti_term[d]
    else:
        lti_term_valid = False
        A, d = np.nan, np.nan
    
    valid_filters = lti_term_valid and aliasing_term_null
    
    return valid_filters, lti_term_valid, aliasing_term_null, A, d

def filter_iterator(x0, x1, A, d, levels=4):
    """
    Gera banco de filtros para decomposição multinível.
    
    Args:
        x0, x1: Filtros base
        A: Ganho do sistema
        d: Atraso do sistema
        levels: Número de níveis
    
    Returns:
        x: Filtros para cada nível
        multirate_factors: Fatores de decimação
        rescale_factors: Fatores de reescala
        advance_values: Valores de avanço temporal
    """
    x = [None] * (levels + 1)
    x0_ = deepcopy(x0)
    x[levels] = x1
    
    # Gera filtros iterativamente
    for k in range(levels - 1, 0, -1):
        x[k] = np.convolve(upsample(x[k+1], 2), x0)
        x0_ = np.convolve(upsample(x0_, 2), x0)
    
    x[0] = x0_
    
    # Calcula parâmetros multitaxa
    T = np.arange(levels + 1, 0, -1)
    T[0] = levels
    multirate_factors = 2 ** T
    rescale_factors = 1.0 / (A ** T)
    advance_values = d * ((2 ** T) - 1)
    
    return x, multirate_factors, rescale_factors, advance_values
# ------------------------------------------------ #


# ----- DECOMPOSIÇÃO E RECONSTRUÇÃO MULTINÍVEL ----- #
def multivel_multirate_decomposition(x, h0, h1, A, d, levels=4):
    """
    Realiza decomposição multinível e multitaxa de um sinal.
    
    Args:
        x: Sinal de entrada
        h0, h1: Filtros de análise
        A, d: Parâmetros do banco QMF
        levels: Número de níveis de decomposição
    
    Returns:
        x_hat: Sinal decomposto concatenado
        x_decomp: Lista com componentes de cada nível
    """
    h, multirate_factors, _, _ = filter_iterator(h0, h1, A, d, levels=levels)
    
    x_decomp = []
    for k, (filt, M) in enumerate(zip(h, multirate_factors)):
        x_filtered = np.convolve(filt, x)
        x_downsampled, _ = downsample(x_filtered, M=M)
        x_decomp.append(x_downsampled)
    
    x_hat = unite_signals(x_decomp)
    
    return x_hat, x_decomp

def multivel_multirate_reconstruction(x_decomp, g0, g1, A, d):
    """
    Reconstrói sinal a partir da decomposição multinível.
    
    Args:
        x_decomp: Lista de componentes decompostas
        g0, g1: Filtros de síntese
        A, d: Parâmetros do banco QMF
    
    Returns:
        xr: Sinal reconstruído
    """
    levels = len(x_decomp) - 1
    g, multirate_factors, rescale_factors, advance_values = filter_iterator(
        g0, g1, A, d, levels=levels
    )
    
    xr = np.zeros(1)
    
    for k in range(len(x_decomp)):
        # Upsampling
        xd = upsample(x_decomp[k], M=multirate_factors[k])
        
        # Filtragem
        xd = np.convolve(g[k], xd)
        
        # Reescala e avanço temporal
        xd *= rescale_factors[k]
        xd = xd[advance_values[k]:]
        
        # Acumula componentes
        xr = signal_sum(xr, xd)
    
    return xr
# -------------------------------------------------- #


# ----- VISUALIZAÇÃO E CONFIGURAÇÃO ----- #
def configure_plot_style():
    """Configura estilo global dos gráficos."""
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,
    })

def plot_signal(x, fs, xlabel='Tempo (s)', ylabel='Amplitude', title=None):
    """
    Plota um sinal no domínio do tempo.
    
    Args:
        x: Sinal
        fs: Frequência de amostragem
        xlabel, ylabel: Rótulos dos eixos
        title: Título do gráfico (opcional)
    """
    t = np.arange(len(x)) / fs
    
    plt.figure(figsize=(12, 4))
    plt.plot(t, x, linewidth=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def print_signal_info(x, fs, Ts):
    """Imprime informações do sinal."""
    print("\n" + "="*50)
    print("INFORMAÇÕES DO SINAL".center(50))
    print("="*50)
    print(f"Taxa de amostragem (fs): {fs:.2f} Hz")
    print(f"Período de amostragem (Ts): {Ts:.6f} s")
    print(f"Número total de amostras: {len(x)}")
    print(f"Duração do sinal: {len(x)/fs:.2f} s")
    print("="*50 + "\n")

def print_decomposition_results(x, xr, levels, fs, rmse):
    """Imprime resultados da decomposição."""
    total_reduction = 2 ** levels
    min_samples = int(np.ceil(len(x) / total_reduction))
    fs_eff = fs / total_reduction
    
    print("\n" + "="*50)
    print("RESULTADOS DA DECOMPOSIÇÃO".center(50))
    print("="*50)
    print(f"Tamanho original: {len(x)} amostras")
    print(f"Tamanho reconstruído: {len(xr)} amostras")
    print(f"Níveis de decomposição: {levels}")
    print(f"Fator total de redução: {total_reduction}x")
    print(f"Taxa efetiva após decomposição: {fs_eff:.2f} Hz")
    print(f"Amostras mínimas para reconstrução: {min_samples}")
    print(f"Erro médio quadrático (RMSE): {rmse:.6e}")
    print("="*50 + "\n")
# --------------------------------------- #


# ----- EXECUÇÃO PRINCIPAL ----- #
def main():
    """Função principal de execução."""
    # Configuração inicial
    configure_plot_style()
    
    # Carrega sinal ECG
    print("Carregando sinal ECG...")
    M = loadmat('ECG_1.mat')
    x = M['x'][:, 0]
    fs = float(M['fs'].squeeze())
    Ts = 1.0 / fs
    x = x - np.mean(x)  # Remove componente DC
    
    print_signal_info(x, fs, Ts)
    plot_signal(x, fs, ylabel='ECG (amplitude normalizada)', 
                title='Sinal ECG Original')
    
    # Carrega filtros
    print("Carregando banco de filtros QMF...")
    M = loadmat('wfilters.mat')
    chosen_wfilter = 4  # Daubechies 5
    h0 = M['h0'][0, chosen_wfilter][0]
    h1 = M['h1'][0, chosen_wfilter][0]
    g0 = M['g0'][0, chosen_wfilter][0]
    g1 = M['g1'][0, chosen_wfilter][0]
    
    # Valida filtros
    print("Validando filtros QMF...")
    valid, lti_valid, alias_null, A, d = qmf_filters_validator(h0, h1, g0, g1)
    
    if not valid:
        print("AVISO: Filtros não formam um banco QMF válido!")
        return
    
    print(f"Filtros validados: A = {A:.4f}, d = {d}")
    
    # Decomposição
    print("\nRealizando decomposição multinível...")
    levels = 5
    x_hat, x_decomp = multivel_multirate_decomposition(x, h0, h1, A, d, levels=levels)
    plot_signal(x_hat, 1, xlabel='Índice k', ylabel='x̂[k]', 
                title='Sinal Decomposto (Concatenado)')
    
    # Reconstrução
    print("Reconstruindo sinal...")
    xr = multivel_multirate_reconstruction(x_decomp, g0, g1, A, d)
    plot_signal(xr, fs, ylabel='Amplitude', 
                title='Sinal Reconstruído')
    
    # Análise de erro
    print("Calculando erro de reconstrução...")
    err = xr[:len(x)] - x
    rmse = np.sqrt(np.mean(err ** 2))
    plot_signal(err, fs, ylabel='Erro', 
                title='Erro de Reconstrução (xr - x)')
    
    # Resultados finais
    print_decomposition_results(x, xr, levels, fs, rmse)

if __name__ == '__main__':
    main()
# ------------------------------ #
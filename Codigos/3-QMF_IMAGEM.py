"""
===== PROCESSAMENTO DE IMAGENS COM FILTROS QMF =====

Implementação didática de Quadrature Mirror Filter Bank para decomposição
e reconstrução de imagens.

Baseado nos conceitos apresentados no minicurso IEEE SPS UnB - SEMUNI 2025

Refatoração e organização: IEEE SPS UnB
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass


# ----- CLASSES DE DADOS ----- #
@dataclass
class QMFFilters:
    """Armazena os filtros QMF de análise e síntese."""
    h0: np.ndarray  # Análise passa-baixa
    h1: np.ndarray  # Análise passa-alta
    g0: np.ndarray  # Síntese passa-baixa
    g1: np.ndarray  # Síntese passa-alta

@dataclass
class Subbands:
    """Armazena as subbandas da decomposição QMF."""
    LL: np.ndarray  # Baixa-Baixa (aproximação)
    LH: np.ndarray  # Baixa-Alta (detalhes horizontais)
    HL: np.ndarray  # Alta-Baixa (detalhes verticais)
    HH: np.ndarray  # Alta-Alta (detalhes diagonais)
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Converte para dicionário."""
        return {'LL': self.LL, 'LH': self.LH, 'HL': self.HL, 'HH': self.HH}

@dataclass
class QualityMetrics:
    """Armazena métricas de qualidade da reconstrução."""    
    mse: float
    psnr: float
# ---------------------------- # 

# ----- FILTROS QMF ----- #
class QMFFilterBank:
    """Banco de filtros QMF Daubechies 5."""

    @staticmethod
    def get_daubechies5() -> QMFFilters:
        """
        Retorna os filtros QMF Daubechies 5 (db5).
        
        Returns:
            QMFFilters: Objeto com filtros de análise e síntese
        """
        # Filtros de análise
        h0 = np.array([
            0.003335725, -0.012580751, -0.006241490, 0.077571493, 
            -0.032244869, -0.242294887, 0.138428146, 0.724308528, 
            0.603829269, 0.160102398
        ])
        
        h1 = np.array([
            0.160102398, -0.603829269, 0.724308528, -0.138428146, 
            -0.242294887, 0.032244869, 0.077571493, 0.006241490, 
            -0.012580751, -0.003335725
        ])
        
        # Filtros de síntese
        g0 = np.array([
            0.160102398, 0.603829269, 0.724308528, 0.138428146, 
            -0.242294887, -0.032244869, 0.077571493, -0.006241490, 
            -0.012580751, 0.003335725
        ])
        
        g1 = np.array([
            -0.003335725, -0.012580751, 0.006241490, 0.077571493, 
            0.032244869, -0.242294887, -0.138428146, 0.724308528, 
            -0.603829269, 0.160102398
        ])
        
        return QMFFilters(h0=h0, h1=h1, g0=g0, g1=g1)
# ----------------------- #

# ----- OPERAÇÕES DE AMOSTRAGEM ----- #
class SamplingOps:
    """Operações de upsampling e downsampling."""

    @staticmethod
    def downsample(x: np.ndarray, factor: int = 2) -> np.ndarray:
        """Downsampling por fator especificado."""
        return x[::factor]
    
    @staticmethod
    def upsample(x: np.ndarray, factor: int = 2) -> np.ndarray:
        """Upsampling por fator especificado (inserção de zeros)."""
        y = np.zeros(len(x) * factor)
        y[::factor] = x
        return y
# ----------------------------------- #

# ----- DECOMPOSIÇÃO QMF ----- #
def apply_filter(x, h):
    y = signal.convolve(x, h, mode = 'same')
    return  y

class QMFDecomposer:
    """Realiza decomposição QMF de imagens."""

    def __init__(self, filters: QMFFilters):
        self.filters = filters
        self.sampling = SamplingOps()
    
    def decompose_1d(self, signal_1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decomposição QMF 1D (uma linha ou coluna).
        
        Args:
            signal_1d: Sinal 1D a processar
        
        Returns:
            Tupla (low, high) com componentes de baixa e alta frequência
        """
        # Convolução com filtros
        low = apply_filter(signal_1d, self.filters.h0)
        high = apply_filter(signal_1d, self.filters.h1)
        
        # Downsampling
        low = self.sampling.downsample(low)
        high = self.sampling.downsample(high)
        
        return low, high
    
    def decompose_2d(self, image: np.ndarray) -> Subbands:
        """
        Decomposição QMF 2D completa de uma imagem.
        Implementa decomposição separável (linhas → colunas).
        
        Args:
            image: Imagem em escala de cinza (2D array)
        
        Returns:
            Subbands: Objeto com as quatro subbandas
        """
        height, width = image.shape
        
        # ETAPA 1: Decomposição por linhas
        rows_L_list = []
        rows_H_list = []
        
        for i in range(height):
            low, high = self.decompose_1d(image[i, :])
            rows_L_list.append(low)
            rows_H_list.append(high)
        
        rows_L = np.array(rows_L_list)
        rows_H = np.array(rows_H_list)
        
        # ETAPA 2: Decomposição por colunas
        half_width = rows_L.shape[1]
        
        LL_list = []
        LH_list = []
        HL_list = []
        HH_list = []
        
        for j in range(half_width):
            ll, lh = self.decompose_1d(rows_L[:, j])
            hl, hh = self.decompose_1d(rows_H[:, j])
            
            LL_list.append(ll)
            LH_list.append(lh)
            HL_list.append(hl)
            HH_list.append(hh)
        
        # Converter para arrays 2D (transpor para formato correto)
        LL = np.array(LL_list).T
        LH = np.array(LH_list).T
        HL = np.array(HL_list).T
        HH = np.array(HH_list).T
        
        return Subbands(LL=LL, LH=LH, HL=HL, HH=HH)
# ---------------------------- #

# ----- RECONSTRUÇÃO QMF ----- #
class QMFReconstructor:
    """Realiza reconstrução QMF de imagens."""
    
    def __init__(self, filters: QMFFilters):
        self.filters = filters
        self.sampling = SamplingOps()
    
    def reconstruct_1d(self, low: np.ndarray, high: np.ndarray, 
                       target_length: int) -> np.ndarray:
        """
        Reconstrução QMF 1D.
        
        Args:
            low: Componente de baixa frequência
            high: Componente de alta frequência
            target_length: Comprimento alvo do sinal reconstruído
        
        Returns:
            Sinal reconstruído
        """
        # Upsampling
        low_up = self.sampling.upsample(low)
        high_up = self.sampling.upsample(high)
        
        # Filtragem
        low_filt = apply_filter(low_up, self.filters.g0)
        high_filt = apply_filter(high_up, self.filters.g1)
        
        # Soma e ajuste de comprimento
        reconstructed = low_filt + high_filt
        
        if len(reconstructed) > target_length:
            reconstructed = reconstructed[:target_length]
        elif len(reconstructed) < target_length:
            reconstructed = np.pad(reconstructed, (0, target_length - len(reconstructed)))
        
        return reconstructed
    
    def reconstruct_2d(self, subbands: Subbands, 
                       target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Reconstrução QMF 2D completa.
        
        Args:
            subbands: Objeto Subbands com as quatro subbandas
            target_shape: Dimensões da imagem original (height, width)
        
        Returns:
            Imagem reconstruída
        """
        target_height, target_width = target_shape
        half_width = subbands.LL.shape[1]
        
        # ETAPA 1: Reconstrução por colunas
        rows_L_list = []
        rows_H_list = []
        
        for j in range(half_width):
            recon_L = self.reconstruct_1d(
                subbands.LL[:, j], subbands.LH[:, j], target_height
            )
            recon_H = self.reconstruct_1d(
                subbands.HL[:, j], subbands.HH[:, j], target_height
            )
            rows_L_list.append(recon_L)
            rows_H_list.append(recon_H)
        
        rows_L = np.array(rows_L_list).T  # Transpor para ter linhas x colunas
        rows_H = np.array(rows_H_list).T
        
        # ETAPA 2: Reconstrução por linhas
        reconstructed_list = []
        
        for i in range(min(rows_L.shape[0], target_height)):
            recon_row = self.reconstruct_1d(
                rows_L[i, :], rows_H[i, :], target_width
            )
            reconstructed_list.append(recon_row)
        
        reconstructed = np.array(reconstructed_list)
        
        # Ajustar às dimensões exatas
        if reconstructed.shape[0] > target_height:
            reconstructed = reconstructed[:target_height, :]
        if reconstructed.shape[1] > target_width:
            reconstructed = reconstructed[:, :target_width]
        
        # Preencher se necessário
        if reconstructed.shape != target_shape:
            padded = np.zeros(target_shape)
            padded[:reconstructed.shape[0], :reconstructed.shape[1]] = reconstructed
            reconstructed = padded
        
        return reconstructed
# ---------------------------- #

# ----- MÉTRICAS E AVALIAÇÃO ----- #
def scale_for_visualization(data: np.ndarray) -> np.ndarray:
    """
    Escala dados para visualização (normaliza para [0, 1]).
    
    Args:
        data: Array de entrada
        
    Returns:
        Array normalizado
    """
    data_abs = np.abs(data)
    data_min = data_abs.min()
    data_max = data_abs.max()
    
    if data_max - data_min < 1e-10:
        return np.zeros_like(data_abs)
    
    return (data_abs - data_min) / (data_max - data_min)

class ImageQualityEvaluator:
    """Calcula métricas de qualidade de imagem."""
    
    @staticmethod
    def calculate_metrics(original: np.ndarray, 
                         reconstructed: np.ndarray) -> QualityMetrics:
        """
        Calcula MSE e PSNR entre imagem original e reconstruída.
        
        Args:
            original: Imagem original
            reconstructed: Imagem reconstruída
        
        Returns:
            QualityMetrics: Objeto com MSE e PSNR
        """
        mse = np.mean((original - reconstructed) ** 2)
        
        if mse == 0:
            psnr = float('inf')
        else:
            max_pixel = np.max(original)  # Assumindo imagem normalizada [0, 1]
            psnr = 10 * np.log10(max_pixel ** 2 / mse)
        
        return QualityMetrics(mse=mse, psnr=psnr)
# -------------------------------- #

# ----- VISUALIZAÇÃO ----- #
class QMFVisualizer:
    """Visualiza resultados da decomposição QMF."""
    
    @staticmethod
    def _normalize_for_display(x: np.ndarray) -> np.ndarray:
        """Normaliza array para visualização."""
        return (x - x.min()) / (x.max() - x.min() + 1e-10)
    
    def plot_decomposition(self, original: np.ndarray, subbands: Subbands,
                          reconstructed: np.ndarray, metrics: QualityMetrics,
                          filename: str = 'qmf_decomposition.png'):
        """
        Visualiza decomposição QMF conforme Figura 5 do artigo.
        
        Args:
            original: Imagem original
            subbands: Objeto com as subbandas
            reconstructed: Imagem reconstruída
            metrics: Métricas de qualidade
            filename: Nome do arquivo para salvar
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Imagem Original
        plt.subplot(1, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.title('Imagem Original', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Decomposição (Figura 5) - Grade 2x2
        plt.subplot(1, 3, 2)
        
        # Criar visualização 2x2 das subbandas
        top = np.hstack([
            self._normalize_for_display(subbands.LL),
            self._normalize_for_display(subbands.LH)
        ])
        bottom = np.hstack([
            self._normalize_for_display(subbands.HL),
            self._normalize_for_display(subbands.HH)
        ])
        decomposed = np.vstack([top, bottom])
        
        plt.imshow(decomposed, cmap='gray')
        plt.title('Decomposição QMF\n(LL | LH)\n(HL | HH)', 
                  fontsize=14, fontweight='bold')
        
        # Adicionar labels
        h, w = subbands.LL.shape
        labels = [
            (w/2, h/2, 'LL\n(Baixa-Baixa)'),
            (w + w/2, h/2, 'LH\n(Baixa-Alta)'),
            (w/2, h + h/2, 'HL\n(Alta-Baixa)'),
            (w + w/2, h + h/2, 'HH\n(Alta-Alta)')
        ]
        
        for x, y, text in labels:
            plt.text(x, y, text, ha='center', va='center',
                    color='red', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.axis('off')
        
        # Imagem Reconstruída
        plt.subplot(1, 3, 3)
        plt.imshow(scale_for_visualization(reconstructed), cmap='gray')
        plt.title(f'Imagem Reconstruída\nPSNR: {metrics.psnr:.2f} dB\n'
                 f'MSE: {metrics.mse:.2e}', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Figura salva como '{filename}'")
        plt.show()
    
    def plot_subbands_detailed(self, subbands: Subbands,
                              filename: str = 'qmf_subbands_detailed.png'):
        """
        Visualiza cada subbanda separadamente para análise detalhada.
        
        Args:
            subbands: Objeto com as subbandas
            filename: Nome do arquivo para salvar
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        configs = [
            ('LL', 'LL (Aproximação)'),
            ('LH', 'LH (Detalhes Horizontais)'),
            ('HL', 'HL (Detalhes Verticais)'),
            ('HH', 'HH (Detalhes Diagonais)')
        ]
        
        for ax, (band_name, title) in zip(axes.flat, configs):
            data = getattr(subbands, band_name)
            im = ax.imshow(scale_for_visualization(data), cmap='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f" Figura salva como '{filename}'")
        plt.show()
# ------------------------ #

# ----- CARREGAMENTO E GERAÇÃO DE IMAGENS ----- #
class ImageLoader:
    """Carrega ou gera imagens para processamento."""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """
        Carrega imagem do disco.
        
        Args:
            image_path: Caminho para a imagem
        
        Returns:
            Imagem normalizada em escala de cinza [0, 1]
        """
        img = Image.open(image_path).convert('F')
        img = np.array(img, dtype=np.float64)
        return img / np.max(img)
    
    @staticmethod
    def generate_synthetic_image(size: int = 256) -> np.ndarray:
        """
        Gera imagem sintética para testes.
        
        Args:
            size: Tamanho da imagem (size x size)
        
        Returns:
            Imagem sintética normalizada [0, 1]
        """
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Círculo central
        circle = np.exp(-((X**2 + Y**2) / 0.2))
        
        # Adicionar padrões
        pattern = np.sin(10 * X) * np.cos(10 * Y)
        image = circle + 0.3 * pattern
        
        # Normalizar
        image = (image - image.min()) / (image.max() - image.min())
        return image
# --------------------------------------------- #

# ----- PIPELINE PRINCIPAL ----- #
class QMFPipeline:
    """Pipeline completo para processamento QMF de imagens."""
    
    def __init__(self):
        self.filters = QMFFilterBank.get_daubechies5()
        self.decomposer = QMFDecomposer(self.filters)
        self.reconstructor = QMFReconstructor(self.filters)
        self.evaluator = ImageQualityEvaluator()
        self.visualizer = QMFVisualizer()
        self.loader = ImageLoader()
    
    def process_image(self, image: np.ndarray, verbose: bool = True) -> dict:
        """
        Processa imagem completa: decomposição, reconstrução e avaliação.
        
        Args:
            image: Imagem em escala de cinza normalizada [0, 1]
            verbose: Se True, imprime informações de progresso
        
        Returns:
            Dicionário com resultados do processamento
        """
        if verbose:
            print(f" Dimensões da imagem: {image.shape}")
            print("\n Realizando decomposição QMF...")
        
        # Decomposição
        subbands = self.decomposer.decompose_2d(image)
        
        if verbose:
            print(f"   LL: {subbands.LL.shape}")
            print(f"   LH: {subbands.LH.shape}")
            print(f"   HL: {subbands.HL.shape}")
            print(f"   HH: {subbands.HH.shape}")
            print("\n Reconstruindo imagem...")
        
        # Reconstrução
        reconstructed = self.reconstructor.reconstruct_2d(subbands, image.shape)
        
        # Avaliação
        metrics = self.evaluator.calculate_metrics(image, reconstructed)
        
        if verbose:
            print("\n MÉTRICAS DE QUALIDADE")
            print("=" * 60)
            print(f"   MSE:  {metrics.mse:.6e}")
            print(f"   PSNR: {metrics.psnr:.2f} dB")
            print("=" * 60)
        
        return {
            'subbands': subbands,
            'reconstructed': reconstructed,
            'metrics': metrics
        }
    
    def run(self, image_path: str = None):
        """
        Executa pipeline completo.
        
        Args:
            image_path: Caminho para imagem (None para gerar sintética)
        """
        print("=" * 60)
        print("PROCESSAMENTO DE IMAGENS COM QMF")
        print("IEEE SPS UnB - SEMUNI 2025")
        print("=" * 60)
        print("\n Filtros QMF carregados (Daubechies 5)")
        
        # Carregar ou gerar imagem
        if image_path and Path(image_path).exists():
            print(f"\n Carregando imagem: {image_path}")
            image = self.loader.load_image(image_path)
        else:
            if image_path:
                print(f"\n Imagem '{image_path}' não encontrada.")
            print(" Gerando imagem de teste sintética...")
            image = self.loader.generate_synthetic_image()
        
        # Processar
        results = self.process_image(image)
        
        # Visualizar
        print("\n Gerando visualizações...")
        self.visualizer.plot_decomposition(
            image, 
            results['subbands'],
            results['reconstructed'],
            results['metrics']
        )
        self.visualizer.plot_subbands_detailed(results['subbands'])
        
        print("\n Processamento concluído com sucesso!")
        print("\n INFORMAÇÕES:")
        print("   • LL: Contém a aproximação (baixas frequências)")
        print("   • LH: Detalhes de borda horizontal")
        print("   • HL: Detalhes de borda vertical")
        print("   • HH: Detalhes de borda diagonal")
# ------------------------------ #

# ----- EXECUÇÃO ----- #
def main():
    """Função principal de execução."""
    # Configurar caminho da imagem (ajuste conforme necessário)
    image_path = r"C:\Users\Adm\Documents\Git\Semana-Universitria_UnB-2025.2\Codigos\IMG.jpg"
    
    # Executar pipeline
    pipeline = QMFPipeline()
    pipeline.run(image_path)

if __name__ == '__main__':
    main()
# -------------------- #
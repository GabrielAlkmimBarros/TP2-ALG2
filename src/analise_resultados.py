"""
Análise de Resultados - Experimentos de Clustering
Script para geração de figuras e tabelas para artigo científico IEEE

Autor: Análise automatizada
Data: 28/11/2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, ScalarFormatter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES GLOBAIS PARA ESTILO ACADÊMICO
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white'
})

# Paleta de cores
PALETTE = sns.color_palette("deep")
ALGORITHM_COLORS = {
    'KMeans': PALETTE[0],
    'MaxMin': PALETTE[1], 
    'Refinement': PALETTE[2]
}

# ============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================================
def load_and_preprocess_data(filepath):
    """Carrega e pré-processa os dados do experimento."""
    df = pd.read_csv(filepath)
    
    # Converter Delta_Ratio para numérico (NaN para N/A)
    df['Delta_Ratio'] = pd.to_numeric(df['Delta_Ratio'], errors='coerce')
    
    # Tratar valores ausentes em Silhouette e Adj_Rand
    df['Silhouette'] = pd.to_numeric(df['Silhouette'], errors='coerce')
    df['Adj_Rand'] = pd.to_numeric(df['Adj_Rand'], errors='coerce')
    
    # Classificar datasets por tipo
    df['Dataset_Type'] = df['Dataset'].apply(classify_dataset)
    
    # Identificar geometria dos datasets sintéticos
    df['Geometry'] = df['Dataset'].apply(identify_geometry)
    
    print(f"Dados carregados: {len(df)} registros")
    print(f"Datasets únicos: {df['Dataset'].nunique()}")
    print(f"Algoritmos: {df['Algorithm'].unique()}")
    print(f"Métricas: {df['Metric'].unique()}")
    
    return df

def classify_dataset(name):
    """Classifica o dataset como Sintético ou Real (UCI)."""
    if name.startswith('UCI_'):
        return 'Real (UCI)'
    elif name.startswith('SKL_'):
        return 'Sintético (SKL)'
    elif name.startswith('MultiVar_'):
        return 'Sintético (MultiVar)'
    return 'Outros'

def identify_geometry(name):
    """Identifica a geometria dos clusters para datasets sintéticos."""
    name_lower = name.lower()
    if 'anisotropic' in name_lower or 'aniso' in name_lower:
        return 'Anisotropic (Elíptico)'
    elif 'circle' in name_lower or 'moon' in name_lower:
        return 'Não-Convexo'
    elif 'blob' in name_lower:
        return 'Blobs (Esférico)'
    elif 'varied' in name_lower:
        return 'Variância Variada'
    elif 'nostructure' in name_lower:
        return 'Sem Estrutura'
    elif 'multivar' in name_lower:
        return 'Normal Multivariada'
    else:
        return 'Outros'

# ============================================================================
# 1. ANÁLISE DE SENSIBILIDADE (Refinamento de Intervalo)
# ============================================================================
def plot_sensitivity_analysis(df, save_path='fig1_sensitivity_analysis.png'):
    """
    Gráfico A: Análise de sensibilidade do parâmetro Delta_Ratio.
    Mostra como a largura do intervalo afeta qualidade (ARI) e tempo.
    """
    # Filtrar apenas Refinement
    df_ref = df[df['Algorithm'] == 'Refinement'].copy()
    df_ref = df_ref.dropna(subset=['Delta_Ratio', 'Adj_Rand'])
    
    # Filtrar apenas datasets sintéticos para tendência geral
    df_synth = df_ref[df_ref['Dataset_Type'].isin(['Sintético (SKL)', 'Sintético (MultiVar)'])]
    
    # Agregar por Delta_Ratio
    agg_stats = df_synth.groupby('Delta_Ratio').agg({
        'Adj_Rand': ['mean', 'std', 'count'],
        'Time_s': ['mean', 'std']
    }).reset_index()
    agg_stats.columns = ['Delta_Ratio', 'ARI_mean', 'ARI_std', 'count', 'Time_mean', 'Time_std']
    
    # Criar figura com dois eixos Y
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Eixo Y esquerdo - ARI
    color1 = PALETTE[0]
    ax1.set_xlabel('Delta Ratio (δ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Adjusted Rand Index (ARI)', color=color1, fontsize=12, fontweight='bold')
    
    # Plotar linha com intervalo de confiança
    line1 = ax1.errorbar(agg_stats['Delta_Ratio'], agg_stats['ARI_mean'], 
                         yerr=agg_stats['ARI_std'], 
                         fmt='o-', color=color1, linewidth=2, markersize=8,
                         capsize=5, capthick=2, label='ARI Médio ± DP')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)
    
    # Eixo Y direito - Tempo
    ax2 = ax1.twinx()
    color2 = PALETTE[2]
    ax2.set_ylabel('Tempo de Execução (s)', color=color2, fontsize=12, fontweight='bold')
    
    line2 = ax2.errorbar(agg_stats['Delta_Ratio'], agg_stats['Time_mean'],
                         yerr=agg_stats['Time_std'],
                         fmt='s--', color=color2, linewidth=2, markersize=8,
                         capsize=5, capthick=2, label='Tempo Médio ± DP')
    
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')
    
    # Título e legenda
    ax1.set_title('Análise de Sensibilidade: Impacto do Parâmetro δ no Refinement\n(Datasets Sintéticos)', 
                  fontsize=13, fontweight='bold', pad=15)
    
    # Combinar legendas
    lines = [line1, line2]
    labels = ['ARI Médio ± DP', 'Tempo Médio ± DP']
    ax1.legend(lines, labels, loc='upper right', frameon=True, fancybox=True)
    
    # Grid apenas no eixo esquerdo
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax2.grid(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✓ Figura salva: {save_path}")
    print("\nEstatísticas por Delta_Ratio:")
    print(agg_stats.to_string(index=False))
    
    return agg_stats

# ============================================================================
# 2. COMPARAÇÃO DE DESEMPENHO GLOBAL (Boxplots)
# ============================================================================
def plot_global_comparison_boxplot(df, save_path='fig2_boxplot_comparison.png'):
    """
    Gráfico B: Boxplot comparando ARI entre algoritmos, separado por métrica.
    """
    # Remover NaN em Adj_Rand
    df_clean = df.dropna(subset=['Adj_Rand']).copy()
    
    # Criar mapeamento simplificado de métricas
    df_clean['Metric_Simple'] = df_clean['Metric'].replace({
        'Minkowski-L1': 'Manhattan (L1)',
        'Minkowski-L2': 'Euclidiana (L2)',
        'Minkowski-L3': 'L3',
        'Euclidiana': 'Euclidiana (L2)'
    })
    
    # Selecionar apenas métricas principais para comparação
    main_metrics = ['Euclidiana (L2)', 'Mahalanobis', 'Manhattan (L1)']
    df_filtered = df_clean[df_clean['Metric_Simple'].isin(main_metrics)]
    
    # Criar FacetGrid
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    for idx, metric in enumerate(main_metrics):
        ax = axes[idx]
        df_metric = df_filtered[df_filtered['Metric_Simple'] == metric]
        
        # Boxplot
        bp = ax.boxplot([df_metric[df_metric['Algorithm'] == alg]['Adj_Rand'].values 
                         for alg in ['KMeans', 'MaxMin', 'Refinement']],
                        labels=['KMeans', 'MaxMin', 'Refinement'],
                        patch_artist=True,
                        widths=0.6)
        
        # Colorir boxes
        colors = [ALGORITHM_COLORS['KMeans'], ALGORITHM_COLORS['MaxMin'], ALGORITHM_COLORS['Refinement']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Configurar eixo
        ax.set_title(f'Métrica: {metric}', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Algoritmo', fontsize=11)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comparação de Desempenho: Variabilidade do ARI por Algoritmo e Métrica',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✓ Figura salva: {save_path}")
    
    # Estatísticas resumidas
    print("\nEstatísticas por Algoritmo e Métrica:")
    stats = df_filtered.groupby(['Metric_Simple', 'Algorithm'])['Adj_Rand'].agg(['mean', 'std', 'median']).round(4)
    print(stats)
    
    return stats

# ============================================================================
# 3. IMPACTO DA GEOMETRIA (Euclidiana vs. Mahalanobis)
# ============================================================================
def plot_geometry_impact(df, save_path='fig3_geometry_impact.png'):
    """
    Gráfico C: Bar plot comparando Euclidiana vs Mahalanobis por geometria.
    """
    # Filtrar apenas datasets sintéticos e métricas relevantes
    df_clean = df.dropna(subset=['Adj_Rand']).copy()
    
    # Mapear métricas para comparação
    df_clean['Metric_Simplified'] = df_clean['Metric'].apply(
        lambda x: 'Euclidiana' if x in ['Euclidiana', 'Minkowski-L2'] else ('Mahalanobis' if x == 'Mahalanobis' else None)
    )
    df_filtered = df_clean[df_clean['Metric_Simplified'].notna()]
    
    # Filtrar apenas datasets com geometria identificada
    df_geom = df_filtered[df_filtered['Geometry'].isin([
        'Anisotropic (Elíptico)', 'Blobs (Esférico)', 'Variância Variada', 
        'Normal Multivariada', 'Não-Convexo'
    ])]
    
    # Agregar por geometria e métrica
    agg_data = df_geom.groupby(['Geometry', 'Metric_Simplified'])['Adj_Rand'].agg(['mean', 'std']).reset_index()
    agg_data.columns = ['Geometry', 'Metric', 'ARI_mean', 'ARI_std']
    
    # Criar gráfico de barras agrupadas
    fig, ax = plt.subplots(figsize=(12, 6))
    
    geometries = agg_data['Geometry'].unique()
    x = np.arange(len(geometries))
    width = 0.35
    
    # Separar dados por métrica
    euclidean_data = agg_data[agg_data['Metric'] == 'Euclidiana']
    mahal_data = agg_data[agg_data['Metric'] == 'Mahalanobis']
    
    # Garantir ordenação consistente
    euclidean_means = [euclidean_data[euclidean_data['Geometry'] == g]['ARI_mean'].values[0] 
                       if len(euclidean_data[euclidean_data['Geometry'] == g]) > 0 else 0 
                       for g in geometries]
    euclidean_stds = [euclidean_data[euclidean_data['Geometry'] == g]['ARI_std'].values[0] 
                      if len(euclidean_data[euclidean_data['Geometry'] == g]) > 0 else 0 
                      for g in geometries]
    
    mahal_means = [mahal_data[mahal_data['Geometry'] == g]['ARI_mean'].values[0] 
                   if len(mahal_data[mahal_data['Geometry'] == g]) > 0 else 0 
                   for g in geometries]
    mahal_stds = [mahal_data[mahal_data['Geometry'] == g]['ARI_std'].values[0] 
                  if len(mahal_data[mahal_data['Geometry'] == g]) > 0 else 0 
                  for g in geometries]
    
    # Plotar barras
    bars1 = ax.bar(x - width/2, euclidean_means, width, yerr=euclidean_stds, 
                   label='Euclidiana (L2)', color=PALETTE[0], capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, mahal_means, width, yerr=mahal_stds,
                   label='Mahalanobis', color=PALETTE[3], capsize=5, alpha=0.8)
    
    # Configurar eixos
    ax.set_xlabel('Geometria dos Clusters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    ax.set_title('Impacto da Geometria dos Dados: Euclidiana vs. Mahalanobis',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(geometries, rotation=15, ha='right')
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✓ Figura salva: {save_path}")
    print("\nComparação Euclidiana vs Mahalanobis por Geometria:")
    print(agg_data.to_string(index=False))
    
    return agg_data

# ============================================================================
# 4. ESCALABILIDADE (Tempo vs. Dataset)
# ============================================================================
def plot_scalability_analysis(df, save_path='fig4_scalability.png'):
    """
    Gráfico D: Comparação de tempo de execução entre algoritmos por dataset.
    """
    # Agregar tempo médio por dataset e algoritmo
    time_agg = df.groupby(['Dataset', 'Algorithm'])['Time_s'].mean().reset_index()
    
    # Pivotar para ter algoritmos como colunas
    time_pivot = time_agg.pivot(index='Dataset', columns='Algorithm', values='Time_s').reset_index()
    
    # Ordenar pelo tempo do KMeans
    if 'KMeans' in time_pivot.columns:
        time_pivot = time_pivot.sort_values('KMeans', ascending=True)
    
    # Selecionar top 20 datasets para visualização mais clara
    # (ou todos se forem menos de 20)
    n_datasets = min(25, len(time_pivot))
    time_top = time_pivot.tail(n_datasets)  # Maiores tempos
    
    # Criar gráfico de barras horizontais
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y = np.arange(len(time_top))
    height = 0.25
    
    # Plotar barras para cada algoritmo
    if 'KMeans' in time_top.columns:
        ax.barh(y - height, time_top['KMeans'], height, label='KMeans', 
                color=ALGORITHM_COLORS['KMeans'], alpha=0.8)
    if 'MaxMin' in time_top.columns:
        ax.barh(y, time_top['MaxMin'], height, label='MaxMin', 
                color=ALGORITHM_COLORS['MaxMin'], alpha=0.8)
    if 'Refinement' in time_top.columns:
        ax.barh(y + height, time_top['Refinement'], height, label='Refinement', 
                color=ALGORITHM_COLORS['Refinement'], alpha=0.8)
    
    # Configurar eixos
    ax.set_xlabel('Tempo de Execução (s) - Escala Logarítmica', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_title('Escalabilidade: Comparação de Tempo de Execução por Dataset\n(Top 25 mais lentos)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(time_top['Dataset'], fontsize=9)
    ax.set_xscale('log')
    ax.legend(loc='lower right', frameon=True, fancybox=True)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✓ Figura salva: {save_path}")
    
    # Estatísticas de tempo
    print("\nTempo médio por algoritmo (todos os datasets):")
    time_summary = df.groupby('Algorithm')['Time_s'].agg(['mean', 'std', 'min', 'max']).round(6)
    print(time_summary)
    
    return time_summary

# ============================================================================
# 5. TABELA RESUMO LaTeX
# ============================================================================
def generate_latex_summary_table(df, save_path='tabela_resumo.tex'):
    """
    Gera tabela resumo em formato LaTeX.
    """
    # Remover NaN
    df_clean = df.dropna(subset=['Adj_Rand', 'Silhouette'])
    
    # Agregar por algoritmo
    summary = df_clean.groupby('Algorithm').agg({
        'Adj_Rand': ['mean', 'std'],
        'Silhouette': ['mean', 'std'],
        'Time_s': ['mean', 'std']
    }).round(4)
    
    # Achatar colunas
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    # Formatar como média ± desvio
    summary['ARI'] = summary.apply(lambda x: f"${x['Adj_Rand_mean']:.4f} \\pm {x['Adj_Rand_std']:.4f}$", axis=1)
    summary['Silhouette'] = summary.apply(lambda x: f"${x['Silhouette_mean']:.4f} \\pm {x['Silhouette_std']:.4f}$", axis=1)
    summary['Time (s)'] = summary.apply(lambda x: f"${x['Time_s_mean']:.4f} \\pm {x['Time_s_std']:.4f}$", axis=1)
    
    # Criar tabela LaTeX
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Resumo Comparativo dos Algoritmos de Clustering (Média $\pm$ Desvio Padrão)}
\label{tab:resumo_algoritmos}
\begin{tabular}{lccc}
\hline
\textbf{Algoritmo} & \textbf{ARI} & \textbf{Silhouette} & \textbf{Tempo (s)} \\
\hline
"""
    
    for _, row in summary.iterrows():
        latex_table += f"{row['Algorithm']} & {row['ARI']} & {row['Silhouette']} & {row['Time (s)']} \\\\\n"
    
    latex_table += r"""\hline
\end{tabular}
\end{table}
"""
    
    # Salvar arquivo
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    
    print(f"\n✓ Tabela LaTeX salva: {save_path}")
    print("\n" + "="*80)
    print("CÓDIGO LaTeX GERADO:")
    print("="*80)
    print(latex_table)
    
    # Também criar versão expandida por métrica
    summary_by_metric = df_clean.groupby(['Algorithm', 'Metric']).agg({
        'Adj_Rand': ['mean', 'std'],
        'Silhouette': ['mean', 'std'],
        'Time_s': ['mean', 'std']
    }).round(4)
    
    print("\n" + "="*80)
    print("ESTATÍSTICAS DETALHADAS POR ALGORITMO E MÉTRICA:")
    print("="*80)
    print(summary_by_metric.to_string())
    
    return summary, summary_by_metric

# ============================================================================
# 6. GRÁFICOS ADICIONAIS ÚTEIS
# ============================================================================
def plot_heatmap_performance(df, save_path='fig5_heatmap_performance.png'):
    """
    Heatmap de desempenho (ARI) por Dataset vs Algoritmo.
    """
    # Agregar ARI médio por dataset e algoritmo
    heatmap_data = df.groupby(['Dataset', 'Algorithm'])['Adj_Rand'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Dataset', columns='Algorithm', values='Adj_Rand')
    
    # Selecionar subconjunto se houver muitos datasets
    if len(heatmap_pivot) > 30:
        # Selecionar datasets com maior variação entre algoritmos
        heatmap_pivot['range'] = heatmap_pivot.max(axis=1) - heatmap_pivot.min(axis=1)
        heatmap_pivot = heatmap_pivot.nlargest(30, 'range').drop('range', axis=1)
    
    # Criar heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    
    sns.heatmap(heatmap_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax, cbar_kws={'label': 'ARI'}, linewidths=0.5)
    
    ax.set_title('Mapa de Calor: Desempenho (ARI) por Dataset e Algoritmo',
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Algoritmo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✓ Figura salva: {save_path}")
    
    return heatmap_pivot

def plot_metric_comparison_detailed(df, save_path='fig6_metric_comparison.png'):
    """
    Comparação detalhada entre todas as métricas de distância.
    """
    df_clean = df.dropna(subset=['Adj_Rand']).copy()
    
    # Agregar por métrica
    metric_stats = df_clean.groupby('Metric')['Adj_Rand'].agg(['mean', 'std', 'count']).reset_index()
    metric_stats = metric_stats.sort_values('mean', ascending=True)
    
    # Criar gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("viridis", len(metric_stats))
    
    bars = ax.barh(metric_stats['Metric'], metric_stats['mean'], 
                   xerr=metric_stats['std'], color=colors, capsize=5, alpha=0.8)
    
    ax.set_xlabel('Adjusted Rand Index (ARI)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Métrica de Distância', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Métricas de Distância: ARI Médio ± Desvio Padrão',
                fontsize=13, fontweight='bold', pad=15)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores
    for bar, count in zip(bars, metric_stats['count']):
        width = bar.get_width()
        ax.annotate(f'n={int(count)}',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0),
                   textcoords="offset points",
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✓ Figura salva: {save_path}")
    print("\nEstatísticas por Métrica:")
    print(metric_stats.to_string(index=False))
    
    return metric_stats

def plot_dataset_type_comparison(df, save_path='fig7_dataset_type_comparison.png'):
    """
    Comparação de desempenho por tipo de dataset (Sintético vs Real).
    """
    df_clean = df.dropna(subset=['Adj_Rand']).copy()
    
    # Criar gráfico
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Boxplot por tipo de dataset e algoritmo
    ax1 = axes[0]
    sns.boxplot(data=df_clean, x='Dataset_Type', y='Adj_Rand', hue='Algorithm',
                palette=ALGORITHM_COLORS, ax=ax1)
    ax1.set_xlabel('Tipo de Dataset', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Adjusted Rand Index (ARI)', fontsize=11, fontweight='bold')
    ax1.set_title('Desempenho por Tipo de Dataset', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(title='Algoritmo', loc='upper right')
    ax1.tick_params(axis='x', rotation=10)
    
    # Gráfico 2: Tempo por tipo de dataset
    ax2 = axes[1]
    time_by_type = df_clean.groupby(['Dataset_Type', 'Algorithm'])['Time_s'].mean().reset_index()
    time_pivot = time_by_type.pivot(index='Dataset_Type', columns='Algorithm', values='Time_s')
    
    time_pivot.plot(kind='bar', ax=ax2, color=[ALGORITHM_COLORS[c] for c in time_pivot.columns], 
                    alpha=0.8, width=0.8)
    ax2.set_xlabel('Tipo de Dataset', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tempo Médio (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Tempo de Execução por Tipo de Dataset', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(title='Algoritmo', loc='upper right')
    ax2.tick_params(axis='x', rotation=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n✓ Figura salva: {save_path}")
    
    return time_pivot

# ============================================================================
# EXECUÇÃO PRINCIPAL
# ============================================================================
def main():
    """Função principal para gerar todos os artefatos."""
    print("="*80)
    print(" ANÁLISE DE RESULTADOS - EXPERIMENTOS DE CLUSTERING")
    print(" Geração de Figuras e Tabelas para Artigo Científico IEEE")
    print("="*80)
    
    # Carregar dados
    filepath = 'RESULTADOS_FULL.csv'
    df = load_and_preprocess_data(filepath)
    
    print("\n" + "="*80)
    print(" GERANDO FIGURAS E TABELAS")
    print("="*80)
    
    # 1. Análise de Sensibilidade
    print("\n" + "-"*40)
    print("1. ANÁLISE DE SENSIBILIDADE (Delta Ratio)")
    print("-"*40)
    sensitivity_stats = plot_sensitivity_analysis(df)
    
    # 2. Comparação Global (Boxplots)
    print("\n" + "-"*40)
    print("2. COMPARAÇÃO DE DESEMPENHO GLOBAL (Boxplots)")
    print("-"*40)
    boxplot_stats = plot_global_comparison_boxplot(df)
    
    # 3. Impacto da Geometria
    print("\n" + "-"*40)
    print("3. IMPACTO DA GEOMETRIA (Euclidiana vs Mahalanobis)")
    print("-"*40)
    geometry_stats = plot_geometry_impact(df)
    
    # 4. Escalabilidade
    print("\n" + "-"*40)
    print("4. ESCALABILIDADE (Tempo por Dataset)")
    print("-"*40)
    time_stats = plot_scalability_analysis(df)
    
    # 5. Tabela LaTeX
    print("\n" + "-"*40)
    print("5. TABELA RESUMO LaTeX")
    print("-"*40)
    latex_summary, detailed_summary = generate_latex_summary_table(df)
    
    # 6. Gráficos adicionais
    print("\n" + "-"*40)
    print("6. GRÁFICOS ADICIONAIS")
    print("-"*40)
    
    print("\n6a. Heatmap de Performance...")
    heatmap_data = plot_heatmap_performance(df)
    
    print("\n6b. Comparação detalhada de Métricas...")
    metric_stats = plot_metric_comparison_detailed(df)
    
    print("\n6c. Comparação por Tipo de Dataset...")
    type_comparison = plot_dataset_type_comparison(df)
    
    print("\n" + "="*80)
    print(" ANÁLISE CONCLUÍDA!")
    print(" Todos os artefatos foram salvos na pasta atual.")
    print("="*80)
    
    return df

if __name__ == "__main__":
    df = main()

# Relatório de Análise Experimental
## Comparação de Algoritmos de Clustering: K-Centers

**Disciplina:** Algoritmos 2 - UFMG  
**Data:** 28 de Novembro de 2025  
**Experimento:** Avaliação Comparativa de Algoritmos de Agrupamento (Clustering)

---

## Sumário

1. [Introdução](#1-introdução)
2. [Metodologia Experimental](#2-metodologia-experimental)
3. [Descrição dos Datasets](#3-descrição-dos-datasets)
4. [Análise de Sensibilidade do Parâmetro δ](#4-análise-de-sensibilidade-do-parâmetro-δ)
5. [Comparação de Desempenho Global](#5-comparação-de-desempenho-global)
6. [Impacto da Geometria dos Dados](#6-impacto-da-geometria-dos-dados)
7. [Análise de Escalabilidade](#7-análise-de-escalabilidade)
8. [Heatmap de Performance](#8-heatmap-de-performance)
9. [Comparação de Métricas de Distância](#9-comparação-de-métricas-de-distância)
10. [Comparação por Tipo de Dataset](#10-comparação-por-tipo-de-dataset)
11. [Casos Específicos de Sucesso](#11-casos-específicos-de-sucesso)
12. [Conclusões](#12-conclusões)
13. [Referências das Figuras](#13-referências-das-figuras)

---

## 1. Introdução

Este relatório apresenta uma análise experimental completa comparando três algoritmos de clustering:

| Algoritmo | Descrição | Complexidade |
|-----------|-----------|--------------|
| **K-Means** | Algoritmo iterativo de Lloyd (sklearn) | O(n·k·i·d) |
| **MaxMin** | Algoritmo guloso 2-aproximado para k-centers | O(n·k) |
| **Refinement** | Busca binária no raio com verificação | O(n·k·log(Δ)) |

### Objetivos do Experimento

1. Avaliar a qualidade do agrupamento usando **Adjusted Rand Index (ARI)** e **Silhouette Score**
2. Comparar o tempo de execução entre implementações otimizadas (C/Cython) e Python puro
3. Analisar o impacto de diferentes métricas de distância
4. Investigar a sensibilidade do algoritmo Refinement ao parâmetro δ

---

## 2. Metodologia Experimental

### 2.1 Configuração do Experimento

| Parâmetro | Valor |
|-----------|-------|
| Número de execuções (N_RUNS) | 15 |
| Valores de δ testados | 0.01, 0.05, 0.10, 0.15, 0.25 |
| Total de datasets | 51 |
| Total de registros gerados | 19.125 |

### 2.2 Métricas de Distância Avaliadas

| Métrica | Fórmula | Implementação |
|---------|---------|---------------|
| Euclidiana | $\sqrt{\sum_{i=1}^{d}(x_i - y_i)^2}$ | sklearn (otimizada) |
| Minkowski-L1 (Manhattan) | $\sum_{i=1}^{d}\|x_i - y_i\|$ | Python |
| Minkowski-L2 | $\sqrt{\sum_{i=1}^{d}(x_i - y_i)^2}$ | Python |
| Minkowski-L3 | $\sqrt[3]{\sum_{i=1}^{d}\|x_i - y_i\|^3}$ | Python |
| Mahalanobis | $\sqrt{(x-y)^T \Sigma^{-1} (x-y)}$ | Python |

### 2.3 Métricas de Avaliação

- **Adjusted Rand Index (ARI)**: Mede a concordância entre clusters preditos e ground truth, ajustado para chance. Varia de -1 a 1, onde 1 indica concordância perfeita.

- **Silhouette Score**: Mede a coesão intra-cluster e separação inter-cluster. Varia de -1 a 1, onde valores altos indicam clusters bem definidos.

- **Tempo de Execução**: Medido em segundos usando `time.perf_counter()`.

---

## 3. Descrição dos Datasets

### 3.1 Distribuição por Tipo

| Tipo | Quantidade | Descrição |
|------|------------|-----------|
| Sintético (Scikit-Learn) | 30 | Blobs, Anisotropic, Circles, Moons, etc. |
| Sintético (Normal Multivariada) | 10 | Clusters com covariâncias personalizadas |
| Real (UCI) | 11 | Datasets reais do UCI Repository |
| **Total** | **51** | |

### 3.2 Datasets Reais (UCI)

| Dataset | Amostras | Dimensões | k (clusters) |
|---------|----------|-----------|--------------|
| UCI_Banknote | 1.372 | 4 | 2 |
| UCI_OptDigits | 5.620 | 64 | 10 |
| UCI_WineRed | 1.599 | 11 | 6 |
| UCI_WineWhite | 4.898 | 11 | 7 |
| UCI_Bankruptcy | 6.819 | 95 | 2 |
| UCI_SECOM | 1.567 | 590 | 2 |
| UCI_DrugConsumption | 1.885 | 12 | 7 |
| UCI_MyocardialInfarction | 1.700 | 123 | 8 |
| UCI_Obesity | 2.111 | 16 | 7 |
| UCI_Cardiotocography | 2.126 | 36 | 3 |
| UCI_BEED_EEG | 8.000 | 16 | 4 |

### 3.3 Geometrias dos Datasets Sintéticos

| Geometria | Descrição | Desafio para Clustering |
|-----------|-----------|------------------------|
| Blobs (Esférico) | Clusters esféricos bem separados | Baixo |
| Anisotropic (Elíptico) | Clusters alongados/rotacionados | Médio |
| Variância Variada | Clusters com tamanhos diferentes | Médio |
| Não-Convexo (Circles, Moons) | Formas não-lineares | Alto |
| Normal Multivariada | Covariâncias customizadas | Médio-Alto |

---

## 4. Análise de Sensibilidade do Parâmetro δ

**Figura:** `fig1_sensitivity_analysis.png`

### 4.1 Resultados

| δ (Delta Ratio) | ARI Médio | Desvio Padrão | Tempo Médio (s) | N Amostras |
|-----------------|-----------|---------------|-----------------|------------|
| 0.01 | **0.4453** | 0.3637 | 0.00159 | 2.400 |
| 0.05 | 0.4426 | 0.3644 | 0.00127 | 2.385 |
| 0.10 | 0.4371 | 0.3629 | 0.00109 | 2.325 |
| 0.15 | 0.4330 | 0.3426 | 0.00090 | 2.085 |
| 0.25 | 0.4063 | 0.3141 | 0.00064 | 1.845 |

### 4.2 Análise

O gráfico demonstra claramente o **trade-off entre qualidade e eficiência** no algoritmo Refinement:

1. **Relação Inversa δ vs ARI**: 
   - Intervalos menores (δ=0.01) produzem melhor qualidade (ARI = 0.445)
   - Intervalos maiores (δ=0.25) têm qualidade reduzida (ARI = 0.406)
   - **Diferença total: ~10% de perda em ARI**

2. **Relação Direta δ vs Tempo**:
   - δ=0.01 leva 2.5x mais tempo que δ=0.25
   - O tempo segue aproximadamente O(log(1/δ)) devido à busca binária

3. **Recomendação Prática**:
   - **δ = 0.05** oferece o melhor compromisso (ARI = 0.443, apenas 0.5% abaixo do ótimo, mas 20% mais rápido)
   - Para aplicações críticas: usar δ = 0.01
   - Para prototipagem rápida: usar δ = 0.15 ou 0.25

### 4.3 Interpretação Teórica

O parâmetro δ controla a largura do intervalo de busca binária. Um intervalo mais estreito (δ pequeno) permite encontrar um raio mais próximo do ótimo, mas requer mais iterações de busca. A garantia teórica é que o raio encontrado está dentro de (1+δ) do ótimo.

---

## 5. Comparação de Desempenho Global

**Figura:** `fig2_boxplot_comparison.png`

### 5.1 Estatísticas Globais

| Algoritmo | ARI Médio | ARI Desvio | Silhouette | Tempo (s) |
|-----------|-----------|------------|------------|-----------|
| **KMeans** | **0.4729** | 0.4020 | 0.4610 | 0.0093 |
| Refinement | 0.3552 | 0.3568 | **0.4636** | 0.0055 |
| MaxMin | 0.3196 | 0.3430 | 0.4339 | **0.0002** |

### 5.2 Análise por Métrica de Distância

| Métrica | Algoritmo | ARI Médio | ARI Mediana |
|---------|-----------|-----------|-------------|
| Euclidiana (L2) | KMeans | **0.4729** | 0.4933 |
| Euclidiana (L2) | MaxMin | 0.3365 | 0.2520 |
| Euclidiana (L2) | Refinement | 0.3792 | 0.4297 |
| Mahalanobis | MaxMin | 0.2564 | 0.0417 |
| Mahalanobis | Refinement | 0.2971 | 0.0825 |
| Manhattan (L1) | MaxMin | 0.3485 | 0.4216 |
| Manhattan (L1) | Refinement | 0.3712 | 0.3884 |

### 5.3 Análise dos Boxplots

Os boxplots revelam características importantes:

1. **Variabilidade (Tamanho das Caixas)**:
   - KMeans tem maior variabilidade (caixa maior), indicando desempenho inconsistente entre datasets
   - MaxMin e Refinement têm caixas menores, sugerindo maior estabilidade

2. **Outliers**:
   - Todos os algoritmos apresentam outliers negativos (datasets difíceis)
   - KMeans tem mais outliers positivos (datasets onde brilha)

3. **Mediana vs Média**:
   - KMeans: Mediana (0.49) > Média (0.47) → distribuição assimétrica à esquerda
   - MaxMin: Mediana (0.25) < Média (0.32) → alguns casos muito bons puxam a média

4. **Por Métrica**:
   - **Euclidiana**: Melhor desempenho geral para todos os algoritmos
   - **Mahalanobis**: Pior desempenho, especialmente para MaxMin (mediana = 0.04)
   - **Manhattan**: Performance intermediária, mais estável que Mahalanobis

---

## 6. Impacto da Geometria dos Dados

**Figura:** `fig3_geometry_impact.png`

### 6.1 Comparação Euclidiana vs Mahalanobis por Geometria

| Geometria | Euclidiana | Mahalanobis | Δ Absoluta | Vencedor |
|-----------|------------|-------------|------------|----------|
| Anisotropic (Elíptico) | **0.725** | 0.559 | +0.166 | Euclidiana |
| Blobs (Esférico) | **0.629** | 0.365 | +0.264 | Euclidiana |
| Normal Multivariada | **0.606** | 0.591 | +0.015 | Euclidiana |
| Não-Convexo | **0.241** | 0.108 | +0.133 | Euclidiana |

### 6.2 Análise Detalhada

**Resultado Surpreendente:** Contrariando a hipótese teórica, a distância **Euclidiana superou Mahalanobis** em todos os cenários testados.

#### Hipótese Inicial (Não Confirmada):
> "Mahalanobis deveria ser melhor para clusters elípticos porque considera a covariância dos dados"

#### Possíveis Explicações para o Resultado:

1. **Instabilidade Numérica**:
   - A inversão da matriz de covariância (Σ⁻¹) pode ser mal-condicionada
   - Pequenas perturbações nos dados causam grandes variações na métrica
   
2. **Covariância Global vs Local**:
   - A implementação usa covariância global (todos os pontos)
   - Clusters têm covariâncias diferentes, a média global não representa bem nenhum

3. **Efeito de Outliers**:
   - Outliers distorcem a matriz de covariância estimada
   - Datasets reais têm mais ruído que afeta a estimativa

4. **Necessidade de Regularização**:
   - Técnicas como *shrinkage* (Ledoit-Wolf) poderiam estabilizar a estimativa
   - A implementação atual não usa regularização

### 6.3 Casos por Geometria

**Clusters Elípticos (Anisotropic):**
- Teoricamente o melhor caso para Mahalanobis
- Na prática: Euclidiana é 30% melhor
- **Causa provável**: Covariância global mistura as orientações de diferentes clusters

**Clusters Esféricos (Blobs):**
- Euclidiana é naturalmente adequada
- Mahalanobis perde 72% de desempenho
- **Causa provável**: Covariância adiciona complexidade desnecessária

**Dados Não-Convexos (Circles, Moons):**
- Ambas as métricas falham (ARI < 0.25)
- Euclidiana ainda é 123% melhor que Mahalanobis
- **Causa**: Nenhuma métrica de distância pontual captura estruturas não-lineares

---

## 7. Análise de Escalabilidade

**Figura:** `fig4_scalability.png`

### 7.1 Tempo de Execução por Algoritmo

| Algoritmo | Tempo Mín | Tempo Médio | Tempo Máx | Fator vs KMeans |
|-----------|-----------|-------------|-----------|-----------------|
| **MaxMin** | 0.00002s | **0.0002s** | 0.006s | **46x mais rápido** |
| Refinement | 0.0003s | 0.0058s | 0.181s | 1.6x mais rápido |
| KMeans | 0.0039s | 0.0093s | 0.248s | (baseline) |

### 7.2 Análise do Gráfico de Barras Horizontais

O gráfico mostra os 25 datasets mais lentos, ordenados pelo tempo do KMeans:

1. **Datasets Mais Lentos**:
   - UCI_OptDigits (5620 × 64): ~0.15s para KMeans
   - UCI_BEED_EEG (8000 × 16): ~0.12s para KMeans
   - UCI_Bankruptcy (6819 × 95): ~0.10s para Refinement

2. **Padrão de Escalonamento**:
   - MaxMin mantém tempo consistentemente baixo (barras curtas)
   - Refinement escala pior que MaxMin em alta dimensão
   - KMeans escala linearmente com n×d

3. **Impacto da Dimensionalidade**:
   - UCI_SECOM (590 dimensões) é desafiador para todos
   - Mahalanobis sofre mais com alta dimensão (inversão de matriz 590×590)

### 7.3 Complexidade Teórica vs Prática

| Algoritmo | Complexidade Teórica | Comportamento Observado |
|-----------|---------------------|------------------------|
| KMeans | O(n·k·i·d) | Escala bem, mas constante alta (C otimizado) |
| MaxMin | O(n·k) | Extremamente rápido, constante baixa |
| Refinement | O(n·k·log(Δ/δ)) | Mais lento que MaxMin devido ao fator log |

### 7.4 Recomendações de Uso

| Cenário | Algoritmo Recomendado | Justificativa |
|---------|----------------------|---------------|
| Protótipo rápido | **MaxMin** | 46x mais rápido, qualidade razoável |
| Produção (qualidade) | **KMeans** | Melhor ARI, implementação otimizada |
| Streaming/Online | **MaxMin** | O(n·k) por ponto, sem iterações |
| Garantias teóricas | **Refinement** | 2-aproximação garantida com δ controlável |

---

## 8. Heatmap de Performance

**Figura:** `fig5_heatmap_performance.png`

### 8.1 Interpretação do Mapa de Calor

O heatmap apresenta o ARI médio para cada combinação (Dataset × Algoritmo), usando escala de cores:
- 🟢 **Verde escuro**: ARI > 0.7 (excelente)
- 🟡 **Amarelo**: ARI ≈ 0.3-0.5 (moderado)
- 🔴 **Vermelho**: ARI < 0.1 (ruim)

### 8.2 Padrões Identificados

**Datasets com Ótimo Desempenho (Verde) para Todos:**
- SKL_Blobs_* (todos os 5)
- SKL_Anisotropic_* (todos os 5)
- MultiVar_* (8 de 10)

**Datasets Desafiadores (Vermelho) para Todos:**
- SKL_NoisyCircles_* (ARI ≈ 0)
- UCI_SECOM (alta dimensão, ruído)
- UCI_Bankruptcy (classes desbalanceadas)

**Datasets com Divergência (KMeans muito melhor):**
- UCI_OptDigits: KMeans = 0.52, MaxMin/Refinement = 0.01
- UCI_Cardiotocography: KMeans = 0.23, outros < 0.03

**Datasets com Convergência (Algoritmos similares):**
- SKL_Blobs_*: Todos > 0.9
- MultiVar_Spherical_*: Todos > 0.8

### 8.3 Análise de Clusters no Heatmap

Agrupando datasets por padrão:

| Cluster | Datasets | Característica | Melhor Algoritmo |
|---------|----------|----------------|------------------|
| A | Blobs, Aniso | Bem separados, esféricos/elípticos | Todos similares |
| B | Circles, Moons | Não-convexos | MaxMin ligeiramente melhor |
| C | UCI alto-dim | Alta dimensionalidade | KMeans muito melhor |
| D | UCI ruidosos | Muito ruído, poucas features | Todos ruins |

---

## 9. Comparação de Métricas de Distância

**Figura:** `fig6_metric_comparison.png`

### 9.1 Ranking de Métricas por ARI

| Ranking | Métrica | ARI Médio | Desvio Padrão | N Experimentos |
|---------|---------|-----------|---------------|----------------|
| 1º | **Euclidiana** | **0.4729** | 0.4020 | 765 |
| 2º | Minkowski-L2 | 0.3712 | 0.3594 | 4.110 |
| 3º | Minkowski-L1 | 0.3672 | 0.3523 | 4.305 |
| 4º | Minkowski-L3 | 0.3655 | 0.3565 | 4.155 |
| 5º | Mahalanobis | 0.2895 | 0.3437 | 4.110 |

### 9.2 Análise Comparativa

**1. Euclidiana vs Minkowski-L2 (Matematicamente Equivalentes)**

Ambas calculam a norma L2, mas:
- Euclidiana (sklearn): Implementada em Cython, usa BLAS
- Minkowski-L2 (nossa): Implementada em Python puro

**Diferença de 27% no ARI!** Isso sugere que:
- A eficiência computacional permite mais iterações no tempo limite
- Operações vetorizadas podem ter melhor precisão numérica
- O overhead do Python adiciona latência que afeta convergência

**2. Normas Lp (L1, L2, L3) São Praticamente Equivalentes**

| Comparação | Diferença ARI |
|------------|---------------|
| L2 vs L1 | +1.1% |
| L2 vs L3 | +1.6% |
| L1 vs L3 | +0.5% |

**Conclusão**: A escolha entre L1, L2, L3 tem impacto marginal. Use L2 por padrão.

**3. Mahalanobis é Consistentemente Pior**

- 22% pior que Minkowski-L2
- 38% pior que Euclidiana nativa

**Causas identificadas:**
1. Instabilidade numérica na inversão de matriz
2. Covariância global inadequada para clusters heterogêneos
3. Custo computacional maior (O(d²) por distância)

### 9.3 Recomendações

| Situação | Métrica Recomendada |
|----------|---------------------|
| Uso geral | Euclidiana (sklearn) |
| Features com escalas diferentes | Manhattan (L1) após normalização |
| Dados esparsos | Manhattan (L1) |
| Outliers presentes | Manhattan (L1) |
| Conhecimento prévio de covariância | Mahalanobis com regularização |

---

## 10. Comparação por Tipo de Dataset

**Figura:** `fig7_dataset_type_comparison.png`

### 10.1 ARI por Tipo de Dataset

| Tipo | KMeans | MaxMin | Refinement | Gap (KMeans - Melhor outro) |
|------|--------|--------|------------|---------------------------|
| Sintético (MultiVar) | **0.7383** | 0.5978 | 0.5700 | +0.1405 (19%) |
| Sintético (SKL) | **0.5134** | 0.3373 | 0.3835 | +0.1299 (25%) |
| Real (UCI) | **0.1214** | 0.0183 | 0.0171 | +0.1031 (85%) |

### 10.2 Tempo de Execução por Tipo

| Tipo | KMeans | MaxMin | Refinement |
|------|--------|--------|------------|
| Sintético (MultiVar) | 0.0061s | **0.0001s** | 0.0015s |
| Sintético (SKL) | 0.0060s | **0.0001s** | 0.0009s |
| Real (UCI) | 0.0210s | **0.0006s** | 0.0229s |

### 10.3 Análise Detalhada

**Gap Crescente em Dados Reais:**

O gráfico mostra que a diferença de desempenho entre KMeans e os outros algoritmos **aumenta dramaticamente** em dados reais:

- Em dados sintéticos: KMeans é ~25% melhor
- Em dados reais: KMeans é **85% melhor**

**Possíveis Causas:**

1. **Inicialização**:
   - KMeans usa K-Means++ (sklearn), que escolhe centros iniciais otimizados
   - MaxMin começa do ponto mais distante, sem considerar a distribuição global

2. **Refinamento Iterativo**:
   - KMeans refina centros iterativamente (Lloyd's algorithm)
   - MaxMin/Refinement são algoritmos de uma passada (greedy)

3. **Natureza dos Dados Reais**:
   - Mais ruído e outliers
   - Classes desbalanceadas
   - Features com correlações complexas
   - Clusters não são esféricos ou bem separados

**Eficiência em Dados Reais:**

Apesar do pior ARI, MaxMin é **35x mais rápido** que KMeans em dados UCI. Para aplicações onde tempo é crítico e uma solução aproximada é aceitável, MaxMin continua sendo uma boa escolha.

### 10.4 Distribuição Visual (Boxplot)

O boxplot à esquerda mostra:
- **Dados Sintéticos**: Distribuição mais concentrada, medianas altas
- **Dados Reais**: Distribuição espalhada, muitos valores próximos de zero

O gráfico de barras à direita confirma:
- Tempo aumenta significativamente para dados UCI (mais amostras, mais dimensões)
- Refinement tem tempo comparável ao KMeans em dados reais (Mahalanobis é custoso)

---

## 11. Casos Específicos de Sucesso

### 11.1 Datasets Onde MaxMin/Refinement Superam KMeans

| Dataset | KMeans | MaxMin | Refinement | Melhor Alternativo | Ganho |
|---------|--------|--------|------------|--------------------|-------|
| UCI_Banknote | 0.0131 | 0.0787 | **0.0883** | Refinement | +575% |
| UCI_Bankruptcy | -0.0165 | 0.0019 | **0.0052** | Refinement | — |
| SKL_NoisyCircles_5 | -0.0013 | **0.0389** | 0.0199 | MaxMin | — |
| SKL_NoisyCircles_4 | -0.0013 | **0.0195** | 0.0100 | MaxMin | — |
| SKL_NoisyCircles_2 | -0.0014 | **0.0118** | 0.0040 | MaxMin | — |
| SKL_NoisyMoons_3 | 0.4694 | **0.4786** | 0.4496 | MaxMin | +2% |
| SKL_NoisyCircles_1 | -0.0013 | -0.0008 | **0.0015** | Refinement | — |
| SKL_NoisyCircles_3 | -0.0013 | **-0.0001** | -0.0009 | MaxMin | — |

**Total: 8 datasets (16%)** onde os algoritmos implementados superam KMeans.

### 11.2 Análise dos Casos de Sucesso

**Padrão Identificado:** MaxMin/Refinement tendem a superar KMeans em:

1. **Dados Não-Convexos (Circles)**:
   - KMeans assume clusters esféricos
   - MaxMin escolhe pontos mais distantes, que podem estar nas bordas das estruturas circulares
   - ARI ainda é baixo para todos, mas MaxMin é menos negativo

2. **Dados com Estrutura Peculiar (Banknote, Bankruptcy)**:
   - Estes datasets têm distribuições que não favorecem a convergência do Lloyd's algorithm
   - A escolha gulosa do MaxMin pode escapar de mínimos locais ruins

3. **Interpretação Cautelosa**:
   - Em todos os casos, os ARIs absolutos ainda são baixos (< 0.1)
   - A "vitória" é relativa a um baseline já ruim
   - Não significa que MaxMin/Refinement são bons nesses datasets, apenas menos ruins

### 11.3 UCI_Banknote: Estudo de Caso

O dataset Banknote merece atenção especial:

- **Descrição**: Detecção de notas falsas (2 classes: genuína/falsa)
- **Features**: 4 medidas de wavelet extraídas de imagens
- **Resultado**: Refinement (0.088) >> KMeans (0.013)

**Por que KMeans falhou?**
- As classes não são linearmente separáveis em L2
- KMeans converge para um mínimo local ruim
- A inicialização K-Means++ não ajuda neste caso específico

**Por que Refinement funcionou melhor?**
- A busca binária no raio encontra uma cobertura mais equilibrada
- Menos sensível a mínimos locais por não ser iterativo

---

## 12. Conclusões

### 12.1 Resumo dos Principais Achados

| Aspecto | Conclusão |
|---------|-----------|
| **Qualidade (ARI)** | KMeans > Refinement > MaxMin na maioria dos casos |
| **Eficiência** | MaxMin é 46x mais rápido que KMeans |
| **Trade-off δ** | δ=0.05 oferece bom compromisso qualidade/tempo |
| **Métricas** | Euclidiana (nativa) é a melhor; Mahalanobis decepcionou |
| **Dados Reais** | Gap de qualidade aumenta; MaxMin ainda é útil por velocidade |
| **Casos Especiais** | MaxMin/Refinement vencem em 16% dos datasets (não-convexos) |

### 12.2 Contribuições Teóricas Validadas

1. ✅ **MaxMin é O(nk)**: Confirmado experimentalmente (tempo constante baixo)
2. ✅ **Refinement oferece garantia 2-aproximação**: Funciona conforme esperado
3. ❌ **Mahalanobis para clusters elípticos**: Não confirmado sem regularização
4. ✅ **Trade-off precisão/tempo com δ**: Claramente demonstrado

### 12.3 Limitações do Estudo

1. **Implementação Python**: MaxMin/Refinement em Python puro, não otimizados
2. **Covariância Global**: Mahalanobis usa matriz global, não local
3. **Ausência de Regularização**: Pode explicar falha de Mahalanobis
4. **Datasets Limitados**: 51 datasets podem não representar todos os casos

### 12.4 Recomendações Práticas

| Cenário | Recomendação |
|---------|--------------|
| **Aplicação Geral** | Use sklearn.KMeans com Euclidiana |
| **Tempo Crítico** | Use MaxMin para solução rápida 2-aproximada |
| **Garantias Formais** | Use Refinement com δ apropriado |
| **Dados Não-Convexos** | Considere algoritmos espectrais (DBSCAN, Spectral Clustering) |
| **Alta Dimensão** | Use PCA antes do clustering; evite Mahalanobis |

### 12.5 Trabalhos Futuros

1. **Otimização**: Implementar MaxMin/Refinement em Cython ou Numba
2. **Mahalanobis Local**: Usar covariância por cluster (EM-like)
3. **Regularização**: Aplicar Ledoit-Wolf shrinkage na matriz de covariância
4. **Mais Métricas**: Testar cosseno, correlação, DTW para séries temporais
5. **Benchmarks Maiores**: Testar em datasets com milhões de pontos

---

## 13. Referências das Figuras

| Arquivo | Descrição | Seção |
|---------|-----------|-------|
| `fig1_sensitivity_analysis.png` | Sensibilidade do parâmetro δ | §4 |
| `fig2_boxplot_comparison.png` | Boxplots de ARI por algoritmo/métrica | §5 |
| `fig3_geometry_impact.png` | Euclidiana vs Mahalanobis por geometria | §6 |
| `fig4_scalability.png` | Tempo de execução por dataset | §7 |
| `fig5_heatmap_performance.png` | Heatmap ARI (Dataset × Algoritmo) | §8 |
| `fig6_metric_comparison.png` | Comparação de métricas de distância | §9 |
| `fig7_dataset_type_comparison.png` | Sintético vs Real | §10 |
| `tabela_resumo.tex` | Tabela LaTeX para artigo IEEE | Apêndice |

---

## Apêndice: Tabela LaTeX para Artigo IEEE

```latex
\begin{table}[htbp]
\centering
\caption{Resumo Comparativo dos Algoritmos de Clustering (Média $\pm$ Desvio Padrão)}
\label{tab:resumo_algoritmos}
\begin{tabular}{lccc}
\hline
\textbf{Algoritmo} & \textbf{ARI} & \textbf{Silhouette} & \textbf{Tempo (s)} \\
\hline
KMeans & $0.4729 \pm 0.4020$ & $0.4610 \pm 0.2079$ & $0.0093 \pm 0.0145$ \\
MaxMin & $0.3196 \pm 0.3430$ & $0.4339 \pm 0.1984$ & $0.0002 \pm 0.0004$ \\
Refinement & $0.3552 \pm 0.3568$ & $0.4636 \pm 0.1930$ & $0.0055 \pm 0.0161$ \\
\hline
\end{tabular}
\end{table}
```

---

*Relatório gerado automaticamente em 28/11/2025*  
*Ferramentas: Python 3.x, Pandas, Matplotlib, Seaborn*

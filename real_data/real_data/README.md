# 📊 Datasets para Trabalho Prático de Clustering

Este diretório contém datasets do **UCI Machine Learning Repository** preparados para uso em algoritmos de agrupamento (clustering).

---

## 📋 Resumo dos Datasets

| # | Dataset | Arquivo | Instâncias | Features | K (Classes) | Status |
|---|---------|---------|------------|----------|-------------|--------|
| 1 | Banknote Authentication | `banknote.csv` | 1,372 | 4 | 2 | ✅ Aprovado |
| 2 | Optical Digits | `optdigits.csv` | 5,620 | 64 | 10 | ✅ Aprovado |
| 3 | Wine Quality (Red) | `winequality_red.csv` | 1,599 | 11 | 6 | ✅ Aprovado |
| 4 | Wine Quality (White) | `winequality_white.csv` | 4,898 | 11 | 7 | ✅ Aprovado |
| 5 | Wine Quality (Combined) | `winequality_combined.csv` | 6,497 | 11 | 7 | ✅ Aprovado |
| 6 | Taiwanese Bankruptcy | `bankruptcy.csv` | 6,819 | 95 | 2 | ✅ Aprovado |
| 7 | SECOM | `secom.csv` | 1,567 | 590 | 2 | ✅ Aprovado |
| 8 | Drug Consumption | `drug_consumption.csv` | 1,885 | 12 | 7 | ✅ Aprovado |
| 9 | Myocardial Infarction | `mi.csv` | 1,700 | 123 | 8 | ✅ Aprovado |
| 10 | Obesity Levels | `obesity.csv` | 2,111 | 16 | 7 | ✅ Aprovado |
| 11 | Cardiotocography | `cardiotocography.csv` | 2,126 | 36 | 3 | ✅ Aprovado |
| 12 | BEED (EEG Epilepsy) | `BEED_Data.csv` | 8,000 | 16 | 4 | ✅ Aprovado |

---

## 📁 Descrição Detalhada

### 1. Banknote Authentication
**Pasta:** `banknote+authentication/`  
**Arquivo:** `banknote.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 1,372 |
| Features | 4 (variance, skewness, curtosis, entropy) |
| Coluna Alvo | `class` |
| Classes | 2 (autêntica=0, falsificada=1) |
| Tipo de Dados | 100% Numérico |

**Descrição:** Dataset para classificação de notas bancárias como autênticas ou falsificadas, baseado em características extraídas de imagens.

---

### 2. Optical Recognition of Handwritten Digits
**Pasta:** `optical+recognition+of+handwritten+digits/`  
**Arquivo:** `optdigits.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 5,620 |
| Features | 64 (pixels de imagem 8x8) |
| Coluna Alvo | `digit` |
| Classes | 10 (dígitos 0-9) |
| Tipo de Dados | 100% Numérico (inteiros 0-16) |

**Descrição:** Imagens de dígitos manuscritos normalizados para 8x8 pixels. Cada pixel é representado por um valor de 0 a 16.

---

### 3. Wine Quality (Red)
**Pasta:** `wine+quality/`  
**Arquivo:** `winequality_red.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 1,599 |
| Features | 11 (propriedades físico-químicas) |
| Coluna Alvo | `quality` |
| Classes | 6 (scores de 3 a 8) |
| Tipo de Dados | 100% Numérico |

**Features:** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol

---

### 4. Wine Quality (White)
**Pasta:** `wine+quality/`  
**Arquivo:** `winequality_white.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 4,898 |
| Features | 11 (propriedades físico-químicas) |
| Coluna Alvo | `quality` |
| Classes | 7 (scores de 3 a 9) |
| Tipo de Dados | 100% Numérico |

---

### 5. Wine Quality (Combined)
**Pasta:** `wine+quality/`  
**Arquivo:** `winequality_combined.csv`

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 6,497 |
| Features | 11 |
| Coluna Alvo | `quality` |
| Classes | 7 |
| Tipo de Dados | 100% Numérico |

**Descrição:** Combinação dos vinhos tintos e brancos em um único dataset.

---

### 6. Taiwanese Bankruptcy Prediction
**Pasta:** `taiwanese+bankruptcy+prediction/`  
**Arquivo:** `bankruptcy.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Taiwanese+Bankruptcy+Prediction)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 6,819 |
| Features | 95 (indicadores financeiros) |
| Coluna Alvo | `bankrupt` |
| Classes | 2 (falência=1, não falência=0) |
| Tipo de Dados | 100% Numérico |

**Descrição:** Indicadores financeiros de empresas taiwanesas para predição de falência.

---

### 7. SECOM
**Pasta:** `secom/`  
**Arquivo:** `secom.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/SECOM)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 1,567 |
| Features | 590 (sensores de manufatura) |
| Coluna Alvo | `label` |
| Classes | 2 (defeito=-1, sem defeito=1) |
| Tipo de Dados | 100% Numérico |

**Tratamento aplicado:** Valores NaN substituídos pela média da coluna.

**Descrição:** Dados de sensores de uma linha de produção de semicondutores para detecção de defeitos.

---

### 8. Drug Consumption (Quantified)
**Pasta:** `drug+consumption+quantified/`  
**Arquivo:** `drug_consumption.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 1,885 |
| Features | 12 (traços de personalidade) |
| Coluna Alvo | `Cannabis_class` |
| Classes | 7 (CL0 a CL6 - níveis de consumo) |
| Tipo de Dados | 100% Numérico |

**Features:** Age, Gender, Education, Country, Ethnicity, Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, SS

**Descrição:** Scores de personalidade (NEO-FFI-R) usados para classificar padrões de consumo de substâncias.

---

### 9. Myocardial Infarction Complications
**Pasta:** `myocardial+infarction+complications/`  
**Arquivo:** `mi.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Myocardial+infarction+complications)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 1,700 |
| Features | 123 (dados clínicos) |
| Coluna Alvo | `target` |
| Classes | 8 (tipos de complicação) |
| Tipo de Dados | 100% Numérico |

**Tratamento aplicado:** Valores '?' substituídos pela média da coluna.

**Descrição:** Dados clínicos de pacientes com infarto do miocárdio para predição de complicações.

---

### 10. Obesity Levels
**Pasta:** `estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition/`  
**Arquivo:** `obesity.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 2,111 |
| Features | 16 (hábitos alimentares e condição física) |
| Coluna Alvo | `NObeyesdad` |
| Classes | 7 (níveis de obesidade) |
| Tipo de Dados | Numérico (após codificação) |

**Classes de obesidade:**
- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

**Tratamento aplicado:** Variáveis categóricas codificadas numericamente.

---

### 11. Cardiotocography
**Pasta:** `cardiotocography/`  
**Arquivo:** `cardiotocography.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Cardiotocography)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 2,126 |
| Features | 36 (características do CTG) |
| Coluna Alvo | `NSP` |
| Classes | 3 (Normal=1, Suspeito=2, Patológico=3) |
| Tipo de Dados | 100% Numérico |

**Descrição:** Exames de cardiotocografia fetal para classificação do estado de saúde do feto.

---

### 12. BEED - Bangalore EEG Epilepsy Dataset ✅
**Pasta:** `beed_+bangalore+eeg+epilepsy+dataset/`  
**Arquivo:** `BEED_Data.csv`  
**Fonte:** [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition)

| Propriedade | Valor |
|-------------|-------|
| Instâncias | 8,000 |
| Features | 16 (sinais EEG) |
| Coluna Alvo | `y` |
| Classes | 4 (0, 1, 2, 3 - tipos de atividade cerebral) |
| Tipo de Dados | 100% Numérico (inteiros) |
| Valores Nulos | 0 |

**Features:** X1 a X16 representam valores de sinais de eletroencefalograma (EEG) amostrados.

**Descrição:** Dataset de sinais EEG para classificação de atividade epiléptica. Cada registro contém 16 valores de amplitude do sinal EEG, classificados em 4 categorias de atividade cerebral.

---

## 🔧 Tratamentos de Dados Aplicados

| Dataset | Tratamento |
|---------|-----------|
| SECOM | Valores `NaN` → média da coluna |
| Myocardial Infarction | Valores `?` → `NaN` → média da coluna |
| Obesity Levels | Variáveis categóricas → codificação numérica |
| Wine Quality | Delimitador `;` → `,` |
| Cardiotocography | Formato `.xls` → `.csv` |
| Optical Digits | Combinação de `.tra` e `.tes` |

---

## 📖 Como Usar

```python
import pandas as pd

# Carregar um dataset
df = pd.read_csv('banknote+authentication/banknote.csv')

# Separar features e target
X = df.drop('class', axis=1)  # Features
y = df['class']               # Target (para validação)

# Aplicar seu algoritmo de clustering
# ...
```

---

## ✅ Requisitos Atendidos

Todos os datasets aprovados atendem aos seguintes requisitos:

1. ✅ **Origem:** UCI Machine Learning Repository
2. ✅ **Tamanho:** Mínimo de 700 instâncias
3. ✅ **Tipo de dados:** Atributos exclusivamente numéricos
4. ✅ **Definição de K:** Dataset original de classificação com classes definidas
5. ✅ **Separação:** Coluna de classe identificável e separável

---

## 📚 Referências

- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/index.php
- Documentação dos datasets originais disponível nos arquivos `.names` em cada pasta

---

*Gerado automaticamente pelo script `convert_datasets.py`*

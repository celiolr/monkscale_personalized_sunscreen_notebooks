# Sistema de Recomendação Personalizada de Protetor Solar com Base em Visão Computacional e na Escala Monk

Este projeto utiliza Visão Computacional e Redes Neurais Convolucionais para estimar o tom de pele na Escala Monk (MST) de forma contínua e, a partir disso, gerar recomendações personalizadas de protetor solar considerando características cromáticas individuais.
O pipeline inclui pré-processamento completo, detecção facial, calibração LAB, treinamento, avaliação, e geração das recomendações finais.

![Python](https://img.shields.io/badge/🐍_Python-3.8+-blue)
![Colab](https://img.shields.io/badge/☁️_Google_Colab-✅-red)
![GPU](https://img.shields.io/badge/🎮_GPU-NVIDIA_T4-green)
![RAM](https://img.shields.io/badge/💾_RAM-8GB+-yellow)
![Storage](https://img.shields.io/badge/💽_Storage-10GB+-orange)
![Dataset](https://img.shields.io/badge/📊_Dataset-Disponível-green)<br>
![EfficientNet-B0](https://img.shields.io/badge/EfficientNet--B0-⚖️_Balance_Precision_Effiency-blue)
![ConvNeXt-Tiny](https://img.shields.io/badge/ConvNeXt--Tiny-🔄_Modern_Architecture-green)
![MobileNet-V3](https://img.shields.io/badge/MobileNet--V3--Large-📱_Mobile_Optimized-orange)
![VGG16](https://img.shields.io/badge/VGG16-🏛️_Classic_Baseline-lightgrey)<br>
![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)
![Non-Commercial](https://img.shields.io/badge/Non--Commercial-🚫-red.svg)
![Modifications](https://img.shields.io/badge/Modifications-✅-green.svg)
![Share Alike](https://img.shields.io/badge/Share_Alike-🔄-blue.svg)

## 📚 Publicação

**Artigo Técnico Científico em Revisão:** [![Status](https://img.shields.io/badge/Artigo-🚧_Em_Breve-orange)](https://github.com/celiolr/monkscale_personalized_sunscreen_notebooks/issues/1)

*Em revisão - acompanhe o progresso no link no emoji acima*

## 📋 Índice

- [🎯 Sobre o Projeto](#-sobre-o-projeto)
- [✨ Características](#-características)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🛠️ Pré-requisitos](#-pré-requisitos)
- [🚀 COMO EXECUTAR NO GOOGLE COLAB](#-como-executar-no-google-colab)
- [🔧 Processamento](#-processamento)
- [🧠 Modelos Implementados](#-modelos-implementados)
- [🏋️ Treinamento](#-treinamento)
- [📈 Avaliação](#-avaliação)
- [📝 Licença](#-licença)
- [🎯 Resultados Esperados](#-resultados-esperados)
- [💡 Dicas para Execução Bem-sucedida](#-dicas-para-execução-bem-sucedida)

## 🎯 Sobre o Projeto

**OBJETIVO PRINCIPAL:** Desenvolver um sistema de visão computacional para estimar tons de pele usando a escala Monk (MST) e gerar recomendações personalizadas de protetor solar.

### 🎯 O Que Este Projeto Faz:
- 🔍 **Estima** tons de pele de forma contínua usando a escala MST
- 🧠 **Utiliza** modelos de Deep Learning (CNNs) para análise facial
- 🧴 **Gera** formulações personalizadas de protetor solar
- 📊 **Fornece** métricas detalhadas de precisão

### 🚀 Casos de Uso:
- **Dermatologia**: Análise automatizada de tons de pele
- **Cosméticos Personalizados**: Recomendações específicas de protetor solar por tom de pele
- **Pesquisa Acadêmica**: Estudos sobre diversidade de tons de pele

[voltar ao topo](#-índice)

## ✨ Características

- **Escala MST Contínua:** Estimativa granular e precisa do tom de pele [![Official MST](https://img.shields.io/badge/Google_Official_MST-🎨-4285F4)](https://skintone.google)
- **Múltiplas Arquiteturas de CNN:**
  - EfficientNet-B0
  - ConvNeXt-Tiny
  - MobileNetV3-Large
  - VGG16
- **Pré-processamento Avançado:** 
  - Detecção facial
  - Normalização LAB
  - Data augmentation
- **Pipeline Completo de Treinamento dos Modelos:** 
  - Treino
  - Validação
  - Teste
  - Separação por identidade
- **Análise Detalhada:** 
  - Métricas de regressão 
    - MAE
    - MSE
    - R²
- **Gerar Formulações Personalizadas de protetor solar** com base nas características cromáticas individuais e o MST estimado

[voltar ao topo](#-índice)

## 📁 Estrutura do Projeto

```markdown
skin-tone-estimation-mst/
├── data/                                                   # Estrutura de dados (no Colab)
│   ├── images_dataset/                                     # Dataset de imagens
│   │   ├── [person_id]/
│   │   │   ├── front-facing/
│   │   │   │   ├── *.jpeg
│   │   │   │   └── ...
│   │   │   ├── monk_scale_value.json
│   │   │   └── ...
│   │   └── calibrate_refer_data.json
│   └── result/                                             # Resultados e modelos treinados
│       └── model/                                          # Modelos treinados com 20% dos dados para validação
│           ├── MODELS.md                                   # Documentação dos modelos treinados
│           ├── MST_model_convnext_tiny_Regression_best_val_DS-20_face-front.pth
│           ├── MST_model_efficientnet_b0_Regression_best_val_DS-20_face-front.pth
│           └── MST_model_mobilenet_v3_large_Regression_best_val_DS-20_face-front.pth
├── notebooks/                                              # Pasta com todos os notebooks
│   ├── pipeline_best_MST_final.ipynb                       # Pipeline final otimizado de treinamento dos modelos para faces frontais
│   └── recommendations_MST_sunscreen_notebook.ipynb        # Pipeline para gerar recomendações personalizadas de protetor solar
├── DATASET_INSTRUCTIONS.md                                 # Instruções para preparar o dataset
├── LICENSE                                                 # Licença MIT
└── README.md                                               # Arquivo de documentação
```
### Arquivo calibrate_refer_data.json:
- Dados de referência para calibração LAB
- Obtido da pessoa 52 (aleatório) do dataset original
- Exemplo de conteúdo:
```json
{
    "mean_L": 87.61049771471089,
    "mean_a": 128.51117400085033,
    "mean_b": 132.42577593537416,
    "std_L": 35.13450777903315,
    "std_a": 5.2290901050019025,
    "std_b": 12.609734192300618
}
```

[voltar ao topo](#-índice)

## 🛠️ Pré-requisitos

- ![Python](https://img.shields.io/badge/🐍_Python-3.8+-blue)  Python 3.8+ 
- ![Colab](https://img.shields.io/badge/☁️_Google_Colab-✅-red) Google Colab (recomendado)
- ![GPU](https://img.shields.io/badge/🎮_GPU-NVIDIA_T4-green) GPU com suporte CUDA se disponível (NVIDIA T4 GPU) `(Sem GPU o tempo de treinamento será significativamente maior)`
- ![RAM](https://img.shields.io/badge/💾_RAM-8GB+-yellow) 8GB+ RAM
- ![Storage](https://img.shields.io/badge/💽_Storage-10GB+-orange) 10GB+ espaço em disco
- ![Dataset](https://img.shields.io/badge/📊_Dataset-Disponível-green) Dataset existente (ver [DATASET_INSTRUCTIONS.md](DATASET_INSTRUCTIONS.md) para detalhes)

[voltar ao topo](#-índice)

## 🚀 COMO EXECUTAR NO GOOGLE COLAB

### ⚡ Execução Rápida no Google Colab:

[![Open in Colab](https://img.shields.io/badge/🔗_Abrir_no_Colab-Treinamento_Modelo-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/celiolr/monkscale_personalized_sunscreen_notebooks/blob/main/notebooks/pipeline_best_MST_final.ipynb)
[![Open in Colab](https://img.shields.io/badge/🔗_Abrir_no_Colab-Recomendações_Fotoprotetor-F9AB00?logo=googlecolab)](https://colab.research.google.com/github/celiolr/monkscale_personalized_sunscreen_notebooks/blob/main/notebooks/recommendations_MST_sunscreen_notebook.ipynb)

**Passo a Passo:**
1. **Clique** em um dos badges acima para abrir no Colab
2. **Conecte** a uma GPU: `Ambiente de execução` > `Alterar o tipo de ambiente de execução` > `GPUs: T4`
3. **Siga as instruções** dentro do notebook para detalhes específicos
4. **Execute** as células sequencialmente
5. **⚠️ Atente para avisos importantes** antes de algumas células (ex: ambiente de execução, reinicializações, descompactação do dataset)
6. **Aguarde** o processamento (pode demorar alguns minutos)

[voltar ao topo](#-índice)

### 📁 Preparar os Dados no Google Drive - Dataset
Esse parágrafo é sobre o notebook: `notebooks/Pipeline_best_MST_final.ipynb`

#### 📊 Dataset

##### Estrutura do Dataset:
- **285 pastas** (pessoas) × **15 imagens** por pose - Total: mais 21k de imagens
- **Poses:** front-facing, left-facing, right-facing, up-facing, down-facing
  - **front-facing:** 285 pessoas × 15 imagens = 4275 imagens foram usados no treinamento
- **Formato:** JPEG + JSON com labels MST

##### Labels MST:
- Arquivo `monk_scale_value.json` em cada pasta de pessoa
- Valores contínuos de 1.0 a 10.0
- Exemplo de conteúdo:
```json
{"value": 5.5}
```

[voltar ao topo](#-índice)

### 📊 Monitoramento

Durante o treinamento, monitore:
- **Loss de treinamento e validação**
- **Métricas `MAE/MSE/R²`**
- **Uso de GPU** 

[voltar ao topo](#-índice)

## 🔧 Processamento

### Pipeline de Processamento:

#### Notebook: pipeline_best_MST_final.ipynb 
1. **Filtragem por Pose:** Apenas faces frontais (configurável)
2. **Detecção Facial:** MTCNN + Haar Cascade (fallback)
3. **Recorte Facial:** Com margem de 15%
4. **Normalização de Cor:** Conversão para espaço LAB
5. **Calibração:** Baseada em dados de referência
6. **Data Augmentation:** Flip horizontal/vertical
7. **Redimensionamento:** 224×224 pixels

#### Notebook: recommendations_MST_sunscreen_notebook.ipynb
1. **Carregamento do Modelo Treinado**
2. **Pré-processamento da Imagem de Entrada**
3. **Predição do MST**
4. **Geração de Recomendação Personalizada de Protetor Solar**
5. **Exibição dos Resultados**

[voltar ao topo](#-índice)

## 🧠 Modelos Implementados

### Arquiteturas:
Disponibilizados modelos com tag DS-20 com 20% dos dados para validação [![Model Files](https://img.shields.io/badge/🧠_Model_Files-3_Models_Available-success)](https://github.com/celiolr/monkscale_personalized_sunscreen_notebooks/tree/main/data/result/model)

| Modelo                                                                      | Status                                     | Melhor Uso |
|-----------------------------------------------------------------------------|--------------------------------------------|------------|
| ![ConvNeXt-Tiny](https://img.shields.io/badge/ConvNeXt--Tiny-✅🔄_Modern_Architecture-blue)        | **Moderno/Recomendado/Melhor no Contexto** | Arquitetura baseada em transformers |
| ![EfficientNet-B0](https://img.shields.io/badge/EfficientNet--B0-✅⚖️_Balance_Precision_Effiency-green)   | **Moderno/Eficiente**                               | Balance ideal entre precisão e velocidade |
| ![MobileNet-V3](https://img.shields.io/badge/MobileNet--V3--Large-✅📱_Mobile_Optimized-orange) | **Eficiente**                              | Otimizado para dispositivos móveis |
| ![VGG16](https://img.shields.io/badge/VGG16-🚫🏛️_Classic_Baseline-lightgrey)                   | Não incluído                               | Baseline para comparação |

[📋 Ver documentação dos modelos](https://github.com/celiolr/monkscale_personalized_sunscreen_notebooks/blob/main/data/result/model/MODELS.md)

[voltar ao topo](#-índice)

## 🏋️ Treinamento

### Hiperparâmetros Principais:

- ![Image Size](https://img.shields.io/badge/🖼️_Image_Size-224×224-blue) **Tamanho da imagem**: 224×224
- ![Batch Size](https://img.shields.io/badge/📦_Batch_Size-32-green) **Batch size**: 32
- ![Epochs](https://img.shields.io/badge/🔄_Epochs-30-orange) **Épocas**: 30
- ![Learning Rate](https://img.shields.io/badge/📈_Learning_Rate-1e--4-red) **Learning rate**: 1e-4
- ![Split](https://img.shields.io/badge/📊_Split-65/20/15-purple) **Divisão**: 65% treino, 20% validação, 15% teste

[voltar ao topo](#-índice)

## 📈 Avaliação

### Métricas e Resultados Principais:

- ![MAE](https://img.shields.io/badge/📏_MAE-Mean_Absolute_Error-blue) **MAE (Mean Absolute Error)**: Erro absoluto médio
- ![MSE](https://img.shields.io/badge/📐_MSE-Mean_Squared_Error-green) **MSE (Mean Squared Error)**: Erro quadrático médio  
- ![R²](https://img.shields.io/badge/📊_R²-R_Squared-red) **R² (Coeficiente de Determinação)**: Variabilidade explicada
- ![Sunscreen](https://img.shields.io/badge/🧴_Sunscreen-Personalized-purple) **Formulações Personalizadas de Protetor Solar**: Baseadas no MST estimado e proporções de pigmentos

[voltar ao topo](#-índice)

## 📝 Licença

Este projeto está licenciado sob a **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

**Em resumo, você pode:**
* ✅ Compartilhar e Adaptar (Remixar/Transformar).

**Você deve:**
* ⚠️ Dar Crédito Apropriado (Atribuição).
* 🚫 Usar Apenas para Fins Não-Comerciais (NonCommercial).
* 🔄 Distribuir as Adaptações sob a Mesma Licença (ShareAlike).

* [**Ver o texto completo da Licença (Inglês)**](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
* [**Ver o Resumo da Licença (Português)**](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.pt)

**Resumo:** Você pode copiar e modificar este material para **uso não-comercial**, desde que dê os créditos e compartilhe as modificações sob a mesma licença.

![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)
![Non-Commercial](https://img.shields.io/badge/Non--Commercial-🚫-red.svg)
![Modifications](https://img.shields.io/badge/Modifications-✅-green.svg)
![Share Alike](https://img.shields.io/badge/Share_Alike-🔄-blue.svg)

[Ver licença](LICENSE)

[voltar ao topo](#-índice)

## 🎯 Resultados Esperados

### ✅ Ao Executar o Projeto, Você Obterá:
- **Modelos Treinados**: CNNs capazes de estimar tons de pele MST
- **Métricas de Performance**: MAE, MSE, R² detalhados
- **Recomendações Personalizadas**: Formulações de protetor solar por tom de pele
- **Visualizações**: Gráficos de predição e análise de erros

### 📈 Aplicações Práticas:
- 🏥 **Clínicas**: Triagem automatizada de tons de pele
- 💄 **Cosméticos**: Produtos personalizados por tom
- 🎓 **Educação**: Ferramenta de ensino sobre diversidade de pele

[voltar ao topo](#-índice)

## 💡 Dicas para Execução Bem-sucedida

1. **Verifique a GPU:** Certifique-se de que está usando GPU no Colab
2. **Siga a sequência:** Execute as células na ordem numérica
3. **Aguarde o processamento:** Algumas células (como descompactação) podem demorar
4. **Monitore recursos:** Verifique o uso de RAM e disco durante execução
5. **Salve resultados:** Faça download dos modelos treinados e métricas

**📞 Dúvidas?** Consulte a documentação dentro de cada célula do notebook para detalhes específicos de implementação. Cada célula contém documentação completa sobre objetivos, ações executadas e justificativas técnicas.

[voltar ao topo](#-índice)

# Sistema de Recomendação Personalizada de Protetor Solar com Base em Visão Computacional e na Escala Monk

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

- [🎯 Visão Geral](#-visão-geral)
- [✨ Características](#-características)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🛠️ Pré-requisitos](#-pré-requisitos)
- [🚀 COMO EXECUTAR NO GOOGLE COLAB](#-como-executar-no-google-colab)
- [🔧 Processamento](#-processamento)
- [🧠 Modelos Implementados](#-modelos-implementados)
- [🏋️ Treinamento](#-treinamento)
- [📈 Avaliação](#-avaliação)
- [📝 Licença](#-licença)
- [💡 Dicas para Execução Bem-sucedida](#-dicas-para-execução-bem-sucedida)

## 🎯 Visão Geral

**OBJETIVO GERAL:** Treinar e avaliar modelos de Deep Learning (CNNs) para estimar tons de pele de forma contínua utilizando a escala Monk Skin Tone (MST), inicialmente focando em faces frontais e, posteriormente, generalizando para outras poses e gerar formulações personalizadas de protetor solar com base nas características cromáticas individuais e o MST estimado.

[voltar ao topo](#-índice)

## ✨ Características

- **Escala MST Contínua:** Estimativa granular e precisa do tom de pele [![Official MST](https://img.shields.io/badge/Google_Official_MST-🎨-4285F4)](https://skintone.google)
- **Múltiplas Arquiteturas de CNN:** EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Large, VGG16
- **Pré-processamento Avançado:** Detecção facial, normalização LAB, data augmentation
- **Pipeline Completo de Treinamento dos Modelos:** Treino, validação e teste com separação por identidade
- **Análise Detalhada:** Métricas de regressão (MAE, MSE, R²) e visualizações
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

### 📥 Passo 1: Carregar o Notebook no Colab

**Opção A - Diretamente do GitHub:**
1. Acesse [Google Colab](https://colab.research.google.com/)
2. Clique em `File` > `Upload notebook`
3. Na aba `GitHub`, cole a URL do repositório
4. Selecione o notebook `notebooks/Pipeline_best_MST_final.ipynb` ou `notebooks/recommendations_MST_sunscreen_notebook.ipynb`

**Opção B - Upload Manual:**
1. Faça download do notebook do GitHub
2. Acesse [Google Colab](https://colab.research.google.com/)
3. Clique em \`File\` > `Upload notebook`
4. Faça upload do arquivo `.ipynb` baixado

[voltar ao topo](#-índice)

### 📁 Passo 2: Preparar os Dados no Google Drive - Dataset
Esse parágrafo é sobre o notebook: `notebooks/Pipeline_best_MST_final.ipynb`

1. **Crie a estrutura de pastas no seu Google Drive:**
```markdown
MyDrive/
└── IA_CD_UFES/
    └── TCC/
        ├── Dataset/
        │   └── images_dataset.zip
        └── images/
            └── calibrate_refer_data.json
```

2. **Faça upload dos arquivos:**
   - `images_dataset.zip` → na pasta `Dataset/`
   - `calibrate_refer_data.json` → na pasta `images/`

### 📊 Dataset

#### Estrutura do Dataset:
- **285 pastas** (pessoas) × **15 imagens** por pose - Total: mais 21k de imagens
- **Poses:** front-facing, left-facing, right-facing, up-facing, down-facing
  - **front-facing:** 285 pessoas × 15 imagens = 4275 imagens foram usados no treinamento
- **Formato:** JPEG + JSON com labels MST

#### Labels MST:
- Arquivo `monk_scale_value.json` em cada pasta de pessoa
- Valores contínuos de 1.0 a 10.0
- Exemplo de conteúdo:
```json
{"value": 5.5}
```

[voltar ao topo](#-índice)

### ⚙️ Passo 3: Executar o Notebook

**📌 IMPORTANTE:** Execute as células **SEQUENCIALMENTE** conforme a numeração. Cada célula está documentada com:

- **OBJETIVO:** O que a célula faz
- **AÇÕES EXECUTADAS:** Passos realizados
- **JUSTIFICATIVA TÉCNICA:** Por que foi implementado dessa forma
- **🎯 Execução por Seções uma após a outra:** A sequência é importante para evitar erros.
- **Uso de GPU** (acesse: `Ambiente de execução` > `Alterar o tipo de ambiente de execução` > `GPUs: T4`)

### ⚠️ AVISOS IMPORTANTES

**⚠️ REINICIALIZAÇÃO NECESSÁRIA:**
Nas células iniciais após instalar dependências, reinicie o ambiente de execução:
- Clique em `Ambiente de execução` > `Reiniciar ambiente de execução`

**⚠️ AJUSTE DE PATHS:**
Nas células indicadas, verifique e ajuste os paths se necessário:
Se sua estrutura de pastas for diferente, atualize os caminhos.

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
`Disponibilizados modelos DS-20 com 20% dos dados para validação` [![Model Files](https://img.shields.io/badge/🧠_Model_Files-3_Models_Available-success)](https://github.com/celiolr/monkscale_personalized_sunscreen_notebooks/tree/main/data/result/model)

- ![EfficientNet-B0](https://img.shields.io/badge/EfficientNet--B0-⚖️_Balance_Precision_Effiency-blue) ✅**EfficientNet-B0** - Balance entre precisão e eficiência
- ![ConvNeXt-Tiny](https://img.shields.io/badge/ConvNeXt--Tiny-🔄_Modern_Architecture-green) ✅**ConvNeXt-Tiny** - Arquitetura moderna baseada em transformers
- ![MobileNet-V3](https://img.shields.io/badge/MobileNet--V3--Large-📱_Mobile_Optimized-orange) ✅**MobileNet-V3-Large** - Otimizado para dispositivos móveis
- ![VGG16](https://img.shields.io/badge/VGG16-🏛️_Classic_Baseline-lightgrey) 🚫**VGG16** - Baseline clássica *(Não disponibilizado no notebook final devido ao desempenho inferior e tamanho do modelo)*

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

![MAE](https://img.shields.io/badge/📏_MAE-Mean_Absolute_Error-blue)
![MSE](https://img.shields.io/badge/📐_MSE-Mean_Squared_Error-green)
![R²](https://img.shields.io/badge/📊_R²-R_Squared-red)
![Sunscreen](https://img.shields.io/badge/🧴_Sunscreen-Personalized-purple)

- ![MAE](https://img.shields.io/badge/📏_MAE-Mean_Absolute_Error-blue) **MAE (Mean Absolute Error)**: Erro absoluto médio
- ![MSE](https://img.shields.io/badge/📐_MSE-Mean_Squared_Error-green) **MSE (Mean Squared Error)**: Erro quadrático médio  
- ![R²](https://img.shields.io/badge/📊_R²-R_Squared-red) **R² (Coeficiente de Determinação)**: Variabilidade explicada
- ![Sunscreen](https://img.shields.io/badge/🧴_Sunscreen-Personalized-purple) **Formulações Personalizadas de Protetor Solar**: Baseadas no MST estimado e proporções de pigmentos

[voltar ao topo](#-índice)

## 📝 Licença

Este projeto está licenciado sob a **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International**.

**Você pode:**
- ✅ **Compartilhar** — copiar e redistribuir o material
- ✅ **Adaptar** — remixar, transformar e criar a partir do material
- 🚫 **Não-Comercial** — não pode usar o material para fins comerciais
- 🔄 **CompartilharIgual** — se adaptar o material, deve distribuir sob a mesma licença

**Sob os termos:**
- **Atribuição** — Você deve dar o crédito apropriado
- **NãoComercial** — Você não pode usar o material para fins comerciais
- **CompartilharIgual** — Se você remixar ou transformar o material, deve distribuir suas contribuições sob a mesma licença

**Resumo:** Você pode copiar e modificar este material para **uso não-comercial**, desde que dê os créditos e compartilhe as modificações sob a mesma licença.
ual** — Se você remixar ou transformar o material, deve distribuir suas contribuições sob a mesma licença

![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)
![Non-Commercial](https://img.shields.io/badge/Non--Commercial-🚫-red.svg)
![Modifications](https://img.shields.io/badge/Modifications-✅-green.svg)
![Share Alike](https://img.shields.io/badge/Share_Alike-🔄-blue.svg)

[Ver licença](LICENSE) | [Resumo em português](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.pt_BR)

[voltar ao topo](#-índice)

## 💡 Dicas para Execução Bem-sucedida

1. **Verifique a GPU:** Certifique-se de que está usando GPU no Colab
2. **Siga a sequência:** Execute as células na ordem numérica
3. **Aguarde o processamento:** Algumas células (como descompactação) podem demorar
4. **Monitore recursos:** Verifique o uso de RAM e disco durante execução
5. **Salve resultados:** Faça download dos modelos treinados e métricas

**📞 Dúvidas?** Consulte a documentação dentro de cada célula do notebook para detalhes específicos de implementação. Cada célula contém documentação completa sobre objetivos, ações executadas e justificativas técnicas.

[voltar ao topo](#-índice)

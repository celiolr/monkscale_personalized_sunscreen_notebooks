# Sistema de Recomendação Personalizada de Protetor Solar com Base em Visão Computacional e na Escala Monk
** Link para o Artigo que deu origem a esse projeto: [Artigo no ResearchGate](https://www) **

## 📋 Índice

- [🎯 Visão Geral](#-visão-geral)
- [✨ Características](#-características)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🛠️ Pré-requisitos](#-pré-requisitos)
- [🚀 COMO EXECUTAR NO GOOGLE COLAB](#-como-executar-no-google-colab)
- [📊 Dataset](#-dataset)
- [🔧 Pré-processamento](#-pré-processamento)
- [🤖 Modelos Implementados](#-modelos-implementados)
- [🏋️ Treinamento](#-treinamento)
- [📈 Avaliação](#-avaliação)
- [📝 Licença](#-licença)
- [💡 Dicas para Execução Bem-sucedida](#-dicas-para-execução-bem-sucedida)

## 🎯 Visão Geral

**OBJETIVO GERAL:** Treinar e avaliar modelos de Deep Learning (CNNs) para estimar tons de pele de forma contínua utilizando a escala Monk Skin Tone (MST), inicialmente focando em faces frontais e, posteriormente, generalizando para outras poses e gerar formulações personalizadas de protetor solar com base nas características cromáticas individuais e o MST estimado.

[voltar ao topo](#-índice)

## ✨ Características

- **Escala MST Contínua:** Estimativa granular e precisa do tom de pele
- **Múltiplas Arquiteturas de CNN:** EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Large, VGG16
- **Pré-processamento Avançado:** Detecção facial, normalização LAB, data augmentation
- **Pipeline Completo:** Treino, validação e teste com separação por identidade
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

- Google Colab (recomendado) com ambiente Python 3.8+ e NVIDIA T4 GPU se disponível `(Sem GPU o tempo de treinamento será muito maior)`
- GPU com suporte CUDA
- 8GB+ RAM
- 10GB+ espaço em disco

[voltar ao topo](#-índice)

## 🚀 COMO EXECUTAR NO GOOGLE COLAB

### 📥 Passo 1: Carregar o Notebook no Colab

**Opção A - Diretamente do GitHub:**
1. Acesse [Google Colab](https://colab.research.google.com/)
2. Clique em `File` > `Upload notebook`
3. Na aba `GitHub`, cole a URL do repositório
4. Selecione o notebook `notebooks/Pipeline_best_MST_final.ipynb`

**Opção B - Upload Manual:**
1. Faça download do notebook do GitHub
2. Acesse [Google Colab](https://colab.research.google.com/)
3. Clique em \`File\` > `Upload notebook`
4. Faça upload do arquivo `.ipynb` baixado

### 📁 Passo 2: Preparar os Dados no Google Drive

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

### ⚙️ Passo 3: Executar o Notebook

**📌 IMPORTANTE:** Execute as células **SEQUENCIALMENTE** conforme a numeração. Cada célula está documentada com:

- **OBJETIVO:** O que a célula faz
- **AÇÕES EXECUTADAS:** Passos realizados
- **JUSTIFICATIVA TÉCNICA:** Por que foi implementado dessa forma
- **🎯 Execução por Seções uma após a outra:** A sequência é importante para evitar erros.

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
- **Uso de GPU** (acesse: `Ambiente de execução` > `Alterar o tipo de ambiente de execução` > `GPUs: T4`)

[voltar ao topo](#-índice)

## 📊 Dataset

### Estrutura do Dataset:
- **285 pastas** (pessoas) × **15 imagens** por pose - Total: mais 21k de imagens
- **Poses:** front-facing, left-facing, right-facing, up-facing, down-facing
  - **front-facing:** 285 pessoas × 15 imagens = 4275 imagens foram usados no treinamento
- **Formato:** JPEG + JSON com labels MST

### Labels MST:
- Arquivo `monk_scale_value.json` em cada pasta de pessoa
- Valores contínuos de 1.0 a 10.0

[voltar ao topo](#-índice)

## 🔧 Pré-processamento

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

## 🤖 Modelos Implementados

### Arquiteturas:
1. **EfficientNet-B0** - Balance entre precisão e eficiência
2. **ConvNeXt-Tiny** - Arquitetura moderna baseada em transformers
3. **MobileNet-V3-Large** - Otimizado para dispositivos móveis
4. **VGG16** - Baseline clássica `(Não disponibilizado no notebook final devido ao desempenho inferior e tamanho do modelo)`

[voltar ao topo](#-índice)

## 🏋️ Treinamento

### Hiperparâmetros Principais:
- **Tamanho da imagem:** 224×224
- **Batch size:** 32
- **Épocas:** 30
- **Learning rate:** 1e-4
- **Divisão:** 65% treino, 20% validação, 15% teste

[voltar ao topo](#-índice)

## 📈 Avaliação

### Métricas e Resultados Principais:
- **MAE (Mean Absolute Error):** Erro absoluto médio
- **MSE (Mean Squared Error):** Erro quadrático médio
- **R² (Coeficiente de Determinação):** Variabilidade explicada
- **Formulações Personalizadas de Protetor Solar (cor final):** Baseadas no MST estimado

[voltar ao topo](#-índice)

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

A licença aplica-se exclusivamente ao código-fonte.

Nenhum dataset, imagem, foto de participante ou material sensível (não público)
está incluído, ou licenciado por este repositório.

[voltar ao topo](#-índice)

## 💡 Dicas para Execução Bem-sucedida

1. **Verifique a GPU:** Certifique-se de que está usando GPU no Colab
2. **Siga a sequência:** Execute as células na ordem numérica
3. **Aguarde o processamento:** Algumas células (como descompactação) podem demorar
4. **Monitore recursos:** Verifique o uso de RAM e disco durante execução
5. **Salve resultados:** Faça download dos modelos treinados e métricas

**📞 Dúvidas?** Consulte a documentação dentro de cada célula do notebook para detalhes específicos de implementação. Cada célula contém documentação completa sobre objetivos, ações executadas e justificativas técnicas.

[voltar ao topo](#-índice)

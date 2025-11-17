# Skin Tone Estimation — MST (Monk Skin Tone)

Este projeto implementa um pipeline completo de Deep Learning para estimar tons de pele de forma contínua utilizando a escala Monk Skin Tone (MST), inicialmente focando em faces frontais e posteriormente generalizando para outras poses.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Características](#características)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [🚀 COMO EXECUTAR NO GOOGLE COLAB](#-como-executar-no-google-colab)
- [Dataset](#dataset)
- [Pré-processamento](#pré-processamento)
- [Modelos Implementados](#modelos-implementados)
- [Treinamento](#treinamento)
- [Avaliação](#avaliação)
- [Licença](#licença)

## 🎯 Visão Geral

**OBJETIVO GERAL:** Treinar e avaliar modelos de Deep Learning (CNNs) para estimar tons de pele de forma contínua utilizando a escala Monk Skin Tone (MST), inicialmente focando em faces frontais e, posteriormente, generalizando para outras poses.

## ✨ Características

- **Escala MST Contínua:** Estimativa granular e precisa do tom de pele
- **Múltiplas Arquiteturas de CNN:** EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Large, VGG16
- **Pré-processamento Avançado:** Detecção facial, normalização LAB, data augmentation
- **Pipeline Completo:** Treino, validação e teste com separação por identidade
- **Análise Detalhada:** Métricas de regressão (MAE, MSE, R²) e visualizações

## 📁 Estrutura do Projeto

```markdown
skin-tone-estimation-mst/
├── notebooks/                         # Pasta com todos os notebooks
│   ├── Pipeline_best_MST_final.ipynb  # Notebook principal
│   └── (outros notebooks futuros)     # Outras versões/experimentos
├── README.md                          # Este arquivo
├── LICENSE                            # Licença MIT
└── data/                              # Estrutura de dados (no Colab)
    ├── images_dataset/                # Dataset de imagens
    │   ├── [person_id]/
    │   │   ├── front-facing/
    │   │   │   ├── *.jpeg
    │   │   │   └── ...
    │   │   ├── monk_scale_value.json
    │   │   └── ...
    │   └── calibrate_refer_data.json
    └── result/                        # Resultados e modelos treinados
        └── model/                     # Modelos treinados com 20% dos dados para validação
            ├── MST_r06b_model_convnext_tiny_Regression_best_val_DS-20_face-front.pth
            ├── MST_r06b_model_efficientnet_b0_Regression_best_val_DS-20_face-front.pth
            ├── MST_r06b_model_mobilenet_v3_large_Regression_best_val_DS-20_face-front.pth
            └── MST_r06b_model_vgg16_Regression_best_val_DS-20_face-front.pth
```

## 🛠️ Pré-requisitos

- Google Colab (recomendado) ou ambiente Python 3.8+ (colab executado em NVIDIA T4 GPU)
- GPU com suporte CUDA
- 8GB+ RAM
- 10GB+ espaço em disco

## 🚀 COMO EXECUTAR NO GOOGLE COLAB

### 📥 Passo 1: Carregar o Notebook no Colab

**Opção A - Diretamente do GitHub:**
1. Acesse [Google Colab](https://colab.research.google.com/)
2. Clique em \`File\` > \`Upload notebook\`
3. Na aba \`GitHub\`, cole a URL do repositório
4. Selecione o notebook \`notebooks/Pipeline_best_MST_final.ipynb\`

**Opção B - Upload Manual:**
1. Faça download do notebook do GitHub
2. Acesse [Google Colab](https://colab.research.google.com/)
3. Clique em \`File\` > \`Upload notebook\`
4. Faça upload do arquivo \`.ipynb\` baixado

### 📁 Passo 2: Preparar os Dados no Google Drive

1. **Crie a estrutura de pastas no seu Google Drive:**
```markdown
<pre>
MyDrive/
└── IA_CD_UFES/
    └── TCC/
        ├── Dataset/
        │   └── images_dataset.zip
        └── images/
            └── calibrate_refer_data.json
</pre>
``` 

2. **Faça upload dos arquivos:**
   - \`images_dataset.zip\` → na pasta \`Dataset/\`
   - \`calibrate_refer_data.json\` → na pasta \`images/\`

### ⚙️ Passo 3: Executar o Notebook

**📌 IMPORTANTE:** Execute as células **SEQUENCIALMENTE** conforme a numeração. Cada célula está documentada com:

- **OBJETIVO:** O que a célula faz
- **AÇÕES EXECUTADAS:** Passos realizados
- **JUSTIFICATIVA TÉCNICA:** Por que foi implementado dessa forma

**Sequência de Execução:**

1. **🔧 Instalação e Configuração** (Células 1.1-1.2)
   - Instala dependências
   - Configura ambiente GPU
   - Define hiperparâmetros

2. **📂 Montagem do Drive** (Células 2.1-2.2)
   - Monta Google Drive
   - Descompacta dataset
   - Carrega labels MST

3. **🛠️ Dataset e Pré-processamento** (Células 3.1-3.2)
   - Implementa detecção facial
   - Cria pipeline de transformações
   - Define classe Dataset personalizada

4. **🤖 Modelos** (Células 4.1-4.2)
   - Implementa arquiteturas CNN
   - Adapta para entrada LAB
   - Define transformações de data augmentation

5. **🏋️ Treinamento** (Células 5.1-5.2)
   - Divisão do dataset
   - Configura DataLoaders
   - Implementa loops de treinamento

### ⚠️ AVISOS IMPORTANTES

**⚠️ REINICIALIZAÇÃO NECESSÁRIA:**
Após a instalação do facenet-pytorch (Célula 1.1), você DEVE reiniciar o ambiente:
- \`Runtime\` > \`Restart and run all\`
- Ou \`Runtime\` > \`Restart session\`

**⚠️ AJUSTE DE PATHS:**
Na célula 1.2, verifique e ajuste se necessário:
\`\`\`python
MYDRIVE_PATH = '/content/drive/MyDrive/IA_CD_UFES/TCC'
\`\`\`
Se sua estrutura de pastas for diferente, atualize este caminho.

### 🎯 Execução por Seções

**Para execução modular, você pode rodar por seções:**

1. **Setup Completo:** Células 1.1 → 2.2
2. **Pré-processamento:** Células 3.1 → 3.2
3. **Modelos:** Células 4.1 → 4.2
4. **Treinamento:** Células 5.1 → 5.2

### 📊 Monitoramento

Durante o treinamento, monitore:
- **Loss de treinamento e validação**
- **Métricas MAE/MSE**
- **Uso de GPU** (acesse: \`Runtime\` > \`Change runtime type\` > \`GPU\`)

## 📊 Dataset

### Estrutura do Dataset:
- **285 pastas** (pessoas) × **15 imagens** cada = 4275 imagens
- **Poses:** front-facing, left-facing, right-facing, up-facing, down-facing
- **Formato:** JPEG + JSON com labels MST

### Labels MST:
- Arquivo \`monk_scale_value.json\` em cada pasta de pessoa
- Valores contínuos de 1.0 a 10.0 (67 valores únicos)

## 🔧 Pré-processamento

### Pipeline de Processamento:

1. **Filtragem por Pose:** Apenas faces frontais (configurável)
2. **Detecção Facial:** MTCNN + Haar Cascade (fallback)
3. **Recorte Facial:** Com margem de 15%
4. **Normalização de Cor:** Conversão para espaço LAB
5. **Calibração:** Baseada em dados de referência
6. **Data Augmentation:** Flip horizontal/vertical
7. **Redimensionamento:** 224×224 pixels

## 🤖 Modelos Implementados

### Arquiteturas:
1. **EfficientNet-B0** - Balance entre precisão e eficiência
2. **ConvNeXt-Tiny** - Arquitetura moderna baseada em transformers
3. **MobileNet-V3-Large** - Otimizado para dispositivos móveis
4. **VGG16** - Baseline clássica

## 🏋️ Treinamento

### Hiperparâmetros Principais:
- **Tamanho da imagem:** 224×224
- **Batch size:** 32
- **Épocas:** 30
- **Learning rate:** 1e-4
- **Divisão:** 65% treino, 20% validação, 15% teste

## 📈 Avaliação

### Métricas Principais:
- **MAE (Mean Absolute Error):** Erro absoluto médio
- **MSE (Mean Squared Error):** Erro quadrático médio
- **R² (Coeficiente de Determinação):** Variabilidade explicada

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

A licença aplica-se exclusivamente ao código-fonte.

Nenhum dataset, imagem, foto de participante ou material sensível 
está incluído, ou licenciado por este repositório.
---

## 💡 Dicas para Execução Bem-sucedida

1. **Verifique a GPU:** Certifique-se de que está usando GPU no Colab
2. **Siga a sequência:** Execute as células na ordem numérica
3. **Aguarde o processamento:** Algumas células (como descompactação) podem demorar
4. **Monitore recursos:** Verifique o uso de RAM e disco durante execução
5. **Salve resultados:** Faça download dos modelos treinados e métricas

**📞 Dúvidas?** Consulte a documentação dentro de cada célula do notebook para detalhes específicos de implementação. Cada célula contém documentação completa sobre objetivos, ações executadas e justificativas técnicas.

## Arquivo calibrate_refer_data.json:
- Dados de referência para calibração LAB
- Obtido da pessoa 52 do dataset original

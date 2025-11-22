# Modelos de Machine Learning

Este documento detalha os modelos de Machine Learning disponíveis neste diretório, focados na demonstração do *pipeline* de avaliação e recomendação de fotoprotetores e na reprodutibilidade.

## 💾 Modelos Disponíveis (Conjunto DS-20)

Os modelos disponíveis nesta pasta são versões de **Prova de Conceito (PoC)**, otimizadas para demonstrar a funcionalidade completa do *pipeline* de processamento de dados e predição.

| Detalhe | Descrição |
| :--- | :--- |
| **Conjunto de Dados** | **DS-20** (20% dos dados) |
| **Objetivo** | Demonstração e Reprodutibilidade |

#### Definição do Conjunto de Dados `DS-20`

O modelo com a tag `DS-20` foi treinado utilizando **apenas 20% do total de dados de imagens frontais (*front-facing*)** disponíveis para treinamento.

Esta limitação intencional foi adotada para:
1.  **Facilitar a Reprodutibilidade:** Permitir que o modelo e o *pipeline* sejam executados rapidamente em ambientes de teste.
2.  **Manter o Foco Didático:** Servir como um exemplo funcional do fluxo de trabalho.
3.  **Permitir executar a execução de protetor solar a partir do MST estimado**

**⚠️ Aviso Importante:**
Devido ao seu treinamento em uma fração limitada do *dataset*, este modelo é estritamente **didático** e não possui a precisão ou robustez necessárias para ser considerado um modelo de **produção final**. Os resultados obtidos com ele devem ser interpretados sob essa perspectiva.

## 🛠️ Treinamento e Modelos de Produção

**Somente 3 modelos DS-20 treinados estão disponíveis nesta pasta.**
VGG16 - Baseline clássica (Não disponibilizado no notebook final devido ao desempenho inferior e tamanho do modelo)

Para treinar novos modelos, com maior robustez e precisão (utilizando o *dataset* completo), siga as instruções detalhadas no *notebook* de treinamento:

[notebooks/Pipeline_best_MST_final.ipynb](../../../notebooks/pipeline_best_MST_final.ipynb)
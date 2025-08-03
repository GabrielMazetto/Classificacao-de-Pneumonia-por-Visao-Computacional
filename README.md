# Classificação de Pneumonia por Visão Computacional (RSNA Challenge)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

Este projeto explora o uso de Visão Computacional e Deep Learning para a classificação de pneumonia em radiografias de tórax, utilizando o dataset do desafio "RSNA Pneumonia Detection Challenge". O objetivo principal foi desenvolver um classificador binário eficaz, superando os desafios impostos pelo desbalanceamento de classes  e otimizando a arquitetura para minimizar a taxa de falsos negativos, uma métrica crítica em diagnósticos médicos.

## 🎯 O Desafio: Desbalanceamento de Dados

O conjunto de dados original do RSNA é composto por 26.684 imagens, mas é severamente desbalanceado: 77% das amostras de treino eram da classe "Não Pneumonia" e apenas 23% da classe "Pneumonia". Um modelo inicial treinado com esses dados apresentou uma performance enganosa, com acurácia de 78,77%, mas tendendo a prever sempre a classe majoritária, resultando em uma altíssima taxa de falsos negativos (81,81%).

## 💡 Metodologia e Experimentos

Para solucionar o problema, adotamos uma abordagem iterativa:

1.  **Balanceamento de Dados:** O conjunto de dados de treino foi balanceado através da exclusão de amostras da classe majoritária, resultando em uma nova distribuição de 50% para cada classe.
2.  **Treinamento Base:** Um modelo ResNet-50 foi treinado no dataset balanceado, servindo como nossa nova linha de base.
3.  **Fine-Tuning:** Para aprimorar a performance, realizamos o descongelamento progressivo das camadas finais do ResNet-50 (primeiro 5, depois 10 camadas), permitindo que o modelo se ajustasse melhor às características específicas das radiografias.
4.  **Data Augmentation:** Testamos o uso de técnicas de aumento de dados como rotação, zoom e alteração de contraste.
5.  **Análise de Interpretabilidade:** Implementamos o Grad-CAM utilizando PyTorch para visualizar quais regiões das imagens o modelo estava utilizando para fazer suas predições, confirmando o foco na área pulmonar.

## 🚀 Resultados Principais

A tabela abaixo resume a evolução da performance do modelo ao longo dos experimentos. A métrica principal, além da acurácia, foi a **Taxa de Falsos Negativos (FN Rate)**, pois errar ao não diagnosticar uma pneumonia é o cenário mais crítico.

| Experimento | Acurácia de Teste | Taxa de Falsos Negativos |
| :--- | :---: | :---: |
| Modelo Inicial (Desbalanceado) | 78.77% | 81.81% |
| Dataset Balanceado | 83.26% | 16.7% |
| Fine-Tuning (5 camadas) | 85.11% | 7.1% |
| **Fine-Tuning (10 camadas)** | **86.51%** | **3.9%** |
| Com Data Augmentation | 85.07% | 19.8% |

O melhor modelo foi o **ResNet-50 com as 10 últimas camadas descongeladas** , que alcançou uma **acurácia de 86,51%** e uma **taxa de falsos negativos de apenas 3,9%**, representando uma melhoria drástica em relação ao modelo inicial.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Frameworks:** TensorFlow/Keras e PyTorch
* **Arquitetura Principal:** ResNet-50
* **Técnicas:** Transfer Learning, Fine-Tuning, Balanceamento de Dados, Grad-CAM

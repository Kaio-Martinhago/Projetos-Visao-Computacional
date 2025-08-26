# Projetos em Visão Computacional

Este repositório contém projetos pessoais e acadêmicos desenvolvidos para aprimorar minhas habilidades em processamento de imagens e visão computacional. Os projetos foram criados como parte do meu estudo de frameworks como OpenCV e dlib.

---

### Detecção de Faces com CNN

Este projeto demonstra a detecção de faces em uma imagem estática utilizando uma Rede Neural Convolucional (CNN) pré-treinada da biblioteca `dlib`. O modelo utilizado é o `mmod_human_face_detector.dat`, conhecido por sua alta precisão.

**Tecnologias:**
- Python
- OpenCV
- dlib

**Como Executar:**
1.  Garanta que você tenha o ambiente virtual ativado e as bibliotecas necessárias instaladas.
2.  Baixe o arquivo de pesos `mmod_human_face_detector.dat` e coloque-o na pasta `Visao_Computacional_Guia_Completo/Weights/`.
3.  Execute o script no terminal:
    ```bash
    python cnn_face_detection.py
    ```

---

### Detecção de Faces com HOG

Neste projeto, a detecção de faces é realizada utilizando o algoritmo Histogram of Oriented Gradients (HOG) do `dlib`. Este método é uma abordagem clássica e eficiente, que serve como uma excelente base para entender os fundamentos da detecção de objetos.

**Tecnologias:**
- Python
- OpenCV
- dlib

**Como Executar:**
1.  Garanta que o ambiente virtual esteja ativado e as dependências instaladas.
2.  Execute o script no terminal:
    ```bash
    python hog_face_detection.py
    ```

# 03-Classificador-Gatos-e-Cachorros

Este projeto é um classificador de imagens que utiliza Redes Neurais Convolucionais (CNN) com o framework **TensorFlow/Keras** para distinguir entre imagens de cães e gatos. O modelo foi treinado com o dataset `cat_dog_2.zip` e demonstra as etapas de pré-processamento de imagens, construção e treinamento de uma CNN e avaliação de desempenho.

### Tecnologias e Dependências

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Seaborn

### Como Executar

1.  **Pré-requisitos:** Garanta que você tenha um ambiente virtual Python configurado e todas as bibliotecas listadas acima instaladas.
    * ```bash
        pip install tensorflow opencv-python numpy matplotlib seaborn
        ```

2.  **Organização de Arquivos:** Certifique-se de que o arquivo `cat_dog_2.zip` está na mesma pasta que o script `classificador_gatos_cachorros.py`.

3.  **Execução:** Abra o terminal na pasta do projeto e execute o script.
    * ```bash
        python classificador_gatos_cachorros.py
        ```
    * O script irá extrair o dataset, treinar o modelo e exibir a matriz de confusão e a acurácia. O modelo treinado será salvo nos arquivos `network.json` e `weights.hdf5` na mesma pasta.

### Sobre o Modelo

A arquitetura da rede neural consiste em camadas convolucionais e de pooling para extrair características das imagens, seguidas por camadas densas para a classificação final.

### Avaliação de Desempenho

O projeto inclui a avaliação do modelo usando métricas como **acurácia**, **matriz de confusão** e **relatório de classificação**, demonstrando a capacidade do modelo de generalizar para novas imagens.

### Teste com Imagem Única

Ao final da execução, o script carrega uma imagem de teste e utiliza o modelo treinado para fazer uma predição, mostrando o resultado final.

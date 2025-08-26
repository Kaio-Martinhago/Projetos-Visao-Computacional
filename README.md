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

1.  **Pré-requisitos:** Garanta que você tenha um ambiente virtual Python configurado e as bibliotecas necessárias instaladas.
    * ```bash
        pip install tensorflow opencv-python numpy matplotlib seaborn
        ```

2.  **Dataset:** Este projeto utiliza o dataset `cat_dog_2.zip`. Por ser um arquivo grande, ele não está incluído neste repositório. Faça o download a partir do link abaixo e coloque-o na mesma pasta do script `classificador_gatos_cachorros.py`.
    * **Link para download do dataset:** [Link para o dataset](https://drive.google.com/file/d/1SSSYSK7cjqGN1J7zTJABHkYUrV5PBT5n/view?usp=sharing)

3.  **Execução:** Abra o terminal na pasta do projeto e execute o script.
    * ```bash
        python classificador_gatos_cachorros.py
        ```
    * O script irá extrair o dataset, treinar o modelo e exibir a matriz de confusão e a acurácia. O modelo treinado será salvo na mesma pasta, nos arquivos `network.json` e `weights.hdf5`.

### Sobre o Modelo

A arquitetura da rede neural consiste em camadas convolucionais e de pooling para extrair características das imagens, seguidas por camadas densas para a classificação final.

### Avaliação de Desempenho

O projeto inclui a avaliação do modelo usando métricas como **acurácia**, **matriz de confusão** e **relatório de classificação**, demonstrando a capacidade do modelo de generalizar para novas imagens.

### Teste com Imagem Única

Ao final da execução, o script carrega uma imagem de teste e utiliza o modelo treinado para fazer uma predição, mostrando o resultado final.

# 04-Deteccao-Objetos-YOLOv4

Este projeto demonstra a detecção de objetos em imagens e vídeos usando o modelo **YOLOv4** (You Only Look Once) e o framework **Darknet**. O YOLOv4 é um algoritmo de detecção de objetos em tempo real que utiliza aprendizado profundo para identificar múltiplas instâncias de objetos em uma única imagem.

### Tecnologias e Pré-requisitos

* **Sistema Operacional:** Linux (Recomendado) ou WSL (Windows Subsystem for Linux) no Windows.
* **Framework:** Darknet
* **Linguagens/Bibliotecas:** C, Python, OpenCV, Matplotlib

### Configuração do Ambiente

1.  **Clone o Repositório Darknet:**
    * Abra o terminal e clone o repositório oficial do Darknet:
    ```bash
    git clone [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
    ```

2.  **Ajuste o Makefile:**
    * Navegue para a pasta `darknet` no seu terminal:
    ```bash
    cd darknet/
    ```
    * Edite o arquivo `Makefile` para habilitar o suporte à GPU, OpenCV e CUDNN (se tiver uma placa de vídeo NVIDIA compatível):
    ```bash
    # Abra o Makefile em um editor de texto e mude:
    # GPU=0  -> GPU=1
    # CUDNN=0 -> CUDNN=1
    # OPENCV=0 -> OPENCV=1
    ```

3.  **Compile o Darknet:**
    * No terminal, compile o projeto:
    ```bash
    make
    ```

4.  **Download dos Pesos do Modelo:**
    * Os pesos do YOLOv4 são muito grandes e não estão incluídos no repositório. Faça o download e coloque o arquivo na pasta `darknet/`.
    * **Link para download:** [yolov4.weights (245MB)](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
    ```bash
    # Ou use o comando wget no terminal:
    # wget [https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
    ```

### Como Executar a Detecção

#### Detecção em Imagem

1.  **Abra o terminal** na pasta **`darknet/`**.
2.  Execute o comando `darknet detect`, especificando o arquivo de configuração, os pesos e a imagem.
    * ```bash
      ./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg
      ```
    * Após a execução, a imagem com as detecções será salva como `predictions.jpg`.

3.  **Para visualizar a imagem**, você pode usar o script `deteccao_yolo_local.py` (dentro da pasta `04-Deteccao-Objetos-YOLOv4/`) com os comandos:
    * ```bash
      cd ../04-Deteccao-Objetos-YOLOv4/
      python deteccao_yolo_local.py
      ```

#### Detecção em Vídeo (Em Tempo Real)

1.  Coloque um arquivo de vídeo na pasta `darknet/` (ou em qualquer pasta com um caminho acessível).
2.  Execute o comando `darknet detector demo`, especificando o arquivo de dados, a configuração, os pesos e o caminho para o vídeo.
    * ```bash
      ./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show <caminho_para_o_video>
      ```

---

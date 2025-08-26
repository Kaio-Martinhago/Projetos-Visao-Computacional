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
2.  Baixe o arquivo de pesos `mmod_human_face_detector.dat` e coloque-o na pasta `Visao_Computacional_Guia_Completo/Weights/`. Você pode encontrar esse arquivo em sites como [o repositório de modelos do dlib](http://dlib.net/files/mmod_human_face_detector.dat.bz2).
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

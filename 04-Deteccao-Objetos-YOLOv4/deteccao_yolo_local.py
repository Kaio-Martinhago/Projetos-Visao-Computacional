import cv2
import matplotlib.pyplot as plt

def mostra_deteccao(path):
  """
  Função para exibir a imagem de detecção de objetos
  gerada pelo YOLO.
  """
  imagem = cv2.imread(path)
  if imagem is None:
    print(f"Erro: Não foi possível carregar a imagem em '{path}'.")
    return

  figura = plt.gcf()
  figura.set_size_inches(18, 10)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
  plt.show()

# O comando para rodar o YOLO deve ser feito no terminal,
# após a compilação do Darknet e download dos pesos.

# Exemplo de como você executaria a detecção no terminal:
# ./darknet detect cfg/yolov4.cfg yolov4.weights data/person.jpg

# Depois da execução, a imagem de resultado é salva como 'predictions.jpg'.
# A linha abaixo é o código Python para visualizar essa imagem.
mostra_deteccao('predictions.jpg')

import cv2
import dlib

imagem = cv2.imread('Visao_Computacional_Guia_Completo\Images\people2.jpg')

detector_face_cnn = dlib.cnn_face_detection_model_v1('Visao_Computacional_Guia_Completo\Weights\mmod_human_face_detector.dat')

deteccoes = detector_face_cnn(imagem, 1)
for face in deteccoes:
    l,t,r,b,c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
    print(c)
    cv2.rectangle(imagem, (l,t), (r,b), (255,255,0),2)
cv2.imshow('Imagem_cnn', imagem)
cv2.waitKey(0)
cv2.destroyWindow('Imagem_cnn')

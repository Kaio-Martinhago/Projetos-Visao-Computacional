import dlib
import cv2

imagem = cv2.imread('02-Deteccao-HOG/people2.jpg')
detector_faces = dlib.get_frontal_face_detector()

deteccoes = detector_faces(imagem, 2)

for face in deteccoes:
    l,t,r,b = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(imagem, (l,t), (r,b), (0,0,255), 2)
cv2.imshow('Imagem_hog', imagem)
cv2.waitKey(0)
cv2.destroyWindow('Imagem_hog')

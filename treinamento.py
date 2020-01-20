import cv2
import os # recursos sistema operacional
import numpy as np

eigenfaces = cv2.face.EigenFaceRecognizer_create(num_components=10, threshold=2)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')] # lista de caminhos da foto \\...
    # print(caminhos)
    #
    faces = [] # lista inicializada com vazio; guarda apenas as faces
    ids = []  # lista inicializada com vazio; guarda apenas o ID de cada pessoa
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY) # ler o path da img, põe em cinza
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1]) # Capturar IPs
        ids.append(id) # coloca na lista
        faces.append(imagemFace)
        # print(id)
        #cv2.imshow("Face", imagemFace) # se rodar e aparecer cada imagem, quer dizer q a linha àcima stá ok
        #cv2.waitKey(10) # milisegundos
    return np.array(ids), faces #converte a lista em np.array

ids, faces = getImagemComId()
#print(faces)
print("treinando...")
eigenfaces.train(faces, ids) #lê as imagens e faz o aprendizado supervisionado com as labels
eigenfaces.write('classificadorEigen.yml') # arquivo que é gerado para reconhecimento facial

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado")







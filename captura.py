import cv2
import numpy as np

# Classificador é treinado por haarcascade
classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml") # Carrega o arquivo para treinamento
classificadorOlho = cv2.CascadeClassifier("haarcascade-eye.xml") # Carrega arq para treinamento em detecção de olhos
# Captura imagem da webcan
"""
O dispositivo que fará a captura é passado pelo parâmetro.
Se parâmetro = 0, então o dispositivo é padrão, ou seja, a câmera do próprio notebook.
Se houver mais de uma câmera, então coloque: 0 para o 1°, 1 para o 2°, ...
"""
camera = cv2.VideoCapture(0)
amostra = 1 # Controla quantas fotos foram tiradas
numeroAmostras = 25 # Tirar 25 fotos de cada pessoa (usar sempre mais de 25)
id = input('Digite seu Identificador') # IP da pessoa: 1, 2, 3...
'''
formato da imagem: 
    pessoa.<id>.<numerofoto>.jpg
'''
largura, altura = 220, 220 # Padroniza o tamanho da imagem, pois o eigen-fisherFace, precisam de imagem de tam padrão
print('Capturando as faces... ')

while (True):  # Loop infinito
    conectado, imagem = camera.read()  # Faz a leitura da Webcan, à partir da var 'camera'
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converte imagem para escala de cinza
    #print(np.average(imagemCinza)) # Primeiro coloca esse cod no if lá em baixo
    '''
       facesdetectadas: set de faces que o algoritimo encontrou; é uma matriz que possui posições: x, y, l, a
       x = inicio da face
       y = término da face
       L = largura da face
       a = altura da face
       imagemCinza: imagem que quero detectar
       scaleFactor: definição da escala da imagem via webcan
       minSize: tamanho mínimo da imagem 
    '''
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.5,
                                                     minSize=(150, 150))
    for(x, y, l, a) in facesDetectadas: # faz o retângulo ao redor da face;
        '''
        .rectangulo: usa a imagem colorida para a detecção
        (0, 0, 225) = vermelho
        2 = borda
        '''
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l] # Ver se tem olhos na regiao da face que ele detectou anteriormente
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY) # Convertendo a regiao para escala de cinza
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho) # detecta o olho
        # Desenha retângulo ao redor dos olhos
        for(ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 225, 0), 2) # Desenha o retangulo na imagem colorida
            '''
            Código para salvar imagem, pois se ele entra no for significa que está encontrando a face
            0xff: cod exadecimal, sempre que 'q' for apertada faz uma imagem
            .resize: redimencionamento de img
            .imwrite(): monta o path do arq
            '''
            if(cv2.waitKey(1)) & 0xFF == ord('q'): # Só entra aqui se detectar a face e olhos
                # if colocado após import numpy, só entra se a imagem for clara
                # imgcinza = img da webcan, se vlr < 110, ela é mais escura
                if np.average(imagemCinza) > 50:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
                    print("[foto " + str(amostra) + "capturada com sucesso]")
                    amostra += 1

    cv2.imshow("Face", imagem)  # imshow: mostra iamgem capturada da webcan; imagen: variável da captura
    cv2.waitKey(1) # 1: esperando uma tecla
    if amostra >= numeroAmostras + 1:
        break # stop a captura da tela

print("Faces capturadas com sucesso")
camera.release()  # release: libera a memória
cv2.destroyAllWindows()

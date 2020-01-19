import cv2

# Classificador é treinado por haarcascade
classificador = cv2.CascadeClassifier("haarcascade-frontalface-default.xml") # Carrega o arquivo para treinamento
# Captura imagem da webcan
"""
O dispositivo que fará a captura é passado pelo parâmetro.
Se parâmetro = 0, então o dispositivo é padrão, ou seja, a câmera do próprio notebook.
Se houver mais de uma câmera, então coloque: 0 para o 1°, 1 para o 2°, ...
"""
camera = cv2.VideoCapture(0)

while (True):  # Loop infinito
    conectado, imagem = camera.read()  # Faz a leitura da Webcan, à partir da var 'camera'
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) # Converte imagem para escala de cinza
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
                                                     minSize=(100, 100))
    for(x, y, l, a) in facesDetectadas: # faz o retângulo ao redor da face;
        '''
        .rectangulo: usa a imagem colorida para a detecção
        (0, 0, 225) = vermelho
        2 = borda
        '''
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

    cv2.imshow("Face", imagem)  # imshow: mostra iamgem capturada da webcan; imagen: variável da captura
    cv2.waitKey(1)

camera.release()  # release: libera a memória
cv2.destroyAllWindows()

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
        '''
         Código para salvar imagem, pois se ele entra no for significa que está encontrando a face
         0xff: cod exadecimal, sempre que 'q' for apertada faz uma imagem
         .resize: redimencionamento de img
         .imwrite(): monta o path do arq
        '''
        if(cv2.waitKey(1)) & 0xFF == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemFace)
            print("[foto " + str(amostra) + "capturada com sucesso]")
            amostra +=1 #incrementação

    cv2.imshow("Face", imagem)  # imshow: mostra iamgem capturada da webcan; imagen: variável da captura
    #cv2.waitKey(1) # 1: esperando uma tecla, só coloque um .waitkeyneste codigo
    if(amostra >= numeroAmostras + 1):
        break # stop a captura da tela

print("Faces capturadas com sucesso")
camera.release()  # release: libera a memória
cv2.destroyAllWindows()

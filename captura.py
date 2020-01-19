import cv2

# Captura imagem da webcan
"""
O dispositivo que fará a captura é passado pelo parâmetro.
Se parâmetro = 0, então o dispositivo é padrão, ou seja, a câmera do próprio notebook.
Se houver mais de uma câmera, então coloque: 0 para o 1°, 1 para o 2°, ...
"""
camera = cv2.VideoCapture(0)

while (True): # Loop infinito
    conectado, imagem = camera.read() # Faz a leitura da Webcan, à partir da var 'camera'

    cv2.imshow("Face", imagem) # imshow: mostra iamgem capturada da webcan; imagen: variável da captura
    cv2.waitKey(1)

camera.release() # release: libera a memória
cv2.destroyAllWindows()
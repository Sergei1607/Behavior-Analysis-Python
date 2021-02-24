# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""
#Importar programas
import cv2
import numpy as np
import sys
import time as time


# Inicializar capturador

cap = cv2.VideoCapture("10R35.mp4")



#crear archivo de salida 
filename= open("10R35",'w')
sys.stdout = filename


#establecer tiempo

timeout = time.time() + 1000000


# Imagen de referencia

video= cv2.VideoCapture("peces4.mp4")
_, first_frame = video.read()
first_gray= cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray= cv2.GaussianBlur(first_gray, (1, 5), 0)

img = cv2.imread("mask2.jpg")
x = 100
y = 80
width = 200
height = 150
roi = img[y: y + height, x: x + width]

# Eliminar background

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist ([hsv_roi], [0], None, [180], [0, 180])
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2. NORM_MINMAX)

subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold =10, detectShadows=False)

_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#Crear video de salida 

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out= cv2.VideoWriter("output.avi", fourcc, 20.0, (640,480)) 

#Optical flow 

lk_params = dict(winSize = (10, 10), maxLevel =10, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Utilización del mouse

def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        point_selected = True
        old_points= np.array([[x, y]], dtype=np.float32)
        
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)        
        
point_selected = False 
point = ()  
old_point = np.array([[]])   
fr=0

#LOOP     

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    test=0
     
    detectado, cuadro = cap.read()
    if detectado:
        fr = fr+1
    
    out.write(frame)
    
    mask = subtractor.apply(frame, cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1))
    
#Generar circulo de referencia    
    
    if point_selected is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)
        
#Detección del movimiento        
        
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, mask, **lk_params) 
        old_gray = gray_frame.copy()
        old_points = new_points
        
# adquirir coordenadas   
            
        print(x,",",y,",",0)
                   
        x, y = new_points.ravel()
        cv2.circle (frame, (x, y), 5, (255, 0, 0), -1)
        
#poner tiempo en pantalla y coordenadas   
        
        #Y (y1,x1), (y2,x2)
        cv2.line(frame, (655, 0), (655, 1000), (255,200,0), 2)
        
        
         #X (y1,x1), (y2,x2)
        cv2.line(frame, (0, 420), (2000, 420), (255,200,0), 2)
        
        
        #Texto en X
        cv2.putText(frame, str(0), (0, 275), 2, 0.5, (0,0,255))
        cv2.putText(frame, str(320), (320, 275), 2, 0.5, (0,0,255))
        cv2.putText(frame, str(640), (610, 275), 2, 0.5, (0,0,255))
        
        #Texto en Y
        cv2.putText(frame, str(0), (320, 12), 2, 0.5, (250,0,0))
        cv2.putText(frame, str(240), (320, 255), 2, 0.5, (250,0,0))
        cv2.putText(frame, str(480), (320, 475), 2, 0.5, (250,0,0))
        
#mostrar pantalla        
    
    cv2.imshow("Frame", frame)
   
     
#cerrar pantalla al tiempo establecido    
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    if test == 5 or time.time() > timeout:
        break
    test = test - 1
    
cap.release()
out.release()
cv2.destroyAllWindows() 


#Exportar los videos en 640*480(4:3) en expandir

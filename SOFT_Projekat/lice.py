import dlib
import numpy as np
import sys
import cv2
import neuronska
import imutils
import argparse
from imutils.video import VideoStream
from imutils import face_utils
from numpy import array
from matplotlib import pyplot as plt

neuronska_mreza = neuronska.Neuronska()

ulazi = array([[0.192, 0.18, 0.184],  #1
               [0.329, 0.314, 0.325],  #1
               [0.484, 0.465, 0.47],  #1
               [0.356, 0.356, 0.376],
               [0.139, 0.137, 0.135],  #1
               [0.3, 0.31, 0.33],
               [0.2, 0.21, 0.28],
               [0.34, 0.31, 0.325],  #1
               [0.49, 0.465, 0.48],  #1
               [0.25, 0.2, 0.22],  #1
               [0.139, 0.14, 0.144],
               [0.5, 0.51, 0.53],
               [0.66, 0.68, 0.69],
               [0.192, 0.19, 0.191],  #1
               [0.329, 0.311, 0.323],  #1
               [0.482, 0.46, 0.473],  #1
               [0.35, 0.351, 0.372],
               [0.139, 0.130, 0.138],  #1
               [0.3, 0.34, 0.38],
               [0.25, 0.28, 0.32],
               [0.38, 0.35, 0.36],  #1
               [0.49, 0.42, 0.46],  #1
               [0.6, 0.55, 0.57],  #1
               [0.139, 0.145, 0.149],
               [0.58, 0.6, 0.62],
               [0.7, 0.72, 0.729],
               [0.192, 0.17, 0.182],  #1
               [0.32, 0.30, 0.311],  #1
               [0.49, 0.465, 0.48],  #1
               [0.6, 0.6, 0.66],
               [0.139, 0.132, 0.136],  #1
               [0.21, 0.22, 0.25],
               [0.211, 0.25, 0.29],
               [0.8, 0.77, 0.78],  #1
               [0.56, 0.522, 0.526],  #1
               [0.46, 0.44, 0.455],  #1
               [0.11, 0.12, 0.144],
               [0.55, 0.56, 0.59],
               [0.668, 0.68, 0.699],
               [0.2, 0.18, 0.191],  #1
               [0.329, 0.3, 0.325],  #1
               [0.66, 0.63, 0.633],  #1
               [0.45, 0.455, 0.46],
               [0.8, 0.75, 0.775],  #1
               [0.5, 0.52, 0.533],
               [0.66, 0.67, 0.68],
               [0.77, 0.72, 0.745],  #1
               [0.55, 0.51, 0.53],  #1
               [0.8, 0.76, 0.78],  #1
               [0.76, 0.78, 0.8],
               [0.58, 0.65, 0.62],
               [0.7, 0.72, 0.71],
               [0.192, 0.18, 0.178],  # 1
               [0.329, 0.314, 0.311],  # 1
               [0.484, 0.465, 0.46],  # 1
               [0.356, 0.376, 0.366],
               [0.139, 0.137, 0.138],  # 1
               [0.3, 0.31, 0.305],
               [0.2, 0.24, 0.22],
               [0.34, 0.31, 0.3],  # 1
               [0.49, 0.465, 0.455],  # 1
               [0.25, 0.2, 0.19],  # 1
               [0.139, 0.14, 0.142],
               [0.5, 0.53, 0.51],
               [0.66, 0.68, 0.67],
               [0.192, 0.18, 0.179],  # 1
               [0.329, 0.322, 0.311],  # 1
               [0.482, 0.479, 0.473],  # 1
               [0.35, 0.373, 0.372],
               [0.139, 0.130, 0.129],  # 1
               [0.3, 0.34, 0.32],
               [0.25, 0.3, 0.32],
               [0.38, 0.35, 0.33],  # 1
               [0.49, 0.42, 0.41],  # 1
               [0.6, 0.55, 0.53],  # 1
               [0.139, 0.145, 0.143],
               [0.58, 0.6, 0.59],
               [0.7, 0.72, 0.715],
               [0.192, 0.17, 0.185],  # 1
               [0.32, 0.30, 0.31],  # 1
               [0.49, 0.465, 0.488],  # 1
               [0.6, 0.7, 0.66],
               [0.139, 0.132, 0.131],  # 1
               [0.21, 0.26, 0.25],
               [0.211, 0.3, 0.29],
               [0.8, 0.77, 0.72],  # 1
               [0.56, 0.522, 0.522],  # 1
               [0.46, 0.455, 0.455],  # 1
               [0.11, 0.12, 0.12],
               [0.55, 0.56, 0.56],
               [0.668, 0.68, 0.68],
               [0.2, 0.18, 0.18],  # 1
               [0.329, 0.3, 0.3],  # 1
               [0.66, 0.63, 0.63],  # 1
               [0.45, 0.455, 0.455],
               [0.8, 0.75, 0.75],  # 1
               [0.5, 0.52, 0.52],
               [0.66, 0.69, 0.67],
               [0.77, 0.76, 0.745],  # 1
               [0.55, 0.51, 0.5],  # 1
               [0.8, 0.76, 0.72],  # 1
               [0.76, 0.82, 0.8],
               [0.58, 0.65, 0.67],
               [0.7, 0.72, 0.735]
               ])
izlazi = array([[1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0
                 ]]).T

neuronska_mreza.obuci(ulazi, izlazi, 20000)
ap = argparse.ArgumentParser()
ap.add_argument("-r","--picamera",type = int, default = -1, help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-p","--shape-predictor",required=True,help="path to facial landmark predictor")
ap.add_argument("-s","--slika",required=True,help="path to the slika")
args = vars(ap.parse_args())
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

vs = VideoStream(usePiCamera=args["picamera"]>0).start()



#plt.show()
while True:
    frame = vs.read()
    frame = imutils.resize(frame,height=800,width=600)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100,120)

    plt.subplot(121), plt.imshow(frame, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
   # plt.show()
    rects = detector(gray,0)
    for rect in rects:
        shape = predictor(gray,rect)
        shape = face_utils.shape_to_np(shape)

        for(x,y) in shape:
            cv2.circle(frame, (x,y),1,(0,0,255),-1)

    coords = np.zeros((68, 2), dtype="int")
    shape = predictor(gray, rect)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    usta = np.zeros((12, 2), dtype="int")
    for j in range(48, 60):
        usta[j - 48] = coords[j]

    obrve = np.zeros((2, 2), dtype="int")
    obrve[0] = coords[21]
    obrve[1] = coords[22]
    cetiri = np.zeros((4, 2), dtype="int")
    cetiri[0] = usta[0]
    cetiri[1] = usta[6]

    maxy = usta[0, 1]
    k = 1

    for i in range(1, 6):
        if (usta[i, 1] <= maxy):
            maxy = usta[i, 1]
            k = i
    # gornja
    cetiri[2] = usta[k]
    k = 7
    miny = usta[7, 1]
    for j in range(7, 12):
        if (usta[j, 1] >= miny):
            miny = usta[j, 1]
            k = j
    # donja
    cetiri[3] = usta[k]
    cv2.rectangle(frame, (cetiri[0, 0], maxy), (cetiri[1, 0], miny), (0, 255, 0), 1)
    cv2.line(frame,  (cetiri[0, 0], (maxy + miny) / 2), (cetiri[1, 0], (maxy + miny) / 2), (0, 0, 255), 1)
    print obrve[1,1]
    x1 = obrve[0,1]+20
    x2 = obrve[1,1]+(-20)
    x3 = obrve[0, 0]+(-10)
    x4 = obrve[1, 0]+10
  #  cv2.line(frame, (obrve[0, 0], obrve[0,1]), (obrve[1, , 0], obrve[1,1]), (0, 0, 255), 1)
    cv2.rectangle(frame, (obrve[0, 0]+(-10), obrve[0,1]+20), (obrve[1, 0]+10, obrve[1,1]+(-20)), (255, 0, 0), 1)
    cropimg = frame[x2:x1,x3:x4]

    yy = (maxy + miny) / 2
    prva = float(yy) / 1000
    druga = float(cetiri[0, 1]) / 1000
    treca = float(cetiri[1, 1]) / 1000

    print "Predvidjanje neuronske mreze: ->  "
    nova = neuronska_mreza.predvidi(array([prva, druga, treca]))
    print neuronska_mreza.predvidi(array([prva, druga, treca]))

    if (nova >= 0.7):
        cv2.putText(frame, "Face is HAPPY", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    elif (nova >= 0.42 and nova < 0.68):
        cv2.putText(frame, "Face is NEUTRAL", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    else:
        cv2.putText(frame, "Face is SAD", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    cv2.imshow("FRAME",frame)
    key = cv2.waitKey(1) & 0xFF
    gray2 = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray2, 100, 120)

    plt.subplot(121), plt.imshow(cropimg, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()
    #cv2.imshow("crop",cropimg)
    if key == ord("e"):
        break
cv2.destroyAllWindows()
vs.stop()
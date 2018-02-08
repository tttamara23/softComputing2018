import dlib
import numpy as np
import cv2
import neuronska
import imutils
import pickle
import argparse
from imutils.video import VideoStream
from numpy import array
from matplotlib import pyplot as plt

neuronska_mreza = neuronska.Neuronska()
def loadDataset(dataFile):
    raw_data = open(dataFile, 'rt')
    data = np.genfromtxt(raw_data, delimiter=",")
    row_count = sum(1 for row in data)
    return data

def parseArguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r","--picamera",type = int, default = -1, help="whether or not the Raspberry Pi camera should be used")
    ap.add_argument("-p","--shape-predictor",required=True,help="path to facial landmark predictor")
    ap.add_argument("-d","--data",required=True,help="path to the dataset file")
    args = vars(ap.parse_args())
    return args

def pronadjiBelePikseleNos(iviceNos):
    brojBelihPikselaNos = 0
    ukupnoPikselaNos = 0
    for i in iviceNos:
        for el in i:
            ukupnoPikselaNos = ukupnoPikselaNos + 1
            if (el == 255):
                brojBelihPikselaNos = brojBelihPikselaNos + 1
    return brojBelihPikselaNos, ukupnoPikselaNos

def pronadjiBelePikseleCelo(iviceCelo):
    brojBelihPikselaCelo = 0
    ukupnoPikselaCelo = 0
    for i in iviceCelo:
        for el in i:
            ukupnoPikselaCelo = ukupnoPikselaCelo + 1
            if (el == 255):
                brojBelihPikselaCelo = brojBelihPikselaCelo + 1
    return brojBelihPikselaCelo, ukupnoPikselaCelo

def printPercents(nova,frame):
    brojac = 0
    for deo in nova:
        procenat = 100*deo / sum(nova)
        if(brojac==0):
             cv2.putText(frame, "Percent of surprised: %f " %procenat, (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
        elif(brojac==1):
             cv2.putText(frame, "Percent of happy: %f " %procenat, (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
        elif(brojac==2):
             cv2.putText(frame, "Percent of angry: %f " %procenat, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
        elif(brojac==3):
            cv2.putText(frame, "Percent of sadness: %f " % procenat, (5, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255),2)
        elif(brojac==4):
            cv2.putText(frame, "Percent of neutral: %f " % procenat, (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255),2)
        brojac = brojac + 1

        print(procenat)

args = parseArguments()
#data = loadDataset(args["data"])
#row_count = sum(1 for row in data)
#ulazi = data[1:row_count,0:5]
#izlazi = data[1:row_count,5:10]
#neuronska_mreza.obuci(ulazi, izlazi, 20000)
#with open('neuronskaMrezaVideo','wb') as output: pickle.dump(neuronska_mreza,output,pickle.HIGHEST_PROTOCOL)
with open('neuronskaMrezaVideo','rb') as input: neuronska_mreza = pickle.load(input)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

vs = VideoStream(usePiCamera=args["picamera"]>0).start()

#plt.show()
while True:
    frame = vs.read()
    frame = imutils.resize(frame,height=800,width=600)
    frame1 = vs.read()
    frame1 = imutils.resize(frame, height=800, width=600)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects = detector(gray,0)
    for (i, rect) in enumerate(rects):
        shape = predictor(frame, rect)	#prediktor vraca shape objekat koji u sebi sadrzi x i y koordinate tacaka
        coords = np.zeros((68,2),dtype="int")

        for i in range(0, 68):
           coords[i] = (shape.part(i).x, shape.part(i).y)

        for (x, y) in coords:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        usta=np.zeros((12,2),dtype="int")
        for j in range(48,60):
            usta[j-48] = coords[j]
        obrve = np.zeros((3, 2), dtype="int")
        obrve[0] = coords[21]  # leva granica za racunanje broja bora na nosu (angry emocija)
        obrve[1] = coords[22]  # desna granica
        obrve[2] = coords[29]  # donja granica(tacka na nosu)

        celo = np.zeros((4, 2), dtype="int")
        celo[0] = coords[19]  # leva granica za racunanje broja bora na celu (sad emocija)
        celo[1] = coords[24]  # desna granica - ROI je iznad obrva(celo)
        celo[2] = coords[27]
        celo[3] = coords[30]

        gornja_ociju = coords[43][1]  # gornja tacka levog oka
        donja_ociju = coords[47][1]  # donja tacka levog oka - za racunanje koliko je oko otvoreno
        gornja_obrva = coords[24][1]  # gornja tacka desne obrve - sve su uzete y koordinate
        razlika_oci = gornja_ociju - donja_ociju  # sirina - koliko je oko otvoreno

        donja_usna = coords[57][
            1]  # donja i gornja tacka usana - za racunanje koliko su usta otvorena (npr kod surprised vise... itd)
        gornja_usna = coords[51][1]
        nos_gore = coords[27][1]
        odnos_donja_nos = (gornja_usna - donja_usna) / float((nos_gore - donja_usna))
        odnos_oci_obrve = (gornja_obrva - gornja_ociju) / float((gornja_obrva - donja_ociju))
        sirina_oci = coords[42][
            0]  # leva tacka levog oka i desna isto - uzima se sirina ociju i pravi se odnos visina ociju/sirina ociju
        sirina_oci_2 = coords[45][0]
        odnos_oci = razlika_oci / float((sirina_oci - sirina_oci_2))
        for (x, y) in usta:
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # tacke koje se koriste za uzimanje regiona od interesa , tj pomeramo se od obrva levo i desno da bi uzeli region nosa
        x2 = obrve[1, 1] + (-15)
        x3 = obrve[0, 0] + float((obrve[1, 0] - obrve[0, 0])) / 10
        x4 = obrve[1, 0] - float((obrve[1, 0] - obrve[0, 0])) / 10
        x1 = obrve[2, 1]

        # tacke za uzimanje regiona cela, tj pomeramo se od obrva gore na celo
        celox1 = celo[0, 0]
        celox2 = celo[1, 0]
        celovisina = celo[2, 1] - celo[3, 1]
        if (celo[0, 1] <= celo[1, 1]):
            celoy1 = celo[0, 1] + 2 * float(celovisina) / 3
            celoy2 = celo[0, 1] + float(celovisina) / 6
        else:
            celoy1 = celo[1, 1] + 2 * float(celovisina) / 3
            celoy2 = celo[1, 1] + float(celovisina) / 6

        isecakNos = frame1[x2:x1,int(x3):int(x4)]
        isecakCelo = frame1[int(celoy1):int(celoy2), int(celox1):int(celox2)]


    sivaSlikaNos = cv2.cvtColor(isecakNos, cv2.COLOR_BGR2GRAY)
    iviceNos = cv2.Canny(sivaSlikaNos, 100, 120)
    sivaSlikaCelo = cv2.cvtColor(isecakCelo, cv2.COLOR_BGR2GRAY)
    iviceCelo = cv2.Canny(sivaSlikaCelo, 100, 120)

    brojBelihPikselaNos, ukupnoPikselaNos = pronadjiBelePikseleNos(iviceNos)
    brojBelihPikselaCelo, ukupnoPikselaCelo = pronadjiBelePikseleCelo(iviceCelo)

    print "Predvidjanje neuronske mreze: ->  "
    print(odnos_oci, odnos_oci_obrve, odnos_donja_nos, brojBelihPikselaNos / float(ukupnoPikselaNos),
          brojBelihPikselaCelo / float(ukupnoPikselaCelo))
    nova = neuronska_mreza.predvidi(array(
        [odnos_oci, odnos_oci_obrve, odnos_donja_nos, brojBelihPikselaNos / float(ukupnoPikselaNos),
         brojBelihPikselaCelo / float(ukupnoPikselaCelo)]))
    print nova
    printPercents(nova, frame)
    cv2.imshow("FRAME",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("e"):
        break
cv2.destroyAllWindows()
vs.stop()
import cv2

print("OpenCV kullanarak Nesne Algılama  ")

#img = cv2.imread('resim1.jpg')
goruntu= cv2.VideoCapture(0)


goruntu.set(8,8)#genişlik
goruntu.set(8,8)#genişlik
list1 = []
veri = 'coco.names'

with open(veri,'rt') as d:
    list1 = d.read( ).rstrip('\n').split('\n')  # read okuma 0 rstrip sondaki karakterleri kaldırır 0 Dizeyi virgül ve ardından boşluk  ayırıcı olarak ayırır
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'
    intr = cv2.dnn_DetectionModel(configPath , weightsPath)
    intr.setInputSize(320,320)  # giriş ölçegi
    intr.setInputScale(1.10 / 127.5)
    intr.setInputSwapRB(True)


while True:

    success,img = goruntu.read()
    Ids, acna, ider = intr.detect(img, confThreshold=0.6)#Nesneyi tespit etmek için eşik(0.6)ve(0.5) en iyi olan
    print(Ids, ider)


    if len(Ids) !=0:
        for Ids, ecna, ide in zip(Ids.flatten(), acna.flatten(), ider):
            cv2.rectangle(img, ide, color=(0, 255, 0), thickness=5)
            cv2.putText(img, list1[Ids-1].upper(), (ide[0]+10,ide[1]+30),#upper büyük harf
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
            cv2.putText(img, str(round(ecna*7,3)), (ide[1]+50,ide[1]+70),
                        cv2.FONT_ITALIC, 1,(0, 255, 255),2)

    if len(acna) !=0:
        for Ids, ecna, ide in zip(Ids.flatten(), acna.flatten(), ider):
            cv2.rectangle(img, ide, color=(0, 255, 0), thickness=5)
            cv2.putText(img, list1[Ids-1].upper(), (ide[0]+10,ide[1]+30),#upper büyük harf
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0), 1)
            cv2.putText(img, str(round(ecna*7,3)), (ide[1]+50,ide[1]+70),
                        cv2.FONT_ITALIC, 1,(0, 255, 255),2)


    cv2.imshow("ekran", img)
    if cv2.waitKey(1) & 0xFF== ord('w'):
       break

#farklı yazı sitileri için
#cv.2.FONT_HERSHEY_SIMPLEX = normal boyutlu sans-serif yazı tipi
#cv2.FONT_HERSHEY_COMPLEX = normal boyutlu serif yazı tipi






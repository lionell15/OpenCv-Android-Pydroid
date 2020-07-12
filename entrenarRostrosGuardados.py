import cv2
import os 
import numpy as np

dataPath = 'Data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList) 
labels = [] 
facesData = [] 
label = 0 
for nameDir in peopleList: 
    personPath = dataPath + '/' + nameDir
    #print('Leyendo las imágenes') 
    for fileName in os.listdir(personPath):
        #print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0)) 

    label = label + 1
#print('labels= ',labels)
nume= len(peopleList)
for name in range(nume):
    print('Número de etiquetas'+str(name)+': ',np.count_nonzero(np.array(labels)==name))
    
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Entrenando...")

face_recognizer.train(facesData, np.array(labels))

face_recognizer.write('modeloLBPHFace.xml')

print("Modelo Entrenado y almacenado")

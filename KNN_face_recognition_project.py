import cv2
import math
import numpy 
import glob
import os
import face_recognition
import sys



def takeTrainImagine():
    #number of folder/files in Train_img
    nof = os.listdir('./Train_img')
    vectors = {}
    for name in nof:
        #Turn variable in curly braces to string, this command is to take a path of every image of a person folder.
        paths = glob.glob(f"./Train_img/{name}/*.jpg") 
        #Array that contain an image of a person
        dataOfPerson = [] 
        # To take a sigle image to execute
        for path in paths: 
            # Read image
            img = cv2.imread(path) 
            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces
            face_locations = face_recognition.face_locations(gray)
            # Restrict output area
            if len(face_locations) ==  0:
                continue

            face_encodings = face_recognition.face_encodings(img, face_locations)
            dataOfPerson.append(face_encodings[0])
        vectors.update({name: dataOfPerson})
    return vectors


def recog(img, source_data):
    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detect faces
    face_locations = face_recognition.face_locations(gray)
    # Restrict output area
    if len(face_locations) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(img, face_locations)
    names = []
    for targets in face_encodings:
        nameOfFace = customKnnAlgorithm(targets, source_data, threshold=0.35)
        names.append(nameOfFace)
    return names, face_locations

def distantVector(img_avg: numpy.ndarray, x: numpy.ndarray)-> float:  
    r = numpy.sqrt(numpy.sum((img_avg - x)**2))
    return r

def customKnnAlgorithm(target:numpy.ndarray, source_data, threshold=0.3):
    '''
    This is a custom KNN Algorithm because during testing I came to realized that 
    the faces are close toghether and not scatterd out, so instead of calculate
    all the distant I create a threshold = 0.3 as a restriction.
    '''
    nameAndValue = dict.fromkeys(source_data)
    for key, value in source_data.items():
        values = numpy.array(value)
        distantToTarget = []
        minOfResults = 0
        meanOfResults = 0
        meanOfMinAndMean = 0
        for value in values:
            d = distantVector(value, target)
            distantToTarget.append(d)d
        minOfResults = min(distantToTarget)
        meanOfResults = numpy.mean(distantToTarget)
        meanOfMinAndMean = numpy.mean([minOfResults, meanOfResults])
        nameAndValue.update({key:meanOfMinAndMean})
    MinOfNAV = min(nameAndValue.keys(), key=lambda x:nameAndValue[x])
    if nameAndValue[MinOfNAV] > threshold:
        return "Unknown"
    else:
        return MinOfNAV

def faceBoxing(x1, y1, x2, y2, name, target):
    imgshow = target.copy()
    cv2.rectangle(imgshow, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(imgshow, name, (x1, y2-5),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    return imgshow

if __name__ == "__main__":
    img = cv2.imread(sys.argv[1] )
    source_data = takeTrainImagine()
    names, face_locations = recog(img, source_data)
    img_show = img.copy()
    for i in range(len(names)):
        y1, x2, y2, x1 = face_locations[i]
        img_show = faceBoxing(x1, y1, x2, y2, names[i], img_show )
    cv2.imwrite("Results/output.png", img_show)



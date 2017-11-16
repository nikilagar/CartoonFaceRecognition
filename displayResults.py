import LandMarkPointsExtractor.extractAndSaveLandmarkConfigurations as LM
import cv2
import numpy as np
import meshWarping
from random import randint
import matplotlib.pyplot as plt

counter = 0

def drawPolyLines(imgFile,landMarkPoints):
    global counter
    image = cv2.imread(imgFile)
    cv2.imshow("Initial Image", image)
    finalFeaturePoints = []
    differentFeatures = [17, 22, 27, 36, 42, 48, 61, 68]
    pre = 0
    for i in range(8):
        listOfPoints = []
        for j in range(pre, differentFeatures[i]):
            listOfPoints.append(landMarkPoints[j])
        pre = j + 1
        finalFeaturePoints.append(listOfPoints)
    for i in range(8):
        arr = np.array(finalFeaturePoints[i])
        cv2.polylines(image, np.int32([arr]), 0, (0, 255, 0))
    cv2.imshow(imgFile + str(counter), image)
    return image

def convImage(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    return img

def showCaricature(model1, model2, meanImageCoordinates):
    global counter
    
    fig = plt.figure()
    axes=[]
    axes.append(fig.add_subplot(2,2,1))
    axes.append(fig.add_subplot(2,2,2))
    axes.append(fig.add_subplot(2,2,3))
    axes.append(fig.add_subplot(2,2,4))

    axes[0].set_title("Original image")
    axes[1].set_title("Caricature of image")
    axes[2].set_title("Initial Landmark Points")
    axes[3].set_title("Final Landmark Points")
    
    imgFile = "Dataset/temp2/6.jpg"
    initialLandmarkPoints = LM.saveAndReturnLandMarkPoints(imgFile)

    ima = cv2.imread(imgFile)
    axes[0].imshow(convImage(ima))
    ima = drawPolyLines(imgFile, initialLandmarkPoints)
    axes[2].imshow(convImage(ima))
    counter += 1

    X = [initialLandmarkPoints[i][0] for i in range(len(initialLandmarkPoints))]
    Y = [initialLandmarkPoints[i][1] for i in range(len(initialLandmarkPoints))]
    Xtest = np.array(X)
    Ytest = np.array(Y)

    for i in range(len(meanImageCoordinates)):
        Xtest[i] = Xtest[i] - meanImageCoordinates[i][0]
        Ytest[i] = Ytest[i] - meanImageCoordinates[i][1]

    lisx = [ Xtest ]
    lisy = [Ytest]
    Xtest = np.array(lisx)
    Ytest = np.array(lisy)


    Xtest = model1.predict(Xtest)
    Ytest = model2.predict(Ytest)

    finalLandMarkPoints = []
    diffLandMarkPoints = []
    for i in range(len(X)):
        X[i] = X[i] + Xtest[0][i]
        Y[i] = Y[i] + Ytest[0][i]
        finalLandMarkPoints.append([X[i],Y[i]])
        diffLandMarkPoints.append([Xtest[0][i], Ytest[0][i]])

    ima = drawPolyLines(imgFile, finalLandMarkPoints)
    axes[3].imshow(convImage(ima))
    ima = meshWarping.getWarpedImage(imgFile, diffLandMarkPoints)
    axes[1].imshow(convImage(ima))
    plt.suptitle('Caricature Generation')
    plt.show()
    # cv2.waitKey(0)

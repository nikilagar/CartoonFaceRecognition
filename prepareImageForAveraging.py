import sys
import os
import glob
import LandMarkPointsExtractor.extractAndSaveLandmarkConfigurations as LM

imagesPath = sys.argv[1]
for image in glob.glob(os.path.join(imagesPath, "*.jp*g")):
    LM.saveAndReturnLandMarkPoints(image,image + ".txt")
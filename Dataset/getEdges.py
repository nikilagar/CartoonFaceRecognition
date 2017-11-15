import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

img = cv2.imread(sys.argv[1],0)
edges = cv2.Canny(img,100,200)

plt.subplot(122),plt.imshow(edges,cmap = 'binary')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

import cv2
import matplotlib.pyplot as plt
    

def convImage(img):
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])
    return img


imgFile = "Dataset/tem/SadMulayam.jpeg"
ima = cv2.imread(imgFile)

fig = plt.figure()
axes = []
axes.append(fig.add_subplot(1,2,1))
axes.append(fig.add_subplot(1,2,2))

axes[0].set_title("Original image")
axes[1].set_title("Sad caricature")

axes[1].imshow(convImage(ima))

imgFile = "Dataset/temp2/OrigMulayam.jpeg"

ima = cv2.imread(imgFile)
axes[0].imshow(convImage(ima))
plt.suptitle('Caricature Generation')
plt.show()
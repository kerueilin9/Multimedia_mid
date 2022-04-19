import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

img2 = cv2.imread("2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
img2 = cv2.resize(img2, (28, 28)).reshape(1, -1)
ret, img2 = cv2.threshold(img2, 187, 255, cv2.THRESH_BINARY)
# =============================================================================
# img2 = img2.reshape(28, 28)
# cv2.imshow("123",img2)
# cv2.waitKey(0)
# =============================================================================
img2 = img2[0]
print(img2)
test_data = np.uint8(np.loadtxt("label.txt"))

video = cv2.VideoCapture("test_dataset.avi")
success = True

video_data = []

while(success):
    success, image = video.read()
    
    if not success:
        break
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    video_data.append(image)
    
data = np.array(video_data).reshape(len(video_data), -1)
# =============================================================================
# 
# _, axes = plt.subplots(nrows = 1, ncols = 5, figsize = (20, 10))
# 
# for ax, image, label in zip(axes, data, test_data):
#     ax.set_axis_off()
#     
#     image = image.reshape(28,28)
#     
#     ax.imshow(image, cmap = plt.cm.gray_r, interpolation = "nearest")
#     
#     ax.set_title("Training: %i" % label)
#     
#     ax.get_figure().savefig("output1.png")
#     
# =============================================================================

clf = KNeighborsClassifier()
x_train, x_test, y_train, y_test = train_test_split(data, test_data, test_size=0.05)

#print(x_train)
clf.fit(x_train, y_train)
MyPredict = clf.predict([img2])
print(MyPredict)



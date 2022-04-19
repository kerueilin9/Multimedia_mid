import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

test_data = np.uint8(np.loadtxt("label.txt"))

video = cv2.VideoCapture("test_dataset.avi")
img2 = cv2.imread("2.jpg")
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

print(x_train)
clf.fit(x_train, y_train)
MyPredict = clf.predict(x_test)



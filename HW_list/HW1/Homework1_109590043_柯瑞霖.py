import cv2
import numpy as np


kernel = np.ones((3, 3), np.uint8)

img = cv2.imread("../../opencv/img/1.jpg")

img = cv2.resize(img, (0,0), fx = 1.2, fy = 1.2)

img2 = np.copy(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eq = cv2.equalizeHist(gray)

blur = cv2.GaussianBlur(eq, (3, 3), 0)

ret, img_out = cv2.threshold(blur, 187, 255, cv2.THRESH_BINARY)

out1 = cv2.morphologyEx(img_out, cv2.MORPH_OPEN, kernel, iterations = 1)

#out2 = cv2.morphologyEx(out1, cv2.MORPH_CLOSE, kernel, iterations = 1)

canny = cv2.Canny(out1, 150, 200)

sp = img.shape

roi = canny[int(sp[0]*0.5):sp[0], 0:sp[1]]
# =============================================================================
# lines = cv2.HoughLinesP(img_out,1 , np.pi/180, 10, 50, 4)
# =============================================================================

minLineLength = int(sp[0] // 8)
lines = cv2.HoughLinesP(roi,0.5, np.pi/180, 20, minLineLength, maxLineGap=20)

print(sp[0], sp[1])

for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        if (abs((x2 - x1) / (y2 - y1)) > 3.5):
            continue
        cv2.line(img2, (x1, y1+int(sp[0]*0.5)), (x2, y2+int(sp[0]*0.5)), (0, 255, 0), 3)

#cv2.line(img, (0, 0), (200, 200), (0, 255, 0), 3)

# =============================================================================
cv2.imshow('img',img)
cv2.imshow('img2',img2)
cv2.imshow('gray',gray)
cv2.imshow('eq',eq)
cv2.imshow('blur',blur)
cv2.imshow('imgO',img_out)
cv2.imshow('imgO1',out1)
cv2.imshow('canny',canny)
cv2.imshow('roi', roi)
#cv2.imshow('out2', out2)
print(len(lines))
cv2.waitKey(0)

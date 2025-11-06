import cv2 
import matplotlib.pyplot as plt



img=cv2.imread("test_image.png")

# Converting into grayscale
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# # Applying Gaussian Blur 
# img_blur=cv2.GaussianBlur(img_gray,(7,7),sigmaX=0,sigmaY=0)


# Creating the image histogram for to check intensity distribution

# hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
# plt.plot(hist)
# plt.title("Histogram")
# plt.xlabel("Pixel Intensity")
# plt.ylabel("Frequency")
# plt.show()

# Using Otsu Thresholding to binarize the image

_,thresh_img=cv2.threshold(img_gray,160,255,cv2.THRESH_OTSU)

# Deskewing image




cv2.imshow("display_1",img_gray)
cv2.imshow("display_2",thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
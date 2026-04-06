import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('image1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg',cv2.IMREAD_GRAYSCALE)

# 덧셈
add_img = cv2.add(img1, img2)
# 뺄셈
sub_img = cv2.subtract(img1, img2)
# 곱셈
img1_bin = img1.astype(np.float32) / 255.0
img2_bin = img2.astype(np.float32) / 255.0
mul_img = img1_bin * img2_bin
mul_img = (mul_img * 255).astype(np.uint8)
div_img = cv2.divide(img1, img2, scale=255)

images = [img1, img2, add_img, sub_img, mul_img, div_img]
titles = [
    'Image 1', 'Image 2', 
    'Addition', 'Subtraction', 
    'Multiplication', 'Division'
]
plt.figure(figsize=(12, 10))
for i in range(6):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
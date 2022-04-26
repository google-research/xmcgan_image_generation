import numpy as np
import matplotlib.pyplot as plt

img = np.load('generated_image.npy')
fig = plt.figure(figsize=(8,8))
B, H, W, C = img.shape
print(img[0].shape)
for i in range(B):
    plt.imshow(img[i])
plt.show()

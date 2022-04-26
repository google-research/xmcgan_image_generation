import numpy as np
import matplotlib.pyplot as plt

img = np.load('generated_image.npy')
fig = plt.figure(figsize=(8,8))
B, H, W, C = img.shape

for i in range(B):
    fig.add_subplot(1, 7, i)
    plt.imshow(img[i])
plt.show()

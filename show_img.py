import numpy as np
import matplotlib.pyplot as plt

img = np.load('generated_image.npy')
B, H, W, C = img.shape

fig = plt.figure(figsize=(8,8))

for i in range(B):
    fig.add_subplot(1, 7, i)
    plt.imshow(img[i])
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Create a black image
img = np.zeros((200, 200, 3), dtype=np.uint8)

print("Image shape:", img.shape)
print("Data type:", img.dtype)
print("Memory:", img.nbytes, "bytes")

plt.imshow(img)
plt.title("Blank Black Image")
plt.axis("off")
plt.show()




img = np.zeros((200, 200, 3), dtype=np.uint8)

# Red block
img[10:60, 20:90] = np.full((50, 70, 3), [255, 0, 0], dtype=np.uint8)

# Green block
img[70:130, 20:90] = np.full((60, 70, 3), [0, 255, 0], dtype=np.uint8)

# Blue block
img[140:190, 20:90] = np.full((50, 70, 3), [0, 0, 255], dtype=np.uint8)

# Yellow stripe
img[10:190, 110:130] = [255, 255, 0]

plt.imshow(img)
plt.title("Synthetic Image with RGB Blocks")
plt.axis("off")
plt.show()




# Crop a portion of the image
cropped = img[10:110, 20:120]

# Flip vertically and horizontally
vertical_flip = img[::-1, :, :]
horizontal_flip = img[:, ::-1, :]

# Extract individual channels
red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]

fig, axes = plt.subplots(2, 3, figsize=(10, 7))

axes[0, 0].imshow(cropped)
axes[0, 0].set_title("Cropped")

axes[0, 1].imshow(vertical_flip)
axes[0, 1].set_title("Vertical Flip")

axes[0, 2].imshow(horizontal_flip)
axes[0, 2].set_title("Horizontal Flip")

axes[1, 0].imshow(red_channel, cmap="gray", vmin=0, vmax=255)
axes[1, 0].set_title("Red Channel")

axes[1, 1].imshow(green_channel, cmap="gray", vmin=0, vmax=255)
axes[1, 1].set_title("Green Channel")

axes[1, 2].imshow(blue_channel, cmap="gray", vmin=0, vmax=255)
axes[1, 2].set_title("Blue Channel")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()






# Convert to float for calculations
img_float = img.astype(np.float32)

# Brighten the image
bright = np.clip(img_float + 50, 0, 255).astype(np.uint8)

# Darken the image
dark = np.clip(img_float - 50, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].imshow(img)
axes[0].set_title("Original")

axes[1].imshow(bright)
axes[1].set_title("Brightened")

axes[2].imshow(dark)
axes[2].set_title("Darkened")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()




# Identify pixels whose average RGB intensity exceeds 180
brightness = img.mean(axis=2)
bright_mask = brightness > 180

print("Bright pixels:", np.count_nonzero(bright_mask))
print("Mask shape:", bright_mask.shape)

# Display only the bright pixels
masked_img = np.zeros_like(img)
masked_img[bright_mask] = img[bright_mask]

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].imshow(img)
axes[0].set_title("Original")

axes[1].imshow(masked_img)
axes[1].set_title("Bright Pixels Only")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()




image_uint8 = np.zeros((200, 200, 3), dtype=np.uint8)
image_float64 = np.zeros((200, 200, 3), dtype=np.float64)

print("uint8 memory:", image_uint8.nbytes, "bytes")
print("float64 memory:", image_float64.nbytes, "bytes")

print("uint8 memory:", image_uint8.nbytes / 1024, "KB")
print("float64 memory:", image_float64.nbytes / 1024, "KB")

print("Memory ratio:",
      image_float64.nbytes // image_uint8.nbytes)

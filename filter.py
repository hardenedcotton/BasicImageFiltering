import tkinter as tk
from winsound import Beep
from time import time
from tkinter import filedialog

import numpy as np
from PIL import Image as im


def gaussian_kernel(size=5, sig=1.):
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    assert size % 2 == 1, 'Size should be an odd number.'

    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def pixelate(img, pixel_size):
    width, height = img.size
    num_blocks_x = width // pixel_size
    num_blocks_y = height // pixel_size

    img = img.resize((num_blocks_x, num_blocks_y), resample=im.NEAREST)
    img = img.resize((width, height), resample=im.NEAREST)

    return img


def sliding_function(img, kernel):
    img = np.array(img)

    kernel_size = kernel.shape[1]
    pad_size = kernel_size // 2

    padded = np.pad(img, pad_size, mode='edge')
    padded_h, padded_w = padded.shape

    for h in range(padded_h-kernel_size+1):
        for w in range(padded_w-kernel_size+1):
            img[h, w] = np.sum(
                padded[h:h+kernel_size, w:w+kernel_size] * kernel)

    return im.fromarray(img)


def rgb_sliding(img, kernel):
    img_arr = np.array(img)
    blurred_image = np.zeros_like(img_arr)
    for idx, channel in enumerate((img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2])):
        blurred_channel = sliding_function(channel, kernel)
        blurred_image[:, :, idx] = blurred_channel
    return im.fromarray(blurred_image)


def crt_effect(img, intensity=1):
    hsv_arr = np.array(img.convert('HSV'))
    gray = hsv_arr[:, :, 2]
    img = np.array(img)
    height = gray.shape[0]

    for h in range(height)[:-intensity:intensity*2]:
        for sub_h in range(intensity):
            gray[h+sub_h, :] -= gray[h+sub_h, :]//3

    hsv_arr[:, :, 0] = (hsv_arr[:, :, 0] - 5) % 256
    hsv_arr[:, :, 1] = (hsv_arr[:, :, 1] - hsv_arr[:, :, 1] // 3) % 256
    hsv_arr[:, :, 2] = gray

    return im.fromarray(hsv_arr, 'HSV').convert('RGB')


def invert_image(mat):
    return 255-mat


def dodge_blend(back, front):
    # https://pylessons.com/pencil-sketch
    result = back*255.0 / front
    result[result > 255] = 255
    return result.astype('uint8')


def overlay_image(img, overlay):
    new_img = np.zeros_like(img)
    for i in range(3):
        channel = np.array(img)[:, :, i].astype(np.float64)
        channel *= overlay/255
        channel = channel.astype(np.uint8)
        new_img[:, :, i] = channel
    return im.fromarray(new_img)


def create_shifted_sketch(img, kernel, shift_x=0, shift_y=0) -> np.ndarray:

    hsv_arr = np.array(img.convert('HSV'))
    gray = hsv_arr[:, :, 2]

    blurred_gray = sliding_function(gray, kernel)
    sketch = dodge_blend(gray, blurred_gray)
    shifted_sketch = np.roll(sketch, shift_x, 1)
    shifted_sketch = np.roll(shifted_sketch, shift_y, 0)

    return shifted_sketch


def purpleise(img):
    hsv_arr = np.array(img.convert('HSV'))
    img = np.array(img)

    h = hsv_arr[:, :, 0]
    s = hsv_arr[:, :, 1]
    v = hsv_arr[:, :, 2]
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    threshold = v < 40
    blurred_mask = np.array(sliding_function(
        threshold.astype(np.float64), gaussian_kernel(25, 3)))

    r += (blurred_mask*150).astype(np.uint8)
    b += (blurred_mask*150).astype(np.uint8)

    img[:, :, 0], img[:, :, 1], img[:, :, 2] = r, g, b
    return im.fromarray(img)


def chromatic_aberration(img, rgb_shift: (5, 0, -5)):
    img = np.array(img)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r_shifted = np.roll(r, rgb_shift[0], axis=(0, 1))
    g_shifted = np.roll(g, rgb_shift[1], axis=(0, 1))
    b_shifted = np.roll(b, rgb_shift[2], axis=(0, 1))

    img[:, :, 0], img[:, :, 1], img[:, :, 2] = r_shifted, g_shifted, b_shifted

    return im.fromarray(img)


root = tk.Tk()
root.lift()
root.attributes("-topmost", True)
root.withdraw()

file_path = filedialog.askopenfilename()

print('\nApplication is running, this takes a few minutes.\nYou will hear a beep when done.\n')

img = im.open(file_path)
hsv = img.convert('HSV')

s = time()
kernel = gaussian_kernel(25, 5)

blurred_image = rgb_sliding(img, kernel)
chromo = chromatic_aberration(blurred_image, (10, 0, -10))

purple = purpleise(chromo)

shifted_sketch = create_shifted_sketch(img, kernel)
ghost_image = overlay_image(purple, shifted_sketch)

crt = crt_effect(ghost_image, intensity=3)

result = crt

e = time()
print(f'### Elapsed Time: {e-s:.1f} seconds ###\n')
Beep(2093, 250)

f = filedialog.asksaveasfile(mode='w', defaultextension=".png", filetypes=[
                             (".png", "*.png"), (".jpg", "*.jpg")])
if f:
    result.save(f.name)
    im.open(f.name).show()
    f.close()

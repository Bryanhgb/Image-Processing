#loading and displaying images

import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def load_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]          # drop alpha
    return img


def save_image(img: np.ndarray, path: str) -> None:
    io.imsave(path, img)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    gray = (0.2989 * img[:, :, 0] +
            0.5870 * img[:, :, 1] +
            0.1140 * img[:, :, 2])
    return np.clip(gray, 0, 255).astype(np.uint8)


def detect_type(img: np.ndarray) -> str:
    if img.ndim == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if np.array_equal(r, g) and np.array_equal(g, b):
            img2d = r          # RGB quand  R=G=B → grayscale
        else:
            return "color"
    else:
        img2d = img

    return "binary" if len(np.unique(img2d)) <= 2 else "grayscale"


def show(img: np.ndarray, title: str = "", gray: bool = False) -> None:
    plt.figure()
    plt.imshow(img, cmap="gray" if (gray or img.ndim == 2) else None)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def show_many(images: list, titles: list = None, cols: int = 3) -> None:
    n    = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], cmap="gray" if images[i].ndim == 2 else None)
            ax.axis("off")
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
        else:
            ax.set_visible(False)
    plt.tight_layout()
    plt.show()
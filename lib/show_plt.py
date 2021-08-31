import matplotlib.pyplot as plt
import numpy as np


def show_image_grid(imgs, titles):
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[row_idx, col_idx].set(title=titles[row_idx][col_idx])
            axs[row_idx, col_idx].title.set_size(8)
    plt.tight_layout()
    plt.show()

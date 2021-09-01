import matplotlib.pyplot
import numpy


def show_image_grid(imgs, titles):
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = matplotlib.pyplot.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(numpy.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[row_idx, col_idx].set(title=titles[row_idx][col_idx])
            axs[row_idx, col_idx].title.set_size(8)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

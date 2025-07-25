import torch
import matplotlib.pyplot as plt

def basic_visualize(tensor, title=None, max_images=8, img_size=3):
    """
    visualize images of shape of (C, H, W) or (B, C, H, W)
    if of multiple images, show at most first 8 images
    """

    # check whether tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    else:
        raise TypeError("The input should be torch.Tensor")

    if tensor.dim() == 3:
        imgs = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        imgs = tensor
    else:
        raise ValueError(f"Unsupported dim: {tensor.shape}")

    B, C, H, W = imgs.shape
    num_show = min(B, max_images)

    # convert from [-1, 1] to [0, 1]
    imgs = (imgs + 1) / 2
    imgs = imgs.clamp(0, 1)

    # create a row of image of num_show cols, with each image of size 3 * 3
    fig, axes = plt.subplots(1, num_show, figsize=(img_size * num_show, img_size))

    if num_show == 1:
        axes = [axes]

    for i in range(num_show):
        img = imgs[i]
        if C == 1:
            img = img[0]
            axes[i].imshow(img, cmap='gray')
        else:
            img = img.permute(1, 2, 0)  # C, H, W -> H, W, C
            axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"{title or ''} [{i}]")

    plt.tight_layout()
    plt.show()

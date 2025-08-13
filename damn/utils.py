import torch
import matplotlib.pyplot as plt

def add_gaussian_noise(image: torch.Tensor, eta: float = 0.5) -> torch.Tensor:
    # image: (c, h, w) or (bs, c, h, w)
    single = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single = True
    noise = torch.randn_like(image)
    noised_img = image + eta * noise
    return noised_img.squeeze(0) if single else noised_img


def add_sp_noise(image: torch.Tensor, amount: float = 0.2) -> torch.Tensor:
    # image: (c, h, w) or (bs, c, h, w)
    single = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single = True
    mask = torch.rand_like(image)
    image[mask < amount / 2] = 0.0
    image[(mask >= amount / 2) & (mask < amount)] = 1.0
    return image.squeeze(0) if single else image


def add_block_mask(image: torch.Tensor, size: float = 10.0) -> torch.Tensor:
    # image: (c, h, w) or (bs, c, h, w)
    single = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        single = True

    B, C, H, W = image.shape
    size = int(size)

    # the left-most part to be blocked, of shape(bs, 1)
    x_start = torch.randint(0, H - size, (B, ), device=image.device).view(-1, 1)
    y_start = torch.randint(0, W - size, (B, ), device=image.device).view(-1, 1)

    # the range to be blocked, of shape(1, size)
    x_range = torch.arange(size, device=image.device).view(1, -1)
    y_range = torch.arange(size, device=image.device).view(1, -1)

    # the bolck part, of shape(bs, size)
    x = x_start + x_range
    y = y_start + y_range

    # construct the indices
    X = x[:, :, None].expand(-1, -1, size)  # (bs, size, 1) -> (bs, size, size)
    Y = y[:, None, :].expand(-1, size, -1)  # (bs, 1, size) -> (bs, size, size)
    batch_idx = torch.arange(B, device=image.device).view(-1, 1, 1).expand(-1, size, size)

    image[batch_idx, :, X, Y] = 0.0

    return image.squeeze(0) if single else image


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
    # imgs = (imgs + 1) / 2
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

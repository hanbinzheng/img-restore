import torch




def stopgrad(x: torch.Tensor) -> torch.Tensor:
    # detach tensor from computation graph
    return x.detach()


def adaptive_l2_loss(error: torch.Tensor, gamma: float = 0.5, c: float = 1e-3) -> torch.Tensor:
    """
    Adaptive L2 loss:
    - sg(w) * ||error||_2^2, where w = 1 / (||error||_2^2 + c)^p, p = 1 - γ
    Args:
    - error: u_tgt - u_theta, shape(bs, c, h, w)
    - gamma: power used in original  ||error||_2^{2γ} loss
    - c    : small constant for stability
    Returns:
    - Adaptive L2 loss
    """
    assert 0 <= gamma <= 1, "gamma should be within [0, 1]"
    assert 0 < c, "c should be positive"
    delta_dq = torch.mean(error ** 2, dim=(1, 2, 3))
    p = 1 - gamma
    w = 1 / (delta_dq + c).pow(p)
    loss = (stopgrad(w) * delta_sq).mean()
    return loss

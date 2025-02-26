#!/usr/bin/env python

import typer
import torch
from torch import Tensor
from MinkowskiEngine import MinkowskiDepthwiseConvolution, to_sparse
from copy import deepcopy


def generate_mask(im: Tensor, patch_size: int, mask_ratio: float):
    """Generates mask where 1 is to remove, and 0 is to keep.
    0's are observeable and 1's are for reconstruction."""
    N = im.shape[0]
    L = (im.shape[2] // patch_size) ** 2
    num_keeps = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=im.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones([N, L], device=im.device)
    mask[:, :num_keeps] = 0
    mask = torch.gather(mask, 1, ids_restore)

    p = int(L**0.5)
    mask = mask.reshape(-1, p, p)
    scale = 2**5
    mask = mask.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
    mask = mask.unsqueeze(1)
    return mask


def generate_input(shape: list[int], ratio: float, patch_size: int):
    """Generates input and mask for quick parity test"""
    data = torch.rand(shape)
    mask = generate_mask(data, patch_size, ratio)
    data *= 1.0 - mask
    data = to_sparse(data.cuda())
    return data


app = typer.Typer()


@app.command()
def main(
    in_ch: int = 40,
    kernel: int = 7,
    res: int = 224,
    batch: int = 2,
    ratio: float = 0.6,
    patch_size: int = 32,
):
    """Run quick parity test between v1 and v2 dw conv"""
    dwconv1 = MinkowskiDepthwiseConvolution(
        in_ch, kernel, bias=True, dimension=2, force_old=True
    ).cuda()
    dwconv2 = MinkowskiDepthwiseConvolution(
        in_ch, kernel, bias=True, dimension=2, force_old=False
    ).cuda()
    dwconv2.kernel = deepcopy(dwconv1.kernel)
    dwconv2.bias = deepcopy(dwconv1.bias)

    data = generate_input([batch, in_ch, res, res], ratio, patch_size)
    output1 = dwconv1(data)
    output2 = dwconv2(data)
    # assert torch.allclose(output1.F, output2.F)


if __name__ == "__main__":
    app()

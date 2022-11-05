import torch
import torch.nn as nn


NUM_CLASSES = 1


class IDetectOps(nn.Module):
    "Operations of the detect layer of YOLO v7 that can't be traced with Torchscript."
    stride = [8, 16, 32]  # strides computed during build
    # Gotten from https://github.com/WongKinYiu/yolov7/cfg/deploy/yolov7-tiny-silu.yaml
    # To use a different architecture/model size, replace with the corresponding values
    anchors = [
        [10,13, 16,30, 33,23],  # P3/8
        [30,61, 62,45, 59,119],  # P4/16
        [116,90, 156,198, 373,326]  # P5/32
    ]

    def __init__(self):
        super(IDetectOps, self).__init__()
        nl = len(self.anchors)  # number of detection layers
        self.no = NUM_CLASSES + 5
        self.grid = []#[torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = torch.Tensor(self.anchors).view(nl, 1, -1, 1, 1, 2)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        z = [] 
        for i in range(len(x)):
            bs, _, ny, nx, _ = x[i].shape
            if len(self.grid) <= i:
                self.grid.append(self._make_grid(nx, ny).to(x[i].device))

            y = x[i].sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.view(bs, -1, self.no))

        return (torch.cat(z, 1), x)

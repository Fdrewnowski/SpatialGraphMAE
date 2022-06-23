import torch.nn as nn

from graphmae.models.edcoder import PreModel


class SGMAE(nn.Module):
    def __init__(
            self,
            gmae: PreModel,
            clf: nn.Module):
        super(SGMAE, self).__init__()
        self.gmae = gmae
        self.clf = clf
    
    def forward(self, g, x):
        rep = self.gmae.embed(g, x)
        out = self.clf(g, rep)
        return out
import torch
from torch.nn.functional import cross_entropy


class LargeMarginCosineLossLayer(torch.nn.Module):
    def __init__(self, feature_dim, label_nums, margin):
        super(LargeMarginCosineLossLayer, self).__init__()
        self.params = torch.nn.Parameter(torch.randn((feature_dim, label_nums)))
        self.margin = margin
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, feature_input, labels):
        # shape of feature_input is (bsz, hidden_size)
        # shape of labels is (bsz, label_nums)
        w = torch.norm(self.params, dim=0).unsqueeze(0)
        x = torch.norm(feature_input, dim=1).unsqueeze(-1)
        norm_v = torch.mm(x, w)
        # print(norm_v.shape)
        matmul = torch.mm(feature_input, self.params)
        # print(matmul.shape)
        normalized = matmul / norm_v
        predict = torch.argmax(normalized, dim=1)
        # 分类结果从normalized推出
        # 以下是计算loss

        logits = (normalized - self.margin)
        logits = self.softmax(logits)
        loss = cross_entropy(logits, labels)
        return predict, loss


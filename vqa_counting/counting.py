import torch
import visualisation
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Counter(nn.Module):
    """
    Core net proposed by the author.

    [1]: Yan Zhang, Jonathon Hare, Adam Prügel-Bennett: Learning to Count Objects in Natural Images for Visual Question Answering.
    https://openreview.net/forum?id=B12Js_yRb
    """

    def __init__(self, objects_num, picewise_num=16):
        super().__init__()
        self.objects_num = objects_num
        self.f = nn.ModuleList([PiecewiseLin(picewise_num) for _ in range(picewise_num)])

    def forward(self, boxes, attention):
        """
        :param boxes: The locations of boxes, dimension:(n, 4, m), n is batch number, m is object number.
        :param attention: The attention weight. dimension:(n,m),  n is batch number, m is object number.Range from 0 to 1.
        :return: count feature vector
        """
        # Equation 1, convert the attention vector into matrix.A = a & a.T.
        relevancy = self.outer_product(attention)

        # Equation 2, calculate the distance matrix
        distance = 1 - self.iou(boxes, boxes)

        # Equation 3, strip the intra-object
        A = self.f[0](relevancy) * self.f[1](distance)

        # inter-object deduplication
        # Used for equation 5.
        dedup_score = self.f[3](relevancy) * self.f[4](distance)
        # Calculate the denominator of equation 5. Sum of all the similar.
        dedup_per_entry, dedup_per_row = self.deduplicate(dedup_score, attention)
        # Equation 5
        score = A / dedup_per_entry

        # visualisation usage(The author did not calculate the C matrix in the paper)
        s = 1 / dedup_per_row
        C = A * self.outer_product(s) + torch.diag(s * self.f[0](attention * attention))

        # aggregate the score
        # can skip putting this on the diagonal since we're just summing over it anyway
        # Equation 6, right part of the correction
        correction = self.f[0](attention * attention) / dedup_per_row
        score = score.sum(dim=2).sum(dim=1, keepdim=True) + correction.sum(dim=1, keepdim=True)
        # Equation 7, square the score and calculate the final score
        score = (score + 1e-20).sqrt()
        # output onehot
        one_hot = self.to_one_hot(score)

        #  output the confidence, equation 9 + equation 10
        att_conf = (self.f[5](attention) - 0.5).abs()
        dist_conf = (self.f[6](distance) - 0.5).abs()
        # Equation 11, average the two confidences and sum them up
        conf = self.f[7](att_conf.mean(dim=1, keepdim=True) + dist_conf.mean(dim=2).mean(dim=1, keepdim=True))
        return one_hot * conf

    def deduplicate(self, dedup_score, att):
        # Equation 4, calculate the similarity between two proposals
        att_diff = self.outer_diff(att)
        score_diff = self.outer_diff(dedup_score)
        sim = self.f[2](1 - score_diff).prod(dim=1) * self.f[2](1 - att_diff)
        # similarity for each row
        row_sims = sim.sum(dim=2)
        # similarity for each entry
        all_sims = self.outer_product(row_sims)
        return all_sims, row_sims

    def to_one_hot(self, scores):
        """
        If it's a integer, put the 1 to the index of the zeros vector.
        If it's a real number, then divide the integer part and the real number part at first
        :param scores: a vector
        :return:
        """
        # sanity check, I don't think this ever does anything (it certainly shouldn't)
        scores = scores.clamp(min=0, max=self.objects_num)
        # compute only on the support
        i = scores.long().data
        f = scores.frac()
        # target_l is the one-hot if the score is rounded down
        # target_r is the one-hot if the score is rounded up
        target_l = scores.data.new(i.size(0), self.objects_num + 1).fill_(0)
        target_r = scores.data.new(i.size(0), self.objects_num + 1).fill_(0)

        target_l.scatter_(dim=1, index=i.clamp(max=self.objects_num), value=1)
        target_r.scatter_(dim=1, index=(i + 1).clamp(max=self.objects_num), value=1)
        # interpolate between these with the fractional part of the score
        return (1 - f) * Variable(target_l) + f * Variable(target_r)

    def outer(self, x):
        size = tuple(x.size()) + (x.size()[-1],)
        a = x.unsqueeze(dim=-1).expand(*size)
        b = x.unsqueeze(dim=-2).expand(*size)
        return a, b

    def outer_product(self, x):
        a, b = self.outer(x)
        return a * b

    def outer_diff(self, x):
        # like outer products, except taking the absolute difference instead
        # Y_ij = | x_i - x_j |
        a, b = self.outer(x)
        return (a - b).abs()

    def iou(self, a, b):
        # this is just the usual way to IoU from bounding boxes
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(2).expand_as(inter)
        area_b = self.area(b).unsqueeze(1).expand_as(inter)
        return inter / (area_a + area_b - inter + 1e-12)

    def area(self, box):
        # calculate the area of the given boxes
        x = (box[:, 2, :] - box[:, 0, :]).clamp(min=0)
        y = (box[:, 3, :] - box[:, 1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        # Calculate the IOU of the given boxes
        size = (a.size(0), 2, a.size(2), b.size(2))
        min_point = torch.max(
            a[:, :2, :].unsqueeze(dim=3).expand(*size),
            b[:, :2, :].unsqueeze(dim=2).expand(*size),
        )
        max_point = torch.min(
            a[:, 2:, :].unsqueeze(dim=3).expand(*size),
            b[:, 2:, :].unsqueeze(dim=2).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[:, 0, :, :] * inter[:, 1, :, :]
        return area


class PiecewiseLin(nn.Module):
    """
    Learn the parameter of piecewise linear function,
    used to remove the intra-object edges, the implementation of equation 12
    """

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))
        # the first weight here is always 0 with a 0 gradient
        self.weight.data[0] = 0

    def forward(self, x):
        # all weights are positive -> function is monotonically increasing
        w = self.weight.abs()
        # make weights sum to one -> f(1) = 1
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        # keep cumulative sum for O(1) time complexity
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1,) + tuple(x.size()))
        w = w.expand_as(csum)

        # figure out which part of the function the input lies on
        y = self.n * x.unsqueeze(0)
        idx = Variable(y.long().data)
        f = y.frac()

        # contribution of the linear parts left of the input
        x = csum.gather(0, idx.clamp(max=self.n))
        # contribution within the linear segment the input falls into
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)

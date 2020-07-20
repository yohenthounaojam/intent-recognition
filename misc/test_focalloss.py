import torch
import numpy as np
from collections import Counter
import torch.nn as nn

from focalloss import FocalLoss
from focalloss1 import FocalLoss1
from focalloss2 import FocalLoss2

clss = np.random.randint(0, 11, 5000)
clss_counter = Counter(clss)
weights = []
for key in range(11):
  weights.append(clss_counter[key])

pred_out = torch.randn(5000, 11)
clss = torch.tensor(clss).long()

print(weights)


"""
fl = FocalLoss(gamma=0, alpha=weights)
print(fl(pred_out, clss).item())
print("*"*80)

fl2 = FocalLoss(gamma=0, alpha=weights, size_average=False)
print(fl2(pred_out, clss).item())
print("*"*80)
fl1_1 = FocalLoss1(gamma=2, alpha=weights)
print(fl1_1(pred_out, clss).item())
print("*"*80)

fl1_2 = FocalLoss1(gamma=2, alpha=weights, reduce=False)
print(fl1_2(pred_out, clss).item())
print("*"*80)

fl2_1 = FocalLoss2(weight=torch.tensor(weights).float(), gamma=2.0)
print(fl2_1(pred_out, clss))
print(fl2_1(pred_out, clss).shape)
print("*"*80)

"""
fl2_2 = FocalLoss2(weight=torch.tensor(weights).float(), gamma=0.0, reduction=
                   "mean")
print(fl2_2(pred_out, clss).item())
print("*"*80)

fl_ce = nn.CrossEntropyLoss(weight=torch.tensor(weights).float())
print(fl_ce(pred_out, clss))
print("*"*80)

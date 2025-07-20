# import torch
# import torch.nn as nn
# class MMD_loss(nn.Module):
# 	def __init__(self, kernel_mul = 2.0, kernel_num = 5):
# 		super(MMD_loss, self).__init__()
# 		self.kernel_num = kernel_num
# 		self.kernel_mul = kernel_mul
# 		self.fix_sigma = None
# 		return
# 	def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
# 		n_samples = int(source.size()[0])+int(target.size()[0])
# 		total = torch.cat([source, target], dim=0).to(torch.float16)
#
# 		total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
# 		total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
# 		L2_distance = ((total0-total1)**2).sum(2)
# 		if fix_sigma:
# 			bandwidth = fix_sigma
# 		else:
# 			bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
# 		bandwidth /= kernel_mul ** (kernel_num // 2)
# 		bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
# 		kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
# 		return sum(kernel_val)
#
# 	def forward(self, source, target):
# 		batch_size = int(source.size()[0])
# 		kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
# 		XX = kernels[:batch_size, :batch_size]
# 		YY = kernels[batch_size:, batch_size:]
# 		XY = kernels[:batch_size, batch_size:]
# 		YX = kernels[batch_size:, :batch_size]
# 		loss = torch.mean(XX + YY - XY -YX)
# 		return loss
#
# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-efficient MMD implementation in JAX."""

import torch

# The bandwidth parameter for the Gaussian RBF kernel. See the paper for more
# details.
_SIGMA = 10
# The following is used to make the metric more human readable. See the paper
# for more details.
_SCALE = 1000


def mmd(x, y):
    """Memory-efficient MMD implementation in JAX.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Note that the first invocation of this function will be considerably slow due
    to JAX JIT compilation.

    Args:
      x: The first set of embeddings of shape (n, embedding_dim).
      y: The second set of embeddings of shape (n, embedding_dim).

    Returns:
      The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * _SIGMA**2)
    k_xx = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, x.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0)))
    )
    k_xy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(x, y.T) + torch.unsqueeze(x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )
    k_yy = torch.mean(
        torch.exp(-gamma * (-2 * torch.matmul(y, y.T) + torch.unsqueeze(y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0)))
    )

    return _SCALE * (k_xx + k_yy - 2 * k_xy)
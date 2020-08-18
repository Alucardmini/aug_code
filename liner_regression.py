# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/8/15 8:46 AM'

import numpy as np

batch_size, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(batch_size, D_in)
y = np.random.randn(batch_size, D_out)
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
learning_rate = 1e-6

for i in range(500):
   h = x.dot(w1)  # BH
   relu_h = np.maximum(h, 0)
   y_pred = relu_h.dot(w2)  # BO
   loss = np.square(y_pred - y).sum()
   print(i, loss)

   loss_grad_w2 = relu_h.T.dot(2.0*(y_pred - y))

   loss_grad_relu = 2.0*(y_pred - y).dot(w2.T)
   loss_grad_relu[h < 0] = 0

   loss_grad_w1 = x.T.dot(loss_grad_relu)

   w1 -= learning_rate * loss_grad_w1
   w2 -= learning_rate * loss_grad_w2










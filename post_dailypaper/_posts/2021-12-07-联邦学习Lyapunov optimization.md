---
layout: post 
title: 联邦学习 Lyapunov optimization  
categories:  PaperRead   
---   

## An Efficiency-boosting Client Selection Scheme for Federated Learning with Fairness Guarantee 一种保证公平性的联邦学习效率提高的客户选择方案

在本文中，我们将保证公平性的客户端选择问题建模为Lyapunov优化问题，然后提出了基于C^2MAB的方法来估计每个客户端与服务器之间的模型交换时间，在此基础上我们设计了一个保证公平性的算法RBCS-F来解决问题。RBCS-F的遗憾是严格限定在有限常数范围内，证明了其理论可行性。除了理论结果，更多的实证数据可以从我们在公共数据集上的真实训练实验中得到。

这促使我们开发一种算法，在训练效率和公平性之间取得良好的平衡。此外，算法应该足够智能，能够根据客户的声誉(或历史表现)预测他们的训练时间，而不是假设它是已知的先验。


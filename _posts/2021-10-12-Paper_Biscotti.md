---
layout: post 
title: Biscotti-使用区块链和加密原语来协调P2P的多方机器学习    
categories:  PaperRead   
---   

## 摘要
> 联邦学习是目前支持安全多方机器学习(ML)的最先进技术:数据在所有者的设备上维护，并通过安全协议聚合模型的更新。  

> 然而，此过程**假设有一个可信任的集中式基础设施进行协调**，并且客户必须相信中央服务不会使用客户数据的副产品。  

> 提出了Biscotti:一种完全去中心化的点对点(P2P)方法来实现多方ML，它使用区块链和加密原语来协调对等客户端之间的隐私保护ML过程。  

> 评估表明，Biscotti是可扩展的、容错的，并且可以防御已知的攻击。  

## 介绍
> 对手可以通过中毒攻击来攻击共享模型，在这种攻击中，对手提供对抗性更新到共享模型参数。  

> 还可以攻击其他客户的隐私:在信息泄露攻击中，对手冒充诚实的数据提供者，通过观察目标的共享模型更新.  

> 在之前的工作中，通过集中异常检测、差异隐私或安全聚合，中毒攻击和信息泄露攻击都已单独进行了防御, 但是目前**还不存在同时解决这两种威胁的私有和分散解决方案**。此外，**这些方法不适用于缺乏可信中央权威的去中心化上下文中**.  

> 由于ML不需要强共识或一致性来收敛，传统的强共识协议，如拜占庭容错(BFT)协议对机器学习工作负载有过度的限制。分布式账本(区块链)已经成为一个更合适的系统，以促进私有的、可验证的、众包计算。  

> 联邦学习依赖可信的聚合器, 不适合点对点(P2P) ML设置. 提出了联邦证明(PoF)，这是一种1层区块链共识协议，它结合了联邦学习的最先进的防御技术，使它们适用于去中心化的P2P设置。  

> 结合**可验证随机函数VRFs**使用基于PoF的一致哈希来为节点选择关键角色，这些节点将帮助协调模型更新的隐私和安全性。  

> Biscotti通过Multi-Krum防御防止P2P毒害模型，通过差异私有噪声提供隐私，并使用Shamir秘密进行安全聚合.  

> 在Azure上评估了Biscotti，并考虑了它的性能、可伸缩性、移动容忍度以及抵抗不同攻击的能力。  

## 挑战和解决办法
1. 女巫攻击: 使用可验证随机函数和联邦证明的一致哈希协议.  
2. 中毒攻击: Biscotti使用名为Multi-Krum的拜占庭容错聚合方案验证SGD更新。Multi-Krum只是Biscotti可以使用的几种算法之一, 其他包括中位数/修剪平均值和Bulyan. **Multi-Krum拒绝与大多数更新方向有很大差异的模型更新。**  
3. 信息泄露攻击: 通过两种方式防止了此类攻击, 使用最新的块哈希来选择验证器，确保恶意对等体无法确定地选择自己来验证受害者的梯度; 噪声点向验证点发送差异私有更新: 在向验证点发送梯度之前，预先提交的差异私有噪声被添加到更新中，掩盖了节点的梯度.  
4. 差异隐私带来的的效用损失: 隐私和效用之间的权衡.  

## 假设和威胁模型
1. 联邦证明(POF): 是一种基于区块链共识机制POS的联邦学习共识. POF将利益定义为衡量一个人对系统的贡献。节点通过提供有益的模式更新或促进共识过程来获得股权。因此，节点在训练中积累的利益与他们对被训练模型的贡献成比例。  
2. 区块链拓扑: 在拓扑结构中，每个点都连接到其他点的某个子集，从而允许基于泛洪的块传播最终到达所有点。
3. 机器学习: 假设P2P训练所需的所有信息在第一个块中被传播到所有的节点，包括模型、超参数、优化算法和学习目标. 使用随机梯度下降(SGD)作为优化算法.  
4. 攻击者假设: 假设对手可以控制女巫攻击中的多个节点，但控制的股份不超过总股份的30%。假设对手不能人为地增加他们在系统中的股份，除非提供通过Multi-Krum的有效更新. Biscotti兼容SGD更新的任何其他聚合方法，包括可能处理后门和梯度上升攻击的未来方法。

## Biscotti设计
![20211013111646](https://cdn.jsdelivr.net/gh/kexve/img/blogImg20211013111646.png)

### 训练初始化
Biscotti使用第一块（生成块）的信息初始化训练过程
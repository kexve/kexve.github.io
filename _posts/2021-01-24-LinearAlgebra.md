---      
layout: post      
title: MIT-线性代数      
categories: Algebra      
extMath: ture
---   

## 方程组的列图像（column picture）
### 思路
三维矩阵，行图像是三个平面；列图像是
$$
x * \begin{pmatrix} 2 \\ -1\\ 0 \end{pmatrix} + y * \begin{pmatrix} -1 \\ 2\\-3 \end{pmatrix} + z * \begin{pmatrix} 0\\ -1\\ 4\end{pmatrix} = \begin{pmatrix} 0\\-1\\4 \end{pmatrix}
$$
(a linear combination，由三个向量合成一个向量)

### 问题
1. can i solve Ax=b for every b?
2. do the linear combinations of the columns fill 3-d space?

## 矩阵乘法
两种矩阵乘法：  
1. $$
\begin{pmatrix} 2 & 5 \\ 1 & 3 \end{pmatrix} * \begin{pmatrix} 1\\2 \end{pmatrix} = 1 * \begin{pmatrix} 2\\1 \end{pmatrix} + 2 * \begin{pmatrix} 5\\3 \end{pmatrix} = \begin{pmatrix} 12\\9 \end{pmatrix}
$$  
2. $$
\begin{pmatrix} 1 & 0 & 0 \end{pmatrix} * \begin{pmatrix} 1& 2& 1\\3& 8 &1\\ 0 & 4 &1 \end{pmatrix} = 1* \begin{pmatrix} 1 & 2 & 1 \end{pmatrix} + 0 * \begin{pmatrix} 3 & 8 & 1 \end{pmatrix} + 0 * \begin{pmatrix} 0 & 4 & 1 \end{pmatrix}
$$

# Tutorial on Variational Autoencoders

## Abstract

在短短的三年中，变分自动编码器（VAE）成为复杂分布的无监督学习最流行的方法之一。VAE之所以吸引人，是因为它们建立在标准函数逼近器（神经网络）之上，并且可以进行随机梯度下降训练。VAE已经显示出产生多种复杂数据的希望，包括手写数字[1,2]，人脸[1,3,4]，房屋编号[5,6]，CIFAR图像[6]，场景的物理模型[4]，分段[7]和根据静态图片[8]预测未来。本教程介绍了VAE背后的含义，解释了它们背后的数学原理，并描述了一些经验行为。假定没有先验的变分贝叶斯方法知识。

## 1 Introduction

“生成建模”是机器学习的一个广阔领域，它处理在某些潜在的高维空间$\mathcal{X}$中的数据点$X$上定义的分布模型$P(X)$。例如，图像是一种流行的数据，我们可以为其创建生成模型。每个“数据点”（图像）都具有成千上万个维度（像素），生成模型的工作就是以某种方式捕获像素之间的依赖关系，例如，附近的像素具有相似的颜色并将其组织为对象。确切地讲，“捕获”这些依赖项的含义确切取决于我们要对模型执行的操作。一种直接的生成模型使我们可以简单地从数值上计算$P(X)$。在图像的例子中，看起来像真实图像的X值应具有较高的概率，而看起来像随机噪声的图像的X值应具有较低的概率。但是，像这样的模型并不一定有用：知道一张不可能的图像并不能帮助我们合成一张可能的图像。

取而代之的是，人们经常在乎产生更多与数据库中已经存在的相似但并不完全相同的示例。我们可以从原始图像的数据库开始，然后合成新的，看不见的图像。我们可能会采用植物等3D模型的数据库，并产生更多的3D模型以填充视频游戏中的森林。我们可以接受手写文本并尝试产生更多手写文本。这样的工具实际上可能对图形设计师有用。我们可以通过说说我们根据一些未知的分布$P_{gt}(X)$来获得示例$X$的分布来对这种设置进行形式化，我们的目标是学习可以从中采样的模型$P$，从而使$P$尽可能与$P_{gt}$相似。

在机器学习社区中，训练这种类型的模型一直是一个长期存在的问题，并且从传统上讲，大多数方法都具有三个严重缺陷之一。首先，他们可能需要对数据的结构做出强有力的假设。其次，它们可能会进行严格的近似，导致模型不理想。第三，他们可能依赖于计算上昂贵的推理程序，例如Markov Chain Monte Carlo。最近，一些工作在通过反向传播训练作为强大的函数逼近器的神经网络方面取得了巨大的进步[9]。这些进展产生了有希望的框架，可以使用基于反向传播的函数逼近器来构建生成模型。

此类框架中最流行的一种是变分自动编码器[1,3]，它是本教程的主题。该模型的假设很弱，并且通过反向传播快速训练。VAE确实可以做出近似，但是这种近似所引入的误差可以说是小量的高容量模型。这些特征促使其迅速流行。

本教程旨在作为VAE的非正式介绍，而不是有关它们的正式科学论文。它适用于可能会使用生成模型但在VAE所基于的变式贝叶斯方法和“最小描述长度”编码模型方面没有强大背景的人们。本教程的开始是作为UC Berkeley和Carnegie Mellon的计算机视觉阅读小组的演示文稿，因此偏向于视觉读者。感谢改进的建议。

### 1.1 Preliminaries: Latent Variable Models

在训练生成模型时，维度之间的依赖性越复杂，模型训练就越困难。例如，生成手写字符图像的问题。为简单起见，我们只关心对数字0-9建模。如果字符的左半部分包含5的左半部分，则右半部分不能包含0的左半部分，否则该字符将很明显看起来不像任何实数位。从直觉上讲，如果模型在将值分配给任何特定像素之前先确定要生成哪个字符，则将很有帮助。这种决策正式称为潜在变量。也就是说，在我们的模型绘制任何内容之前，它首先从集合$[0,\ldots,9]$中随机采样一个数字值$z$，然后确保所有笔画均与该字符匹配。$z$被称为“潜在”，因为仅给定一个由模型产生的字符，我们不一定知道潜在变量的哪些设置会生成字符。我们将需要使用计算机视觉之类的工具来推断它。

在可以说我们的模型代表我们的数据集之前，我们需要确保对于数据集中的每个数据点$X$，都有一个（多个）潜在变量设置，这些潜在变量会使模型生成与$X$非常相似的东西。形式上说，我们在高维空间$\mathcal{Z}$中有一个潜在变量向量$z$，我们可以根据在$\mathcal{Z}$上定义的某些概率密度函数（PDF）$P(z)$轻松进行采样。然后说我们有一个确定性函数族$f(z;\theta)$，参数为某个空间$\Theta$中的向量$\theta$，其中$f:\mathcal{Z}\times\Theta\rarr\mathcal{X}$。$f$是确定性的，但是如果$z$随机且$\theta$是固定的，则$f(z;\theta)$是空间$\mathcal{X}$中的随机变量。我们希望优化$\theta$，以便可以从$P(z)$采样$z$，并且有很大概率使得$f(z;\theta)$就像我们数据集中的$X$一样。

为了使这一概念在数学上更加精确，我们的目标是在整个生成过程中最大程度地提高训练集中每个$X$的概率，具体取决于：
$$
P(X)=\int P(X|z;\theta)P(z)dz \tag{1}
$$

在这里，$f(z;\theta)$被分布$P(X|z;\theta)$代替，这使我们能够通过使用总概率定律来明确$z$上$X$上的依赖。该框架背后的直觉（称为“最大似然”）是，如果模型可能会产生训练集样本，那么它也可能会产生相似的样本，而不太可能产生不同的样本。在VAE中，此输出分布的选择通常是高斯分布，即$P(X|z;\theta)=N(X|f(z;\theta),\sigma^2*I)$。也就是说，它的均值为$f(z;\theta)$和协方差等于单位矩阵$I$乘以某个标量$\sigma$（这是一个超参数）。这种替换对于形式化一些$z$需要产生仅类似于$X$的样本的直觉是必要的。一般而言，尤其是在训练早期，我们的模型将不会产生与任何特定$X$相同的输出。通过具有高斯分布，我们可以使用梯度下降（或任何其他优化技术）通过使$f(z;\theta)$使一些$z$逼近某点$X$来增长$P(X)$，即在生成模型下逐渐使训练数据变得更相似。如果$P(X|z)$是Dirac delta函数，那将是不可能的，就像我们确定性用$X=f(z;\theta)$地那样！注意，输出分布不需要一定为高斯分布：例如，如果$X$是二进制的，则$P(X|z)$可能是由$f(z;\theta)$参数化的Bernoulli分布。重要的性质只是$P(X|z)$是可以计算的，并且在$\theta$中连续。从这里开始，我们将省略$f(z;\theta)$中的$\theta$以避免混乱。

## 2 Variational Autoencoders

实际上，VAE的数学基础与经典自动编码器（例如，稀疏自动编码器[10,11]或降噪自动编码器[12,13]。VAE根据图1中所示的模型，使方程1近似最大化。之所以将它们称为“自动编码器”，是因为从该设置派生的最终训练目标确实具有编码器和解码器，并且类似于传统的自动编码器。与稀疏自动编码器不同，通常没有类似于稀疏惩罚的调整参数。与稀疏和去噪自动编码器不同，我们可以直接从$P(X)$进行采样（不执行Markov Chain Monte Carlo，如[14]中所示）。

为了解决等式1，VAE必须解决两个问题：如何定义潜变量$z$（即，确定它们代表什么信息），以及如何处理沿$z$的积分。VAE为这两者提供了明确的答案。

首先，我们如何选择潜在变量$z$以捕获潜在信息？回到我们的数字示例，该模型在开始绘制数字之前需要做出的“潜在”决定实际上相当复杂。它不仅需要选择数字，还需要选择数字绘制的角度，笔划宽度以及抽象的样式属性。更糟糕的是，这些属性可能是相关的：如果写得更快，则可能会产生更大角度的数字，这也可能导致笔画变细。理想情况下，我们要避免手动确定$z$编码的每个维度的信息（尽管对于某些维度我们可能希望手动指定[4]）。我们还希望避免明确描述$z$的维之间的依赖关系（即潜在结构）。VAE采取了一种不寻常的方法来解决这个问题：他们假设对$z$的维没有简单的解释，而是断言可以从简单的分布即$N(0,I)$提取$z$的样本，其中$I$是单位矩阵。这怎么可能？关键是要注意，可以通过获取一组$d$个正态分布的变量并通过足够复杂的函数将它们映射来生成任何的$d$维的分布。例如，假设我们要构造一个二维随机变量，其值位于一个环上。如果$z$是2D且满足正态分布，则$g(z)=z/10+z/\|z\|$大致呈环形，如图2所示。因此，提供了功能强大的函数逼近器，我们可以简单地学习一个函数，该函数将我们独立正态分布的$z$值映射到模型可能需要的任何潜在变量，然后将这些潜在变量映射到$X$。实际上，回想一下$P(X|z;\theta)=N(X|f(z;\theta),\sigma^2*I)$。如果$f(z;\theta)$是一个多层神经网络，那么我们可以想象该网络使用其前几层将正态分布的$z$精确映射到具有正确的统计数据的隐藏值(例如数字标识，笔划粗细，角度等)。然后，它可以使用以后的层将这些隐藏值映射到完全渲染的数字。通常，我们不必担心要确保存在隐藏结构。如果这种隐藏结构有助于模型准确地复制训练集（即最大程度地提高训练集的可能性），则网络将在某层学习该结构。

现在剩下的就是最大化等式1，其中$P(z)=N(z|0,I)$。正如机器学习中常见的那样，如果我们可以找到一个可计算的$P(X)$公式，那么我们可以采用该公式的梯度，然后我们可以使用随机梯度上升来优化模型。实际上，从概念上讲，直接近似计算$P(X)$是直截了当的：我们首先对$z$值进行大量采样$\{z_1,\ldots,z_n\}$，然后计算出$P(X)\approx\frac{1}{n}\sum_iP(X|z_i)$。这里的问题是，在高维空间中，在我们需要精确估计$P(X)$之前，可能需要将$n$做得非常大。要了解原因，请考虑我们的手写数字示例。假设我们的数字数据点存储在像素空间中，在如图3所示的$28\times28$图像。由于$P(X|z)$是各向异性高斯，因此$X$的负对数概率是与$f(z)$和$X$之间的平方欧几里德距离成比例的。假设图3（a）是我们试图为$P(X)$找到的目标$(X)$。产生图3（b）所示图像的模型可能是一个坏模型，因为该数字与2不太相似。因此，我们应该设置高斯分布的超参数$\sigma$，以使这种错误的数字不会对$P(X)$产生影响。另一方面，产生图3（c）的模型（与$X$相同，但向右下移了半个像素）可能是一个很好的模型，我们希望这个样本对$P(X)$有贡献。不幸的是，但是，我们不能同时使用这两种方法：$X$和图3（c）之间的平方距离是0.2693（假设像素介于0和1之间），而$X$和图3（b）之间的平方距离仅为.0387。这里的教训是，为了拒绝如图3（b）所示的样本，我们需要将$\sigma$设置得非常小，以使模型需要生成比图3（c）更显著像$X$的东西！即使我们的模型是一个精确的数字生成器，在生成与图3（a）中的数字足够相似的2之前，我们可能仍需要对数千个数字进行采样。我们也许可以通过使用更好的相似性指标来解决此问题，但实际上，很难对这些复杂的领域（例如视觉）进行工程设计，并且如果没有表明哪些数据点彼此相似的标签也很难进行训练。取而代之的是，VAE会更改采样过程以使其更快，而无需更改相似性度量。

### 2.1 Setting up the objective

在使用采样来计算式1时，我们可以采取捷径吗？实际上，对于大多数$z$，$P(X|z)$几乎为零，因此对我们对$P(X)$的估计几乎没有贡献。变分自动编码器背后的关键思想是尝试对可能已产生$X$的$z$值进行采样，并仅从这些值中计算出$P(X)$。这意味着我们需要一个新的函数$Q(z|X)$，该函数可以采用$X$的值，并给我们一个可能产生$X$的$z$值分布。希望可能在$Q$下的$z$值的空间比在先验$P(z)$下的$z$值的全空间小得多。例如，这使我们可以相对轻松地计算$E_{z\sim Q}P(X|z)$。但是，如果从PDF为$Q(z)$的任意分布(不是$N(0,I)$)中采样，那么这如何帮助我们优化$P(X)$？我们要做的第一件事是将$E_{z\sim Q}P(X|z)$和$P(X)$关联起来。我们稍后会看到$Q$的来源。

$E_{z\sim Q}P(X|z)$与$P(X)$之间的关系是变分贝叶斯方法的基石之一。我们从$P(z|X)$和$Q(z)$之间的Kullback-Leibler散度（KL散度或$\mathcal{D}$）的定义开始，对于一些任意$Q$（可能取决于$X$，也可能不取决于$X$）：
$$
\mathcal{D}[Q(z) \| P(z | X)]=E_{z \sim Q}[\log Q(z)-\log P(z | X)] \tag{2}
$$
通过将贝叶斯规则应用于$P(z|X)$，我们可以将$P(X)$和$P(X|z)$都包含到该方程中：
$$
\mathcal{D}[Q(z) \| P(z | X)]=E_{z \sim Q}[\log Q(z)-\log P(X | z)-\log P(z)]+\log P(X) \tag{3}
$$
在这里，$\log P(X)$因为不依赖于$z$而不用取期望。两边取反，重新排列以及将$E_{z\sim Q}P(X|z)$的一项收缩到KL散度，将得出：
$$
\log P(X)-\mathcal{D}[Q(z) \| P(z | X)]=E_{z \sim Q}[\log P(X|z)-\mathcal{D}[Q(z) \| P(z)]] \tag{4}
$$
请注意，$X$是固定的，并且$Q$可以是任何分布，而不仅仅是将$X$映射到可以产生良好的$X$的$z$的分布。由于我们对推断$P(X)$感兴趣，因此构造一个依赖于$X$的$Q$是有意义的，尤其是构造一个使$\mathcal{D}[Q(z) \| P(z|X)]$小的$Q$：
$$
\log P(X)-\mathcal{D}[Q(z|X) \| P(z|X)]=E_{z \sim Q}[\log P(X|z)-\mathcal{D}[Q(z|X) \| P(z)]] \tag{5}
$$
这个方程式是变分自动编码器的核心，值得花些时间思考它所说的。用两句话，左侧有我们要最大化的项：$\log P(X)$（加上一个误差项，这使得$Q$产生可以再现给定的$X$的$z$；如果$Q$具有高容量，则该项将变小）。在给定正确的$Q$选择的情况下，右侧是我们可以通过随机梯度下降来优化的东西（尽管可能尚不明显）。请注意，该框架（尤其是等式5的右侧）突然采取了一种看起来像自动编码器的形式，因为$Q$是“编码” $X$到$z$，而$P$是“解码”以重构$X$。我们将在以后更详细地探讨这种联系。

现在来详细介绍等式5。从左边开始，我们使$\log P(X)$最大化，同时使$\mathcal{D}[Q(z|X) \| P(z|X)]$最小化。$P(z|X)$是我们无法进行分析计算：它描述了在图1中的模型下可能会产生样本$X$的$z$值。但是，左边的第二项是将$Q(z|X)$拉到匹配$P(z|X)$。假设我们对$Q(z|X)$使用任意大容量模型，然后希望$Q(z|x)$实际匹配$P(z|X)$，在这种情况下，该KL散度项将为零，我们将直接优化$\log P(X)$。作为额外的奖励，我们使难处理的$P(z|X)$易于处理：我们可以使用$Q(z|x)$进行计算。

### 2.2 Optimizing the objective

那么，如何在等式5的右侧执行随机梯度下降呢？首先，我们需要更加具体地说明$Q(z|X)$的形式。通常的选择是说$Q(z|X)=N(z|\mu(X;\theta),\Sigma(X;\theta))$，其中$\mu$和$\Sigma$是任意具有确定参数$\theta$的确定性函数，$\theta$可以从数据中学习（我们将在后面的公式中省略$\theta$）。在实践中，$\mu$和$\Sigma$又通过神经网络实现，并且$\Sigma$被约束为对角矩阵。这种选择的优势是计算性，因为它们使我们清楚了如何计算右手边。现在，最后一项$\mathcal{D}[Q(z|X) \| P(z|X)]$-是两个多元高斯分布之间的KL-散度，可以将其闭式计算为：
$$
\begin{aligned}
\mathcal{D}[\mathcal{N}(\mu_{0}, \Sigma_{0}) &\| \mathcal{N}(\mu_{1}, \Sigma_{1})] =\\ &
\frac{1}{2}(\operatorname{tr}(\Sigma_{1}^{-1} \Sigma_{0})+(\mu_{1}-\mu_{0})^{\top} \Sigma_{1}^{-1}(\mu_{1}-\mu_{0})-k+\log (\frac{\operatorname{det} \Sigma_{1}}{\operatorname{det} \Sigma_{0}}))
\end{aligned} \tag{6}
$$
其中$k$是分布的维数。在我们的情况下，这简化为：
$$
\begin{aligned}
\mathcal{D}[\mathcal{N}(\mu(X), \Sigma(X)) &\| \mathcal{N}(0, I)] =\\ &
\frac{1}{2}(\operatorname{tr}(\Sigma(X))+(\mu(X))^{\top}(\mu(X))-k-\log (\operatorname{det} \Sigma(X)))
\end{aligned} \tag{7}
$$
等式5右侧的第一项比较棘手。我们可以使用采样来估计$E_{z \sim Q}[\log P(X|z)$，但是要获得良好的估计，则需要传递许多$z$的样本通过$f$，这将很昂贵。因此，作为一个标准的随机梯度下降，我们采用$z$的一个样本并对$z$进行$P(X|z)$处理，作为$E_{z \sim Q}[\log P(X|z)$的一个近似。毕竟，我们已经在从数据集D采样的不同X值上进行了随机梯度下降。我们要优化的完整方程式是：
$$
\begin{array}{c}
E_{X \sim D}[\log P(X)-\mathcal{D}[Q(z|X)\|P(z|X)]]= \\
E_{X \sim D}[E_{z \sim Q}[\log P(X | z)]-\mathcal{D}[Q(z | X) \| P(z)]]
\end{array} \tag{8}
$$
如果采用该方程式的梯度，则可以将梯度符号移到期望值中。因此，我们可以从分布$Q(z|X)$采样$X$的单个值和$z$的单个值，并计算以下项的梯度：
$$
\log P(X|z)-\mathcal{D}[Q(z|X)\|P(z)] \tag{9}
$$
然后，我们可以在任意多个$X$和$z$样本上对该函数的梯度求平均，结果收敛到公式8的梯度。

但是，公式9存在一个重大问题。$E_{z \sim Q}[\log P(X|z)]$不仅取决于P的参数，还取决于Q的参数。但是，在公式9中，这种依赖性消失了！为了使VAE正常工作，必须驱动Q生成能够可靠解码的X代码。为了以不同的方式看待问题，方程9中描述的网络与图4中所示的网络非常相似（左）。该网络的前向传递工作良好，如果对Xandz的许多样本进行平均输出，则可以生成正确的期望值。

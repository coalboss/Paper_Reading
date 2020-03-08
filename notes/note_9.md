# Audio-visual Speech Enhancement Using Conditional Variational Auto-Encoder

摘要—可变自动编码器（VAE）是深层的生成隐变量模型，可用于学习复杂数据的分布。 VAE已成功用于学习语音信号之前的概率先验，然后将其用于语音增强。这种生成方法的一个优点是在训练时它不需要成对的干净和嘈杂的语音信号。在本文中，我们提出了VAE的视听变体，用于单声道和独立于说话者的语音增强。我们开发了条件VAE（CVAE），其中语音生成过程以嘴唇区域的视觉信息为条件。测试时，将视听语音生成模型与基于非负矩阵分解的噪声模型相结合，语音增强依赖于蒙特卡罗期望最大化算法。使用最新发布的NTCD-TIMIT数据集以及GRIDcorpus进行实验。结果证实，与纯音频VAE模型相比，所提出的视听CVAE有效地融合了音频和视频信息，并且改善了语音增强性能，尤其是当语音信号被噪声严重破坏时。我们还表明，拟议的无监督视听语音增强方法的性能优于最新的有监督的深度学习方法。

INTRODUCTION-语音增强（SE）的问题在于从嘈杂的单通道或多通道音频记录中估计干净的语音信号。音频语音增强（ASE）方法及相关算法，软件和系统的历史由来已久，例如[1]–[3]。在本文中，我们解决了视听语音增强（AVSE）的问题：除了音频之外，我们还利用唇动视频录制提供的视觉语音信息的优势。 AVSE的基本原理是，与音频信息不同，视觉信息（嘴唇运动）不会被声学扰动破坏，因此，视觉信息可以帮助语音增强过程，尤其是在存在低信噪比的音频信号的情况下（SNR）。

尽管已经表明视觉和音频信息的融合对于各种语音感知任务是有益的，例如[4] – [6]，AVSE的研究远远少于ASE。 AVSE方法可以追溯到[7]和后续工作，例如[8] – [13]。毫不奇怪，最近在深度神经网络（DNN）框架中解决了AVSE并且开发了许多有趣的架构和性能良好的算法，例如[14] – [18]。

在本文中，我们提出在可变自动编码器（VAE）的框架中融合单通道音频和单机视觉信息以增强语音。可以将其视为[19] – [24]中基于VAE的方法的多模式扩展，据我们所知，在无监督的学习环境中，可以提供最先进的ASE性能。为了将视觉观察结合到VAE语音增强框架中，我们建议使用条件可变自动编码器（CVAE）[25]。与[20]中一样，我们分三步进行。

首先，利用同步清晰的语音数据和视频语音数据，学习了音视频CVAE（AV-CVAE）体系结构的参数。这就产生了一个视听演讲先验模型。训练是完全无监督的，在这个意义上不需要混合各种噪声信号的语音信号。这与需要在多种噪声类型和噪声水平下进行训练以确保概括和良好性能的监督dnn方法形成对比，例如[14]–[16]，[26]，所得语音先验与混合模型和非负矩阵分解（NMF）噪声方差模型结合使用，以推断模拟语音信号时变响度的增益和NMF参数。第三，利用语音先验（VAE参数）以及推导出的增益和噪声方差重构干净语音。后者很可能被视为概率滤波器。利用NTCD-TIMIT数据集[27]和包含音频视频记录的网格语料库[28]，对学习的VAE结构及其变体、增益和噪声参数推断算法以及提出的语音重建方法进行了全面测试，并与最新方法进行了比较。

论文的其余部分安排如下。第二节总结了相关工作。在第三节中，我们简要地回顾了如何使用VAE来建立语音先验分布模型，然后在第四节中，我们介绍了两种VAE网络方差来从视觉数据中学习语音先验。在第五节中，我们提出了一种AV-CVAE模型，用于模拟视觉信息条件下的声速分布。在第六节中，我们讨论了推理阶段，即实际的语音增强过程。最后，我们的实验结果在第七节中给出。有关视听和可视语音增强示例的补充材料，请访问 https://team.inria.fr/perception/research/av-vae-se/

RELATED_WORK-在过去的几十年中，语音增强一直是一个受到广泛研究的话题，而完整的技术水平已超出了本文的范围。我们简要回顾有关单通道SE的文献，然后讨论AVSE中最重要的工作。

经典方法在短时傅立叶变换（STFT）域中基于噪声和/或语音功率谱密度（PSD）估计使用频谱减法[29]和Wienerfiltering [30]。另一种流行的方法是短期频谱幅度估计器[31]，最初是基于语音STFT系数的局部复数值高斯模型，然后扩展到其他密度模型[32]，[33]和对数频谱幅度估计器[34]，[35]。对语音信号PSD建模的流行技术是NMF，例如[37]-[39]。

最近，在DNN的框架中已经解决了SE [40]。有监督的方法可以学习噪声语音和清晰语音频谱图之间的映射，然后将其用于重建语音波形[41] – [43]。或者，将有噪声的输入映射到时间频率（TF）掩码，然后将其应用于输入以消除噪声并尽可能保留语音信息[26]，[44]-[46]。为了使这些有监督的学习方法能很好地概括并产生最新的结果，训练数据必须在说话者方面，甚至在噪声类型和噪声水平方面，都具有很大的可变性[42]， [44];实际上，这导致麻烦的学习过程。因此，无监督的DNN模型是一个很好的选择。 VAE提供了一种有趣的生成公式[47]。结合NMF，基于VAE的方法可在无人监督的学习环境中产生最新的SE性能[19] – [24]。以演讲者身份为条件的VAE也已用于依赖演讲者的多麦克风语音分离[48]，[49]和去耦[50]

在心理学中已经充分研究了使用视觉提示来补充音频，无论何时出现嘈杂，模棱两可或不完整的情况[4] – [6]。确实，言语产生意味着通过声道以及舌头和嘴唇的运动同时进行空气循环，因此言语感知是多峰的。提出了几种计算模型以利用音频和视觉信息之间的相关性来感知语音，例如[9]，[12]。在[8]中提出了一种多层感知器体系结构，以将与视觉特征相联系的噪声语音线性预测特征​​映射到净语音线性预测特征​​上。然后建立了Wiener滤波器进行去噪。视听Wienerfiltering后来使用特定于音素的高斯混合回归和filterbank音频功能进行了扩展[51]。其他AVSE方法利用无噪声的视觉信息[10]，[11]或利用双隐马尔可夫模型（HMM）[13]。

最新的监督式AVSE方法基于DNN。 [14]，[16]的基本原理是使用视觉信息来预测STFT域中的TF软掩码，并将此掩码应用于音频输入以消除噪声。在[16]中，为数据集中的每个说话者训练了视频语音架构，这产生了一个取决于说话者的AVSE方法。文献[14]的体系结构由以视频和音频数据作为输入的幅值子网和仅以音频作为输入的相位子网组成。两个子网都使用地面真实语音训练。然后，幅度子网预测一个二进制掩码，然后将其应用于输入信号的幅度和相位频谱图，从而预测滤波后的语音频谱图。 [17]和[15]的体系结构非常相似：它们由两个子网组成，一个用于处理嘈杂的语音，一个用于处理可视语音。然后将这两种编码连接起来并进行处理，最终获得增强的语音频谱图。 [17]和[15]之间的主要区别在于，前者预测增强的视觉和音频语音，而后者仅预测音频语音。在[18]中采用了获取用于将看不见的说话人的语音与未知噪声分离的二进制掩码的想法：混合DNN模型集成了堆叠式长短期记忆（LSTM）和卷积LSTM，用于视听（AV）掩码估计。

在刚刚提到的监督式深度学习方法中，将看不见的数据泛化是一个关键问题。主要问题是噪音和说话者变异性。因此，训练这些方法需要大量噪声类型和说话者的嘈杂混合物，以保证泛化。相比之下，该方法完全不受监督：其训练基于VAE，仅需要清晰的语音和视觉语音即可。增益和噪声方差是在测试中使用蒙特卡洛期望最大值（MCEM）算法估算的[52]。然后使用学习到的参数从音频和视频输入中重建干净的语音。后者很可能被视为概率维纳滤波器。与大多数基于监督的基于DNN的AVSE方法相反，该方法预测应用于噪声输入的TF掩码。基于标准SE分数并使用广泛使用的公共可用数据集的经验验证表明，我们的方法优于ASE方法[ 20]以及最新的监督式AVSE方法[15]。

Audio_VAE-在本节中，我们简要回顾[19]中首次提出的深度生成语音模型及其使用VAE的参数估计过程[47]。令$S_{fn}$表示频率索引$f\in\{0，...，F-1\}$和帧索引n处的复值语音STFT系数。在每个TF单元中，我们都有以下模型，将其称为audio VAE（A-VAE）：
$$
s_{fn}|\mathbf{z}_n \sim \mathcal{N}_{c}(0, \sigma_{f}(\mathbf{z}_{n})) \tag{1}
$$
$$
\mathbf{z}_n \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{2}
$$
其中$\mathbf{z}_n\in\mathbb{R}^L$，且$L\ll F$，是描述语音生成过程的潜在随机变量，$\mathcal{N}(\mathbf{0}, \mathbf{I})$是具有恒等协方差矩阵的零均值高斯分布，$\mathcal{N}_{c}(0, \sigma)$是单变量复正态高斯均值和方差为零的分布。让$\mathbf{s}_n\in\mathbb{C}^F$为向量，其成分为第$n$帧的语音的STFT系数。非线性函数集$\{\sigma_f：\mathbb{R}^L\longmapsto \mathbb{R}_+\}^{ F-1}_{f = 0}$被建模为共享输入$\mathbf{z}_n\in\mathbb{R}^L$的神经网络。这些神经网络的参数统称为$\mathbb{\theta}$。这个方差可以解释为语音信号的短期PSD模型。

VAE的一个重要特性是提供一种有效的方法来学习此类生成模型的参数[47]，并借鉴变分推理[53]，[54]的想法。让$\mathbf{s}=\{\mathbf{s}_n\in\mathbb{C}^F\}^{N_{tr}-1}_{n=0}$是干净语音STFT帧的训练数据集，让$\mathbf{z}=\{\mathbf{z}_n\in\mathbb{R}^L\}^{N_{tr}-1}_{n=0}$是相关的潜在变量。在VAE框架中，参数θ通过最大化对数可能性的下限$\ln p(\mathbf{s};\mathbf{\theta})$来估计，称为证据下界（ELBO），定义为：
$$
\mathcal{L}(\mathbf{s} ; \mathbf{\theta}, \mathbf{\psi})=\mathbb{E}_{q(\mathbf{z} | \mathbf{s} ; \mathbf{\psi})}[\ln p(\mathbf{s} | \mathbf{z} ; \mathbf{\theta})]-D_{KL}(q(\mathbf{z} | \mathbf{s} ; \mathbf{\psi}) \| p(\mathbf{z})) \tag{3}
$$
其中$q(\mathbf{z} | \mathbf{s} ; \mathbf{\psi})$表示难解的真实后验分布$p(\mathbf{z} | \mathbf{s} ; \mathbf{\psi})$的近似值，$p(\mathbf{z})$是$\mathbf{z}$的先验分布，而$D_{KL}(q\|p)$是Kullback-Leibler散度。独立地，对于所有的$l\in\{0,\ldots,L-1\}$和所有的$n\in\{0,\ldots,N_{tr}-1\}$，$q(\mathbf{z} | \mathbf{s} ; \mathbf{\psi})$定义为：
$$
z_{l n} | \mathbf{s}_{n} \sim \mathcal{N}(\tilde{\mu}_{l}(\tilde{\mathbf{s}}_{n}), \tilde{\sigma}_{l}(\tilde{\mathbf{s}}_{n})) \tag{4}
$$
其中$\tilde{\mathbf{s}}_{n} \triangleq(|s_{0 n}|^{2} \dots|s_{F-1 n}|^{2})^{\top}$。非线性函数$\{\tilde{\mu}_l:\mathbb{R}^F_+\mapsto\mathbb{R}\}^{L-1}_{l=0}$和$\{\tilde{\sigma}_l:\mathbb{R}^F_+\mapsto\mathbb{R}\}^{L-1}_{l=0}$被神经网络建模，共享作为输入的语音功率谱帧$\tilde{\mathbf{s}}_{n}$，并由$\mathbf{\psi}$共同地参数化。还通过最大化在（3）中定义的变量下界来估计参数集$\mathbf{\psi}$，这实际上等效于最小化$q(\mathbf{z} | \mathbf{s} ; \mathbf{\psi})$和难处理的真实后验分布$p(\mathbf{z} | \mathbf{s} ; \mathbf{\theta})$之间的Kullback-Leibler散度[53]。使用（1），（2）和（4）可以按如下方式开发此目标函数：
$$
\begin{array}{c}
\mathcal{L}(\mathbf{s} ; \mathbf{\theta}, \mathbf{\psi}) \stackrel{c}{=}-\sum_{f=0}^{F-1} \sum_{n=0}^{N_{t r}-1} \mathbb{E}_{q(\mathbf{z}_{n} | \mathbf{s}_{n} ; \mathbf{\psi})}[d_{\mathrm{IS}}(|s_{f n}|^{2} ; \sigma_{f}(\mathbf{z}_{n}))] \\
+\frac{1}{2} \sum_{l=0}^{L-1} \sum_{n=0}^{N_{t r}-1}[\ln \tilde{\sigma}_{l}(\tilde{\mathbf{s}}_{n})-\tilde{\mu}_{l}^{2}(\tilde{\mathbf{s}}_{n})-\tilde{\sigma}_{l}(\tilde{\mathbf{s}}_{n})]
\end{array} \tag{5}
$$
其中，$d_\mathrm{IS}(x;y)=x/y-\ln(x/y)-1$是Itakura-Saito发散度[36]。最后，使用采样技术结合所谓的“重新参数化技巧” [47]来逼近（5）中的棘手的期望，可以得到一个目标函数，该目标函数在$\mathbf{\theta}$和$\mathbf{\psi}$上均是可微的，并且可以使用梯度上升算法进行优化[47]。图1总结了A-VAE语音先验的编码器-解码器体系结构。

Visual_VAE-现在，我们介​​绍两个VAE网络变体，用于从视觉数据中学习语音先验，它们分别称为基本视觉VAE（V-VAE）和增强型V-VAE，在图2中进行了概述。可以看出，该体系结构类似于A-VAE ，但有明显的区别，输入是视觉观察结果（即嘴唇图像）。更详细地，标准的计算机视觉算法用于从说话的脸部图像中提取固定大小的边界框，并以嘴唇为中心例如一个嘴唇ROI。使用两层完全连接的网络（以下称为基础网络）将此ROI嵌入到视觉特征向量$\mathbf{v}_n\in\mathbb{R}^M$中，$M$是视觉嵌入的尺寸。视情况而定，可以使用由3D卷积层组成的附加的预先训练的前端网络（虚线框），然后是具有34层的ResNet，作为专门为监督视听语音识别而训练的网络的一部分[55]。第二个选项称为augmented V-VAE.

在变分推论[53]，[54]中，可以考虑潜在变量$\mathbf{z}$的任何分布来近似难处理的后验$p(\mathbf{z}|\mathbf{s};\mathbf{\theta})$并定义ELBO。对于V-VAE模型，我们探索使用近似后验分布$q(\mathbf{z}|\mathbf{v};\mathbf{\gamma})$定义为：
$$
z_{l n} | \mathbf{v}_{n} \sim \mathcal{N}(\bar{\mu}_{l}(\mathbf{v}_{n}), \bar{\sigma}_{l}(\mathbf{v}_{n})) \tag{6}
$$
其中$\mathbf{v}=\{\mathbf{v}_n\}^{N_{tr}-1}_{n=0}$是视觉特征的训练集，其中非线性函数$\{\bar{\mu}_{l}: \mathbb{R}^{M} \mapsto \mathbb{R}\}_{l=0}^{L-1}$和$\{\bar{\sigma}_{l}: \mathbb{R}^{M} \mapsto \mathbb{R}\}_{l=0}^{L-1}$共同由$\gamma$参数化的神经网络建模，需要输入$\mathbf{v}_n$。请注意，V-VAE和A-VAE共享相同的解码器体系结构，即（1）。最终，V-VAE的目标函数具有与（5）相同的结构，因此可以使用与上述相同的梯度上升算法来估计V-VAE网络的参数。

Audio-Visual VAE-现在我们研究视听VAE模型，即将音频语音与视觉语音相结合的模型。这种多模态方法背后的理由是音频数据经常被噪声破坏，而视觉数据却没有。在不失一般性的前提下，将假定音频和视频数据是同步的，即每个音频帧都与一个视频帧相关联。

为了结合上述A-VAE和V-VAE公式，我们考虑CVAE框架以学习结构化输出表示[25]。在训练中，将为CVAE提供数据以及相关的类别标签，以使网络能够学习结构化的数据分布。在测试中，为受过训练的网络提供类别标签，以从相应的类别中生成样本。 CVAE已被证明对于缺失值推断问题非常有效，例如，输入输出对部分可用的计算机视觉问题[25]。

在AV语音增强的情况下，我们考虑对AV特征的$N_{tr}$个同步帧进行训练，即$(\mathbf{s,v})=\{(\mathbf{s}_n,\mathbf{v}_n)\}^{N_{tr}-1}_{n=0}$，其中，如上所述，$\mathbf{V}_n\in\mathbb{R}^M$是嘴唇ROI嵌入。干净的语音仅在训练时可用，有条件于观察到的视觉语音。然而，视觉信息在训练和测试时都是可用的，因此，它作为所需的清晰音频语音的确定性先验。有趣的是，它也影响$\mathbf{z}_n$的先验分布。总结一下，考虑以下潜在空间模型，独立于虑所有的$l\in\{0,\ldots,L-1\}$和所有TF单元$(f,n)$：
$$
\mathbf{s}_{f n} | \mathbf{z}_{n}, \mathbf{v}_{n} \sim \mathcal{N}_{c}(0, \sigma_{f}(\mathbf{z}_{n}, \mathbf{v}_{n})) \tag{7}
$$
$$
z_{l n} | \mathbf{v}_{n} \sim \mathcal{N}(\bar{\mu}_{l}(\mathbf{v}_{n}), \bar{\sigma}_{l}(\mathbf{v}_{n})) \tag{8}
$$
其中非线性函数$\{\sigma_{f}: \mathbb{R}^{L} \times \mathbb{R}^{M} \mapsto \mathbb{R}_{+}\}_{f=0}^{F-1}$被建模为由$\mathbf{\theta}$参数化并接受$\mathbf{z}_{n}$和$\mathbf{v}_{n}$输入的神经网络，并且其中(8)与(6)相同，但是相应的参数集γ将具有不同的估计，如下所述。另外，请注意，(1)和(7)中的$\sigma_f$是不同的，但它们都对应于生成语音模型的功率谱密度。这助长了贯穿整篇论文的记号的滥用。建议的体系结构称为AV-CVAE，如图3所示。与第三节和图1的A-VAE和第四节和图2的V-VAE相比，$\mathbf{z}_n$先验分布的均值和方差由视觉输入决定。

现在我们引入分布$q(\mathbf{z}|\mathbf{s},\mathbf{v};\mathbf{\psi})$，它逼近如上所述的难以处理的后验分布$p(\mathbf{z}|\mathbf{s},\mathbf{v};\mathbf{\theta})$，独立于所有$l\in\{0,\ldots,L-1\}$和所有帧：
$$
z_{l n} | \mathbf{s}_{n}, \mathbf{v}_{n} \sim \mathcal{N}(\tilde{\mu}_{l}(\tilde{\mathbf{s}}_{n}, \mathbf{v}_{n}), \tilde{\sigma}_{l}(\tilde{\mathbf{s}}_{n}, \mathbf{v}_{n})) \tag{9}
$$
其中非线性函数$\{\tilde{\mu}_{l}: \mathbb{R}_{+}^{F} \times \mathbb{R}^{M} \mapsto \mathbb{R}\}_{l=0}^{L-1}$和$\{\tilde{\sigma}_{l}: \mathbb{R}_{+}^{F} \times \mathbb{R}^{M} \mapsto \mathbb{R}\}_{l=0}^{L-1}$共同建模为编码器神经网络，该编码器神经网络由$\mathbf{\psi}$参数化，该编码器神经网络在每帧将语音功率谱及其关联的视觉特征向量作为输入。模型参数的完整集合，即$\mathbf{\gamma}$，$\mathbf{\theta}$和$\mathbf{\psi}$，可以通过最大化训练数据集上的条件对数似然$\ln p(\mathbf{s}|\mathbf{v};\mathbf{\theta},\mathbf{\gamma})$的下界来估计，定义如下：
$$
\begin{aligned} \mathcal{L}_{\text {av-cvae }}(\mathbf{s}, \mathbf{v} ; \mathbf{\theta}, \mathbf{\psi}, \mathbf{\gamma}) &=\mathbb{E}_{q(\mathbf{z} | \mathbf{s}, \mathbf{v} ; \mathbf{\psi})}[\ln p(\mathbf{s} | \mathbf{z}, \mathbf{v} ; \mathbf{\theta})] \\ &-D_{\mathbf{K L}}(q(\mathbf{z} | \mathbf{s}, \mathbf{v} ; \mathbf{\psi}) \| p(\mathbf{z} | \mathbf{v} ; \mathbf{\gamma})) \end{aligned} \tag{10}
$$
其中$\mathbf{z}=\{\mathbf{z}_{n} \in \mathbb{R}^{L}\}_{n=0}^{N_{t r}-1}$。这种网络体系结构似乎对于手头的任务非常有效。实际上，如果看一下（10）中的成本函数，可以看出$\mathbf{K L}$项实现了$q(\mathbf{z} | \mathbf{s}, \mathbf{v} ; \mathbf{\psi})=p(\mathbf{z} | \mathbf{v} ; \gamma)$的最优值。从图3中可以看出，这可以通过忽略音频输入的贡献来发生。此外，代价函数（10）中的第一项试图在解码器的输出处尽可能地重构音频语音矢量。这可以通过在编码器的输入中尽可能多地使用音频矢量来完成。这与第二项的最佳行为相反，后者试图忽略音频输入。通过最小化总成本，可以将视频和音频信息融合到编码器中。

在AV-CVAE训练期间，从编码器建模的近似后验采样的变量$\mathbf{z}_n$，然后将其传递到解码器。但是，在测试时，仅使用解码器和先前的网络，而丢弃编码器。因此，从先前的网络采样$\mathbf{z}_n$，这与编码器网络基本不同。成本函数（10）中的KL散度项负责尽可能减少回归与先前网络之间的差异。人们甚至可以通过对KL散度项加权$\beta>1$来控制这一点：
$$
\begin{aligned} \mathcal{L}_{\beta \text { -av-cvae }}(\mathbf{s}, \mathbf{v} ; \mathbf{\theta}, \mathbf{\psi}, \mathbf{\gamma}) &=\mathbb{E}_{q(\mathbf{z} | \mathbf{s}, \mathbf{v} ; \mathbf{\psi})}[\ln p(\mathbf{s} | \mathbf{z}, \mathbf{v} ; \mathbf{\theta})] \\ &-\beta D_{K L}(q(\mathbf{z} | \mathbf{s}, \mathbf{v} ; \mathbf{\psi}) \| p(\mathbf{z} | \mathbf{v} ; \mathbf{\gamma})) \end{aligned} \tag{11}
$$
这是在[56]中引入的，即$\beta$-VAE，并被证明有助于自动发现可解释的因子化潜在表示。但是，在提出的AV-CVAE体系结构的情况下，我们遵循[25]中提出的不同策略，以缩小认知与现有网络之间的差距。结果，对（10）中定义的ELBO进行了如下修改：
$$
\begin{aligned} \tilde{\mathcal{L}}_{\text {av-cvae }}(\mathbf{s}, \mathbf{v} ; \mathbf{\theta}, \mathbf{\psi}, \mathbf{\gamma}) &=\alpha \mathcal{L}_{\text {av-cvae }}(\mathbf{s}, \mathbf{v} ; \mathbf{\theta}, \mathbf{\psi}, \mathbf{\gamma}) \\ &+(1-\alpha) \mathbb{E}_{p(\mathbf{z} | \mathbf{v} ; \mathbf{\gamma})}[\ln p(\mathbf{s} | \mathbf{z}, \mathbf{v} ; \mathbf{\theta})] \end{aligned} \tag{12}
$$
其中$0\le\alpha\le1$是一个权衡参数。请注意，原始ELBO是通过设置α= 1来获得的。上述成本函数右侧的新项实际上是（10）中的原始重建成本，但每个$\mathbf{z}_n$都是从先验分布中采样的，即$p(\mathbf{z}_n|\mathbf{v}_n;\mathbf{\gamma})$。这样，现有网络被迫学习适合于重构相应语音帧的潜在矢量。如下所示，该方法显着提高了整体语音增强性能。

为了在（12）中建立成本函数，我们注意到KL-散度项采用封闭形式的解，因为所有分布都是高斯分布。此外，由于关于$\mathbf{z}_n$的近似后验和先验的期望不易处理，因此我们通常在实践中使用蒙特卡罗估计对它们进行近似。经过一些数学上的操作后，获得以下成本函数：
$$
\begin{aligned}
\tilde{\mathcal{L}}_{\mathrm{av}-\mathrm{cvae}}&(\mathbf{s}, \mathbf{v} ; \mathbf{\theta}, \mathbf{\psi}, \mathbf{\gamma}) \\
=& \frac{1}{R} \sum_{r=1}^{R} \sum_{n=0}^{N_{t r}-1}(\alpha \ln p(\mathbf{s}_{n} | \mathbf{z}_{n, 1}^{(r)}, \mathbf{v}_{n} ; \mathbf{\theta}).\\
+&.(1-\alpha) \ln p(\mathbf{s}_{n} | \mathbf{z}_{n, 2}^{(r)}, \mathbf{v}_{n} ; \mathbf{\theta})) \\
+& \frac{\alpha}{2} \sum_{l=0}^{L-1} \sum_{n=0}^{N_{t r}-1}(\ln \frac{\tilde{\sigma}_{l}(\tilde{\mathbf{s}}_{n}, \mathbf{v}_{n})}{\bar{\sigma}_{l}(\mathbf{v}_{n})}.\\
-&.\frac{\ln \tilde{\sigma}_{l}(\tilde{\mathbf{s}}_{n}, \mathbf{v}_{n})+(\tilde{\mu}_{l}(\tilde{\mathbf{s}}_{n}, \mathbf{v}_{n})-\bar{\mu}_{l}(\mathbf{v}_{n}))^{2}}{\bar{\sigma}_{l}(\mathbf{v}_{n})})
\end{aligned} \tag{13}
$$
其中$\mathbf{z}_{n, 1}^{(r)} \sim q(\mathbf{z}_{n} | \mathbf{s}_{n}, \mathbf{v}_{n} ; \mathbf{\psi})$和$\mathbf{z}_{n, 2}^{(r)} \sim p(\mathbf{z}_{n} | \mathbf{v}_{n} ; \mathbf{\gamma})$。可以通过与经典VAE类似的方式来优化此成本函数，即通过将重新参数化技巧与随机梯度上升算法一起使用来进行优化。请注意，重新参数化技巧必须使用两次，对$\mathbf{z}_{n, 1}^{(r)}$和对$\mathbf{z}_{n, 2}^{(r)}$。

AV-CVAE for Speech Enhancement-本节介绍了基于提出的AV-CVAE语音模型的语音增强算法。它与[20]中提出的使用VAE进行纯音频语音增强的算法非常相似。首先提出了无监督噪声模型，其次是混合模型，并提出了估计噪声模型参数的算法。最后，描述了干净的语音推理过程。通过本节，$\mathbf{v}=\{\mathbf{v}_n\}^{N-1}_{n=0}$，$\mathbf{s}=\{\mathbf{s}_n\}^{N-1}_{n=0}$和$\mathbf{z}=\{\mathbf{z}_n\}^{N-1}_{n=0}$表示测试集的视觉特征，清晰语音STFT特征和潜在向量。这些变量与$N$帧的噪声语音测试序列相关。应当注意，测试数据与前几节中使用的训练数据不同。观察到的麦克风（混合物）帧用$\mathbf{x}=\{\mathbf{x}_n\}^{N-1}_{n=0}$表示。

Unsupervised Noise Model-像[19]，[20]中一样，我们使用基于NMF的无监督高斯噪声模型，该模型假设在TF单元间是独立的：
$$
b_{fn}\sim\mathcal{N}_c(0, (\mathbf{W}_b\mathbf{H}_b)_{fn}) \tag{14}
$$
其中$\mathbf{W}_{b} \in \mathbb{R}_{+}^{F \times K}$是频谱功率模式的非负矩阵，而$\mathbf{H}_{b} \in \mathbb{R}_{+}^{K \times N}$是时间激活的非负矩阵，选择$K$使得$K(F+N)\leqslant FN$ [36]。我们提醒您，需要根据观察到的麦克风信号来估算$\mathbf{W}_{b}$和$\mathbf{H}_{b}$。

Mixture Model-观察到的混合物（麦克风）信号的模型如下：
$$
x_{fn}=\sqrt{g_n}s_{fn}+b_{fn} \tag{15}
$$
对于所有TF单元$(f,n)$，其中$g_n\in\mathbb{R}_+$表示与帧有关且与频率无关的增益，如[20]中所描述的。考虑到跨帧的语音信号的可能高度变化的响度，该增益提供了AV-CVAE模型的鲁棒性。让我们用$\mathbf{g}=(g_0\ldots g_{N-1})^T$表示必须估计的增益参数的向量。进一步假设语音和噪声信号是相互独立的，因此通过组合（7），（14）和（15），我们可以获得所有TF单元$(f,n)$：
$$
x_{f n} | \mathbf{z}_{n}, \mathbf{v}_{n} \sim \mathcal{N}_{c}(0, g_{n} \sigma_{f}(\mathbf{z}_{n}, \mathbf{v}_{n})+(\mathbf{W}_{b} \mathbf{H}_{b})_{f, n}) \tag{16}
$$
让$\mathbf{x}_n\in\mathbb{C}^F$是向量，其分量为$n$帧处噪声混合的STFT系数。

Parameter Estimation-在定义了语音生成模型（7）和观察到的混合模型（16）之后，推理过程需要从观察到的STFT系数$\mathbf{x}$和观察到的视觉特征$\mathbf{v}$的集合中估计模型参数的集合$\mathbf{\phi}=\{\mathbf{W}_{b}, \mathbf{H}_{b}, \mathbf{g}\}$。然后，这些参数将用于估计干净语音的STFT系数。由于相对于潜在变量的积分是难处理的，因此不可能直接进行最大似然估计。另外，可以利用模型的潜在变量结构来推导期望最大化（EM）算法[57]。从一组初始的模型参数$\phi^\star$开始，EM直到收敛包括迭代：
-E步：计算$Q(\mathbf{\phi} ; \mathbf{\phi}^{\star})=\mathbb{E}_{p(\mathbf{z} | \mathbf{x}, \mathbf{v} ; \mathbf{\phi}^{\star})}[\ln p(\mathbf{x}, \mathbf{z}, \mathbf{v} ; \mathbf{\phi})]$
-M步：更新$\phi^{\star} \leftarrow \operatorname{argmax}_{\phi} Q(\phi ; \phi^{\star})$
1）E-步：由于观测值与（16）中的潜在变量之间存在非线性关系，因此我们无法计算后验分布$p(\mathbf{z}|\mathbf{x}, \mathbf{v} ; \mathbf{\phi})$，因此无法解析地评估$Q(\phi ; \phi^{\star})$。如[20]中所示，我们因此依赖于以下蒙特卡洛近似：
$$
\begin{aligned}
& Q\left(\phi ; \phi^{\star}\right) \approx \tilde{Q}\left(\phi ; \phi^{\star}\right)\\
& \stackrel{c}{=}-\frac{1}{R} \sum_{r=1}^{R} \sum_{(f, n)}\left(\ln \left(g_{n} \sigma_{f}\left(\mathbf{z}_{n}^{(r)}, \mathbf{v}_{n}\right)+\left(\mathbf{W}_{b} \mathbf{H}_{b}\right)_{f, n}\right)\right.\\
& \left.+\frac{\left|x_{f n}\right|^{2}}{g_{n} \sigma_{f}\left(\mathbf{z}_{n}^{(r)}, \mathbf{v}_{n}\right)+\left(\mathbf{W}_{b} \mathbf{H}_{b}\right)_{f, n}}\right)
\end{aligned} \tag{17}
$$
其中，$\stackrel{c}{=}$表示等于不依赖于$\phi$和$\phi^{\star}$的加法项，其中$\{\mathbf{z}_n^{(r)}\}^R_{r=1}$是使用马尔可夫链蒙特卡洛（MCMC）从后验$p\left(\mathbf{z}_{n} | \mathbf{x}_{n}, \mathbf{v}_{n} ; \mathbf{\phi}^{\star}\right)$提取的样本序列采样。在实践中，我们使用Metropolis-Hastings算法[58]，该算法构成了MCEM算法[52]的基础。在它们的Metropolis-Hastings算法的第$m$个迭代中，并且对于$n\in\{0,\ldots,N-1\}$独立，首先从随机游动分布中抽取出一个样本$\mathbf{z}_n$:
$$
\mathbf{z}_{n} | \mathbf{z}_{n}^{(m-1)} ; \epsilon^{2} \sim \mathcal{N}(\mathbf{z}_{n}^{(m-1)}, \epsilon^{2} \mathbf{I}) \tag{18}
$$
.....
Speech Reconstruction-令$\phi^\star=\{\mathbf{w}_b^*,\mathbf{H}_b^*,\mathbf{g}^*\}$表示由上述MCEM算法估计的参数集。令$\tilde{\mathbf{s}}_{f n}=\sqrt{g_{n}^{*}} s_{f n}$为（15）中引入的语音STFT系数的缩放版本，其中$g_n^*=(\mathbf{g}^*)_n$。最后一步是根据它们的后验均值估计这些系数[20]：
$$
\begin{aligned}
\hat{\tilde{s}}_{f n} &=\mathbb{E}_{p\left(\tilde{s}_{f n} | x_{f n}, \mathbf{v}_{n} ; \boldsymbol{\phi}^{*}\right)}\left[\tilde{s}_{f n}\right] \\
&=\mathbb{E}_{p\left(\mathbf{z}_{n} | \mathbf{x}_{n}, \mathbf{v}_{n} ; \boldsymbol{\phi}^{*}\right)}\left[\mathbb{E}_{\left.p\left(\tilde{s}_{f n} | \mathbf{z}_{n}, \mathbf{v}_{n}, \mathbf{x}_{n} ; \boldsymbol{\phi}^{*}\right)\left[\tilde{s}_{f n}\right]\right]}\right.\\
&=\mathbb{E}_{p\left(\mathbf{z}_{n} | \mathbf{x}_{n}, \mathbf{v}_{n} ; \boldsymbol{\phi}^{*}\right)}\left[\frac{g_{n}^{*} \sigma_{f}^{2}\left(\mathbf{z}_{n}, \mathbf{v}_{n}\right)}{g_{n}^{*} \sigma_{f}^{2}\left(\mathbf{z}_{n}, \mathbf{v}_{n}\right)+\left(\mathbf{W}_{b}^{*} \mathbf{H}_{b}^{*}\right)_{f, n}}\right] x_{f n}
\end{aligned} \tag{23}
$$
该估计对应于维纳滤波的“概率”版本，其中对潜在变量的后验分布进行平均滤波。如上所述，不能通过分析来计算此期望，而是可以使用与第VI-C1节相同的Metropolis-Hastings算法来近似。最终，从具有重叠叠加的逆STFT中获得语音信号的时域估计。

算法1中总结了完整的语音增强过程，我们将其称为AV-CVAE语音增强。

Algorithm  1
Audio-visual CVAE speech enhancement

1. Inputs:
   - Learned CVAE generative model for clean speech, i.e.,(7) and (9).
   - Noisy microphone frames $\mathbf{x}=\{\mathbf{x}_n\}_{n=0}^{N-1}$.
   - Video frames$\mathbf{v}=\{\mathbf{v}_n\}_{n=0}^{N-1}$.
2. Initialization:
   - Initialization  of  NMF noise parameters $H_b$ and $W_b$ with random nonnegative values.
   - Initialization  of  latent  codes $\mathbf{z}=\{\mathbf{z}_n\}_{n=0}^{N-1}$ using  the learned  encoder  network  (9)  with $\mathbf{x}=\{\mathbf{x}_n\}_{n=0}^{N-1}$ and $\mathbf{v}=\{\mathbf{v}_n\}_{n=0}^{N-1}$.
   - Initialization of the gain vector $\mathbf{g}=(g_0,\ldots,g_{N-1})^{\top}=\mathbf{1}$
3. while stop criterion not met do:
4. E-step: Compute (17) using the Metropolis-Hastings algorithm
5. M-$\mathbf{H}_b$-step: Update $\mathbf{H}_b$ using (20)
6. M-$\mathbf{W}_b$-step: Update $\mathbf{W}_b$ using (21)
7. M-$\mathbf{g}$-step: Update $\mathbf{g}$ using (22)
8. end while
9. Speech reconstruction: Estimate $\mathbf{s}=\{\mathbf{s}_n\}_{n=0}^{N-1}$ with (23)

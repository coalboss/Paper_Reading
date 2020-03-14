# Audio-Visual Speech Separation and Dereverberation with a Two-Stage Multimodal Network

这是一篇由腾讯AI Lab的研究者们发布的基于音视频的语音分离+去混响的一篇技术类paper，发表时间20200303，被IEEE JSTSP Special Issue on Deep Learning for Multi-modal Intelligence across Speech, Language, Vision, and Heterogeneous Signals接收。

## Abstract

在真实的听觉环境中，背景噪声，干扰语音和房间混响经常会使目标语音失真。在这项研究中，我们着眼于联合语音分离和混响，旨在将目标语音与背景噪声，干扰语音和房间混响区分开。为了解决这个根本上困难的问题，我们提出了一种利用音频和视频信号的新型多模态网络。所提出的网络体系结构采用两阶段策略，其中在第一阶段采用分离模块来衰减背景噪声和干扰语音，在第二阶段采用去混响模块来抑制房间的混响。这两个模块首先分别进行训练，然后基于新的多目标损失函数进行集成以进行联合训练。我们的实验结果表明，所提出的多模态网络比几个第一阶段和第两阶段基线产生的客观可理解性和感知质量始终更好。我们发现，与未经处理的混合物相比，我们的网络将ESTOI提高了21.10％，将PESQ提高了0.79。此外，我们的网络架构不需要了解说话者数量。

## 1. Introduction

在像鸡尾酒会这样的声学环境中，人工听觉系统能够在出现说话者干扰，背景噪音和房间混响的情况下，追踪单个目标语音源。语音分离，通常也称为鸡尾酒会问题，是将目标语音与背景干扰分离的任务[6]，[45]。来自其他来源的干扰声音和来自表面反射的混响都会破坏目标语音，这可能会严重降低听众的语音清晰度，以及语音处理计算系统的性能。为了改善语音分离的性能，已经进行了许多研究工作。受计算听觉场景分析（CASA）中时频（T-F）掩蔽概念的启发，语音分离最近已被公式化为监督学习，其中从训练数据中学习目标语音或背景干扰中的判别模式[44]。由于使用了深度学习，在过去的十年中，监督语音分离的性能得到了显着提高[49]，[45]。然而，在不利的声学环境中产生高质量的分离语音仍然是一个棘手的问题。

说话者分离在最近几年吸引了相当多的研究注意力，其目标是提取多个语音源，每个说话者一个。说话者无关的语音分离，其中在训练和测试之间不需要说话者是相同的，容易受到标签歧义（或排列）问题的困扰[51]，[14]。与说话者无关的语音分离的重要方法包括深度聚类[14]和不变排列训练（PIT）[57]，它们从不同角度解决了标签的歧义。深度聚类将说话人分离作为频谱聚类，而PIT使用动态计算的损失函数进行训练。最近的许多研究扩展了这两种方法。例如，在[27]中使用了名为TasNet的扩张卷积神经网络（CNN）进行时域语音分离，在训练过程中应用了句子级PIT [24]。解决标签歧义性的另一种方法是使用目标说话人的说话人-区别性声音提示，作为分离的辅助输入。在最近的一项研究中[46]，预先录制的来自目标说话者的简短句子被用作注意力控制的锚点，从而选择了要分离的目标说话者。类似地，在[48]中，说话者识别网络从目标说话者的参考信号中产生说话者辨别嵌入。然后将嵌入矢量与带噪混合物的频谱图一起馈入分离网络。这种方法的潜在优势是不需要知道说话者的数量。

说话者的面部动作或嘴唇动作等视觉提示可以补充说话者语音中的信息，从而有助于语音感知，尤其是在带噪的环境中[28]，[29]，[38]。基于这一发现，人们开发了多种算法来组合音频和视频信号，以多模态方式进行语音分离[36]。最近有使用深度神经网络（DNN）来实现此目标的兴趣。Houet等人[17]设计了一种基于多任务学习的视听语音增强框架。他们的实验结果表明，视听增强框架始终优于在没有视觉输入的情况下相同的体系结构。在[9]中开发了类似的模型，其中训练了CNN以直接从嘈杂的语音和输入视频中估计干净语音的幅度谱图。此外，Gabbayet等人[8]使用视频到语音的方法来合成语音，随后将其用于构造用于语音分离的T-F mask。其他相关研究包括[16]，[54]，[21]。

尽管上述基于深度学习的视听方法比传统的视听方法大大提高了分离性能，但它们不能解决说话人泛化的问题，这是有监督语音分离中的关键问题。换句话说，他们仅以与说话者有关的方式进行评估，不允许说话者从训练到测试的转变。最近的研究[7]，[1]，[33]，[30]，[53]已经开发出了与说话者无关的语音分离算法。埃弗拉特[7]设计了一个基于扩散卷积和双向长短期记忆（BLSTM）的多流神经网络，与以前的多个依赖于说话人的模型相比，它的性能要好得多。Afouraset等人[1]利用两个子网分别预测干净语音的幅度频谱图和相位频谱图。在[33]中，训练DNN预测音频和视频流是否在时间上同步，然后为语音分离用于产生多感官特征。Wu[53]开发了用于目标说话人分离的时域视听模型。请注意，这些研究针对的是近距离交谈场景中的单声道分离。

在真实的声学环境中，语音信号通常会因表面反射产生的混响而失真。去混响已被积极研究了数十年[3]，[32]，[31]，[11]。尽管基于深度学习的方法近年来显着改善了混响效果[10]，[52]，[55]，[58]，但是混响仍然是公认的挑战，尤其是当它与背景噪音，语音干扰或两者同时出现时。尽管在视听语音分离方面取得了令人鼓舞的进展，但最近的研究很少以多模态方式同时处理语音分离和混响。考虑到在嘈杂和混响环境中对人和机器听众进行分离和混响的重要性（例如自动语音识别），我们在本研究中着眼于独立于说话者的多通道语音分离和混响，其目的是将目标语音与干扰语音，背景噪声和房间混响区分开。受近期有关语音分离的著作[22]，[42]，[58]的启发，我们认为，由于它们的固有差异，在单独的阶段解决分离和混响效果可能更有效。因此，我们首先使用扩张的CNN将目标带混响语音与干扰语音和背景噪声分离开，然后采用BLSTM来消除分离语音信号的混响。随后，对两阶段模型进行联合训练以优化新的多目标损失函数，该函数将TF域中的均方误差（MSE）损失与时域中标度不变的信噪比（SI-SNR）损失相结合。我们的实验结果表明，所提出的多模态网络在未经处理的混合物上可将扩展的短时目标清晰度（ESTOI）[20]提高21.10％，并将语音质量的感知评估（PESQ）[37]提高0.79。除此之外，我们发现提出的网络大大优于几个一阶段和两阶段基准。在这项研究中，在存在干扰语音，背景噪声和房间混响的远场场景中，对基于视听的联合语音分离和混响进行了彻底研究。

本文的其余部分安排如下。在第二部分，我们介绍了多通道远场信号模型。第三节简要介绍了本研究中使用的几种听觉和视觉特征。在第四部分，我们详细描述了我们提出的视听多模态网络体系结构。第五节提供了实验设置。在第六节中，我们介绍并讨论了实验结果。第七节总结了本文。
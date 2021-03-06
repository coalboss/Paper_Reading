# Seeing Through Noise: Visually Driven Speaker Separation And Enhancement

这是一篇来自耶路撒冷希伯来大学的研究者的有关音视频语音增强和分离的技术类paper，发表时间2018年2月，被ICASSP2018接收。

## Abstract

当在嘈杂的环境中拍摄视频时，如何在过滤其他声音或背景噪音的同时隔离特定人的声音具有挑战性。我们提出了视听方法，以隔离单个说话人的声音并消除无关的声音。首先，通过将无声视频帧通过基于视频到语音的神经网络模型，将视频中捕获的面部动作用于估计说话者的声音。然后，将语音预测作为滤波器应用于有噪声的输入音频。这种方法避免了在学习过程中使用声音的混合，因为这种可能的混合的数量巨大，并且不可避免地会偏向训练后的模型。我们在两个视听数据集GRID和TCD-TIMIT上评估了我们的方法，并证明了我们的方法相对于众所周知的纯音频方法和原始视频到语音的预测，能够达到SDR和PESQ显着改善。

## 1. Intrduction

单通道说话人分离和语音增强已经得到了广泛的研究[1、2]。最近对神经网络进行了训练，可以将音频混合分离到它们的源中[3]。这些模型能够学习独特的语音特征，如谱带，音高和chrip [4]。纯音频方法的主要困难是它们在分离相似的人类声音（例如相同性别的混合物）方面表现不佳。

我们首先描述面部在视频中可见的两个说话者的混合语音的分离。我们继续将单个可见说话人的语音与背景声音隔离开来。这项工作建立在机器语音读取的最新进展的基础上，通过面部和嘴巴的可见运动生成语音[5，6，7]。

与其他利用在语音和噪声或两种声音的混合中训练模型的方法不同，我们的方法是依赖于说话者且噪声不变的。这样一来，即使在同一个人的两个声音重叠的情况下，我们也可以使用少得多的数据来训练模型，并且仍然可以获得良好的结果。

### 1.1. Related Work

**纯音频语音增强和分离** 用于单通道或单声道语音增强和分离的先前方法大多使用纯音频输入。通用频谱mask方法会生成mask矩阵，其中包含每个说话者占主导地位的时频（TF）分量[8、9]。Huang [10]这是第一个使用基于深度学习的方法进行与说话者相关的语音分离的方法。

Isiket等人[4]通过深度聚类来解决单通道多说话人$分离问题，其中以区别训练的语音嵌入作为聚类和分离语音的基础。Kolbaeket等人[11]介绍一种简单的方法，其中他们使用排列不变的损失函数，这有助于基础神经网络在不同的说话者之间进行区分。

**视听语音处理** 最近在视听语音处理中的研究广泛使用了神经网络。Ngiamet等人的工作[12]是这方面的开创性工作。具有视觉输入的神经网络已用于唇读[13]，声音预测[14]和学习无监督的声音表示[15]。

关于视听语音的增强和分离的工作也已经完成[16，17]。Kahn和Milner [18，19]使用手工制作的视觉特征来导出用于说话人分离的二进制和软mask。Houet等人[20]提出了基于CNN的模型来增强嘈杂的语音。他们的网络生成表示语音增强的频谱图。

## 2. VISUALLY-DERIVED SPEECH GENERATION

存在几种从说话人的无声视频帧中产生可理解语音的方法[5、6、7]。在这项工作中，我们依靠vid2speech [6]，在第2节中进行了简要介绍。2.1。应当注意的是，这些方法取决于说话者，这意味着必须为每个说话者训练一个单独的专用模型。

### 2.1 Vidspeech

在Ephratet等人的最新论文中。[6] 提出了一种基于神经网络的方法，用于从讲话人的无声视频帧序列中生成语音频谱图。他们的模型有两个输入：（i）一个K连续帧的视频剪辑，以及（ii）一个在连续帧之间的（K-1）密集光流场的“剪辑”。网络体系结构由双塔ResNet [21]组成，该网络采用上述输入并将其编码为表示视觉特征的潜矢量，随后将其馈送到一系列两个完全连接的层中，生成预测的mel谱图。随后是一个后处理网络，该网络汇总多个连续的预测并将其映射到表示最终语音预测的线性谱图。

## 3. Audio-visual Speech Separation

我们建议检查音频输入的频谱图（多种来源的混合），并将每个时频（TF）元素分配给其各自的来源。生成的频谱图用于重建估计的单个源信号。

上面的分配操作基于每个演讲者的语音频谱估计图，该语音频谱图是由Sec.2的视频语音模型生成的。由于视频到语音处理无法生成完美的语音信号，因此我们仅将其用作分离嘈杂混合物的先验知识。

### 3.1 Speech Separation for Two Speakers

在这种情况下，两个说话人$(D_1, D_2)$使用单个麦克风面对相机。我们假设发言人是已知的，即我们预先训练了两个单独的视频语音网络$(N_1, N_2)$，每个演讲者一个，其中$N_1$是使用说话者$D_1$的视听数据集训练的，而$N_2$是在说话者$D_2$上训练的。

给定说话人$D_1$和$D_2$的视频，它们的音轨包含他们的混合语音，语音分离过程如下：

1. 使用面部检测方法在视频中检测说话者$D_1$和$D_2$的面部[22]。
2. 使用网络$N_1$和$N_2$从各个面部预测说话人$D_1$和$D_2$的语音mel频谱图$S_1$和$S_2$。
3. 从输入音频生成混合mel频谱图$C$。
4. 对于每个$(t,f)$:
   $$
   \begin{array}{c}
   F_{1}(t, f)=\{\begin{array}{cc}
   1 & S_{1}(t, f)>S_{2}(t, f) \\
   0 & \text { otherwise }
   \end{array}.
   \end{array} \tag{1}
   $$
   $$
   F_2(t,f) = 1-F_1(t,f) \tag{2}
   $$
5. 每个说话人的分离谱图$P_i$由混合谱图$C$通过$P_i=C\odot F_i$生成，其中$\odot$表示逐元素相乘。
6. 从频谱图（$P_1$或$P_2$）重构分离的语音信号，为每个隔离频率使用原始相位。

可以修改上面第4步中的“胜者为王”的二进制分离，以生成比率mask，该比率mask为每个TF单元提供0和1之间的连续值，即，可以通过以下方式来生成两个mask$F_1$和$F_2$：
$$
F_{i}(t, f)=(\frac{S_{i}^{2}(t, f)}{S_{1}^{2}(t, f)+S_{2}^{2}(t, f)})^{\frac{1}{2}}, \quad i=1,2 \tag{3}
$$

Figure1：基于男性说话者的长期语音频谱（LTSS）的阈值功能示例。在此，对于每个频率$f$，阈值$\tau(f)$设置为训练数据的所有可见量的75％

### 3.2 Speech enhancement of a single speaker

在语音增强情况下，一个说话者$(D)$面对一台带有单个麦克风的摄像机。还会记录可能包括其他（看不见的）说话者声音的背景噪音。任务是将说话者的声音与背景噪音区分开。和以前一样，我们假设在该说话者的视听数据集上预先训练了视频语音网络$(N)$。但是与语音分离不同，只有单个语音预测可用。

由于我们假设说话者以前是已知的，因此我们将从说话者的训练数据中计算长期语音频谱（LTSS），以获取说话者语音中每个频率的分布。对于每个频率，我们选择一个阈值$\tau()$，以指示该频率何时可能来自此讲话者的语音，并且在抑制噪声时应保留该频率。例如，给定频率的阈值可以设置为前百分之$X$（在这种情况下，$X$是超参数）。阈值函数的一个例子可以在图2中看到。

给定同一说话者的新视频，且声音嘈杂，隔离说话者声音的过程如下:

1. 从训练数据的长期语音谱（LTSS）计算阈值函数$\tau(f)$。
2. 使用面部检测方法在输入视频中检测到说话者$D$的面部。
3. 使用网络$N$从检测到的脸部预测说话人的语音mel语谱图。
4. 从嘈杂的音频输入生成嘈杂的梅尔频谱图$C$。
5. 使用阈值$\tau(f)$构造的分离mask$F$：对于频谱图中的每个$(t,f)$，我们计算:
   $$
   \begin{array}{c}
   F(t, f)=\{\begin{array}{cc}
   1 & S(t, f)>\tau(f) \\
   0 & \text { otherwise }
   \end{array}.
   \end{array} \tag{4}
   $$
6. 嘈杂的mel频谱图通过以下操作进行过滤：$P=C\odot F$生成，其中$\odot$表示逐元素相乘。
7. 从频谱图$P$重构分离的语音信号，使用原始相位。

## 4. Experiment

### 4.1. Datasets

**GRID语料库** 我们对GRID视听句子语料库[23]进行了实验，这是一个庞大的音频和视频（面部）录音数据集，包含34个人说的1,000个3秒句子。GRID语料库总共包含51个不同的词。

**TCD-TIMIT** 我们在TCD-TIMIT数据集上进行了另外的实验[24]。该数据集由60位演讲者（每个演讲者约200个视频）以及3位口语者组成，他们经过专门训练以帮助口语者理解他们的视觉语音。演讲者使用前置摄像头和30度摄像头记录了TIMIT数据集[25]中的各种句子。

**混合方案** 对于每个实验，我们从两个相同性别的说话者的语音信号中合成音频混合。给定音频信号$s_1(t)$，$s_2(t)$，使用每个源的原始非归一化增益，将其混合物合成为$s_1(t)+s_2(t)$。所有实验中的信号均来自训练相关vid2speech模型时看不见的数据。

### 4.2. Performance evaluation

我们使用客观源分离评估评分来评估我们的实验结果，包括SDR，SIR和SAR [26]和PESQ [27]。除了这些度量外，我们还使用非正式的人类听觉定性评估了结果的清晰度和质量。我们大力鼓励读者观看和收听我们的项目网页1上提供的补充视频，这表明我们的方法有效。

### 4.3. Results

**分离** 表1显示了从GRID和TCD-TIMIT数据集中说出的句子的合成混合物的分离实验结果。GRID实验涉及测试来自两个男性说话人（S2和S3）的随机语音混合。TCD-TIMIT实验涉及一位女性讲话者（lipspeaker3）和她自己的声音的随机混合语音，强调了我们的方法的作用。我们对通过使用Huang等人的纯音频方法[10]获得的结果进行比较。此外，我们将vid2speech生成的原始语音预测与未使用任何分离方法的语音预测进行了比较。

可以看出，当处理诸如GRID之类的约束词汇数据集时，原始语音预测具有合理的质量（PESQ分数）。但是，当处理更复杂的数据集（如TCD-TIMIT）时，vid2speechgen生成的语音质量较差，而且语音理解几乎是难以理解的，该数据集包含来自较大词汇量的句子。在这种情况下，我们的分离方法会产生实际影响，最终的语音信号听起来要比原始语音预测好得多。我们使用真实源信号的频谱图来构造理想的二进制和比率mask，并将它们的分离分数作为我们分离方法的性能上限。分离的频谱图的示例如图2所示。

Figure2：来自GRID数据集的分离测试数据中某一段的频谱图

**增强** 表2显示了从GRID和TCD-TIMIT数据集中说出的句子的合成混合物进行增强实验的结果。GRID实验涉及两名男性说话者（目标说话者S2和背景说话者S3）的随机语音混合。TCD-TIMIT实验涉及两名女性讲话者的随机语音混合（lipspeaker3作为目标，lipspeaker2作为背景）。在这里，我们还提供了与vid2speech生成的原始语音预测的比较。我们使用真实源信号的频谱图作为“oracle”来评估我们方法的性能上限。

我们还定性地评估了语音和非语音背景噪声混合的增强方法，可以在我们的项目网页上看到其示例。

**未知讲话者的语音分离** 尝试使用在其他讲话者上训练的模型来预测未知讲话者的语音通常会导致不良结果。在本实验中，我们尝试将两个“未知”发言人的语音分开。首先，我们针对“已知”演讲者（来自GRID的S2）的数据对vid2speech网络 [5]进行了培训。训练数据由随机选择的句子组成（总共40分钟）。在根据分离方法预测每个“未知”发言人（来自GRID的S3和S5）的语音之前，我们使用少量实际发言人的样本（总计5分钟）对网络进行了微调。然后，我们将语音分离过程应用于由未知说话者说出的看不见句子的合成混合词。结果总结在表3中。

## 5. CONCLUDING REMARKS

这项工作表明，可以通过利用视觉信息来执行高质量的单通道语音分离和增强。与1.1节中提到的纯音频技术相比。由于我们获得了视觉信息的消歧能力，因此我们的方法不受同性别语音分离中通常观察到的相似语音声音特征问题的影响。

本文所描述的工作可以作为未来几个研究方向的基础。其中包括使用无约束的视听数据集，其中包括现实世界中的多扬声器和嘈杂的录音。需要考虑的另一个有趣点是使用我们的增强方法来改善语音识别系统的性能。以端到端的方式实施类似的语音增强系统可能也是一个有希望的方向。

Table1：使用二进制和比率mask对GRID和TCD-TIMIT数据集的分离质量与Huang等人的仅音频分离方法[10]进行比较，以及和原始vid2speech[6]预测的进行比较。

Table2：使用LTSS作为mask阈值函数的增强质量评估。

Table3：使用转移学习比较来自GRID语料库的未知说话者的分离质量。

# Multimodal SpeakerBeam: Single Channel Target Speech Extraction with Audio-Visual Speaker Clues

## Abstract

近年来，随着深度学习的出现，混合语音的处理有了显著的进展。特别是，利用神经网络实现了目标语音提取，它利用代表目标说话人特征的辅助线索，从语音混合物中提取目标说话人的语音信号。例如，从目标说话人所说的辅助句子中获得的音频线索被用来描述目标说话人的特征。音频线索应该捕捉目标说话人声音的细粒度特征（例如音高）。另外，还研究了从在混音中的目标说话人的面部讲话的视频中获得的视觉线索。视觉线索应主要包括从嘴唇运动中提取的语音信息。本文提出了一种新的目标语音提取方案，该方案结合了目标说话人的视听线索，充分利用了两种模式所提供的信息，并引入了一种attention机制，它在每一个时间帧上都强调说话人线索的最连续性。对两个说话人的混合实验表明，与传统的基于音频或视频说话人线索的方法相比，本文提出的基于音频或视频说话人线索的方法显著提高了提取性能。

Figure 1：说话人感知mask估计网络概述

Table 1: SDR（dB），用于评估使用仅音频、仅视频和音视频说话人线索的方法

Table 2：无多任务学习和有多任务学习的提出的方法的SDR（dB）

## 1. Introduction

近年来，随着深度学习技术的发展，基于神经网络的单通道声源分离方法在增强重叠说话人语音信号方面得到了广泛的应用。这些研究大多集中在盲源分离（BSS）方法上，如深度聚类（deep cluster）或排列不变训练（permutation invariant training）等，这种方法在没有任何先验信息的情况下（即，仅利用观察到的语音混合）将一个被观察到的语音混合分离到每个源中。BSS方法可以以完全盲的方式工作，但它们通常会受到全局排列模糊性的影响，即混合中的说话人和输出之间的映射是任意的。为了解决这一歧义问题，研究人员还研究了目标语音提取方法[3-8]，该方法利用目标说话人的其他特征线索，从观察到的语音混合中仅提取目标说话人的语音信号。通过利用目标说话人的线索，目标语音提取方案不存在全局排列模糊问题，即可以在一句话中跟踪特定说话人的语音。此外，通过在提取阶段利用附加信息，它有可能获得比BSS方案更好的语音质量[4，6]。

目标语音提取研究的重点是利用音频[3-5]或视频说话人线索[6-8]。两种方法都在目标语音提取方面提供了可喜的结果，并显示出不同的优缺点。先前使用音频说话人提示（例如，SpeakerBeam [3，4]）进行的研究假设仅音频流（用于输入混合和说话人提示）可用于提取目标说话人，而说话人提示（即目标说话人的语音），可以预先录制。[3-5]使用目标说话人说的预先录制的句子作为说话人线索，以适应网络行为，以便它从观察到的语音混合中提取目标说话人。基于音频的线索应捕获目标说话人语音的细粒度特征（例如，音高，音色等）。已经证实，在音频线索下，当混合物中的说话人具有与目标说话人相似的语音特性时，提取性能会下降[4]。

另一方面，以前使用视频说话人的研究，例如[6–8]，假设音频流（用于输入混合）和视频流（用于说话人线索）可用于提取目标说话人。作为说话人的线索，[6-8]利用了目标说话人在混合物中的视频录制的不规则的脸区域。基于视觉的线索应该主要从目标说话人的嘴唇运动中捕捉到语音信息，并且可以帮助我们更好地处理包含具有相似特征的声音的混合物[6]。然而，视觉图像的质量可能会受到目标说话人在任何时刻的行为的影响，并且在实践中可能会受到面部运动或遮挡的影响，不像音频线索一旦被预先录制，其质量就不会改变。

本文提出了一种新的基于多（音视频）说话人线索的目标语音提取方案，称之为多模式（音视频）说话人串，以充分利用音频和视频的说话人线索，提高对其中任何一条线索缺失或中断的鲁棒性。与使用单一模态作为目标说话人线索的传统方案相比，该方案同时利用音频和视觉模态作为说话人线索。本文针对基于attention的视听融合在自动视觉识别[9]和视频描述[10]任务中的成功应用，提出了一种融合视听说话人线索的attention机制。利用所提出的attention方案，我们可以在每个时间帧上利用更多的信息的说话人线索来提取目标语音。此外，我们提出了一个基于多任务学习的训练过程[11]，该过程同时考虑了使用音频-视频、仅音频和仅视频的说话人线索的提取损失。它使提出的音频或视频说话人提示的提取系统能够工作，即使音频或视频说话人提示不可用。

## 2. Conventional target speech extraction

### 2.1. Framework

在本文中，我们使用一种基于mask的方法[12]从观察到的语音混合物中提取目标说话人。通过在短时傅立叶变换（STFT）域中将时频mask$\mathbf{M}_s\in\mathbb{R}^{T\times F}$应用于观察到的语音混合物$\mathbf{Y}\in\mathbb{C}^{T\times F}$，来提取目标说话人的语音信号$\hat{\mathbf{X}}_s\in\mathbb{R}^{T\times F}$，如下 ：
$$
\hat{\mathbf{X}}_s=\mathbf{M}_s\odot\mathbf{Y} \tag{1}
$$
其中$\odot$表示逐元素的乘积，并且$T$和$F$分别表示时间帧数和频率单元数。

在目标语音提取设置中，我们假设在从混合语音中提取目标说话人的语音信号时，可以使用附加的说话人线索$\mathbf{C}_s$。目标说话人的时频mask $\mathbf{M}_s$是由说话人感知mask估计网络估计的，如下所示：
$$
\mathbf{M}_s=\operatorname{DNN}(|\mathbf{Y}|,\mathbf{C}_s) \tag{2}
$$
其中$\operatorname{DNN}(\cdot)$是深神经网络（DNN）的非线性变换，$|\mathbf{Y}|$表示$\mathbf{Y}$的幅度谱系数。

图1-（a）显示了一个典型的语音提取网络架构的示意图，该架构由说话人线索提取网络（SCnet）和一个通过集成模块利用说话人线索的mask估计网络组成。给定说话人线索作为输入，SCnet生成说话人线索的中间表示$\mathbf{Z}_s$。mask估计网络的集成模块将这种表示与从mask估计网络底层导出的观测混合语音的中间特征$\mathbf{Z}_M=\{\mathbf{z}_M^t;t=1,2,\ldots,T\}$结合起来。集成模块的输出$\mathbf{I}_s=\{\mathbf{i}_{st};t=1,2,\ldots,T\}$对应于目标说话人的一个中间表示，这使得可以通过mask估计网络的上层预测目标说话人的时频mask $\mathbf{M}_s$。通过在mask估计网络中集成额外的说话人线索，网络行为可以适应目标说话人的提取。

在集成过程中，我们可以考虑执行几种上述操作的方法。例如，[4]采用了基于逐元素乘法集成，例如$\mathbf{I}_s=\mathbf{Z}^M\odot\mathbf{Z}_s$。在对比中，[6，7]采用了基于级联的集成，例如$\mathbf{I}_s=[\mathbf{Z}^M,\mathbf{Z}_s]$，其中，$[\cdot]$表示在特征维上的级联。在下面，为了与前面的工作一致[4]，我们采用了基于逐元素乘法的集成。

### 2.2. Speech extraction based on audio speaker clue

对于音频说话人线索[3-5]，附加信息包括一个基于STFT的幅度谱特征序列$\mathbf{C}_{s}^{\mathrm{A}} \in \mathbb{R}^{T^{\mathrm{A}} \times F}$，这是由目标说话人所说的预先录制的句子得出的，其中$T^{\mathrm{A}}$是音频说话人的线索时间帧数。

给定音频说话人线索$\mathbf{C}_s^A$作为输入，集成特征$\mathbf{i}_{st}^A\in\mathbb{R}^{1\times H}$计算如下：
$$
\mathbf{z}_s^A=\operatorname{Avg}(\operatorname{SCnet}^A(\mathbf{C}_s^A)) \tag{3}
$$
$$
\mathbf{i}_{s t}^{\mathrm{A}}=\mathbf{z}_{t}^{\mathrm{M}} \odot \mathbf{z}_{s}^{\mathrm{A}} \quad(t=1,2, \cdots, T) \tag{4}
$$
其中$\mathbf{z}_{s}^{\mathrm{A}} \in \mathbb{R}^{1 \times H}$表示音频线索的提取特征，$\operatorname{SCnet}^A(\cdot)$是音频线索的特征提取网络，$\operatorname{Avg}(\cdot)$是时间轴上的平均运算，$H$是$\operatorname{SCnet}^A(\cdot)$的输出维数。$\operatorname{Avg}(\operatorname{SCnet}^A(\cdot))$对应直接从输入话语中提取语音特征的序列摘要网络[13]。

注意，根据音频说话人的线索，$\operatorname{SCnet}^A(\cdot)$将振幅谱系数的序列$\mathbf{C}_s^A$映射为如等式（3）所示的向量。音频线索是时间不变量。

### 2.3. Speech extraction based on visual speaker clue

当使用视频说话人线索[6–8]时，附加信息包括基于视频的特征$\mathbf{C}_{s}^{\mathrm{V}} \in \mathbb{R}^{T^{\mathrm{V}} \times D}$，从目标说话人的裁剪面区域提取。在文献[6]的基础上，本文采用预先训练好的人脸识别模型Facenet[14]提取的人脸嵌入特征作为基于视频的特征。这里，$T^V$是可视说话人线索的时间维数，$D$显示空间嵌入的维度。

给定视觉说话人线索$\mathbf{C}_s^V$作为输入，集成特征$\mathbf{i}_{st}^V\in\mathbb{R}^{1\times H}$计算如下：
$$
\mathbf{Z}_s^V=\operatorname{Avg}(\operatorname{SCnet}^V(\mathbf{C}_s^V)) \tag{5}
$$
$$
\mathbf{i}_{s t}^{\mathrm{V}}=\mathbf{z}_{t}^{\mathrm{M}} \odot \mathbf{z}_{st}^{\mathrm{V}} \quad(t=1,2, \cdots, T) \tag{6}
$$
其中$\mathbf{Z}_{s}^{\mathrm{V}}=\{\mathbf{z}_{st}^{\mathrm{V}};t=1,2,\ldots,T^V\}$表示视觉线索的提取特征，$\operatorname{SCnet}^V(\cdot)$是视觉线索的特征提取网络。

注意，与音频说话人线索相反，视觉说话人线索在等式(5)中使用是时变的。作为视听处理中的常见情况，在音频和视频流之间的每秒帧数（fps）中存在差异；例如，视频为25 fps（40 ms），而音频为50 fps（20 ms）。输入序列的每个视频帧都必须对应于音频帧。在本文中，我们通过对多个音频帧重复一个视频帧来对齐它们，例如，$\tilde{\mathbf{Z}}^{\mathrm{V}}=\{\mathbf{z}_{s, 1}^{\mathrm{V}}, \mathbf{z}_{s, 1}^{\mathrm{V}}, \mathbf{z}_{s, 2}^{\mathrm{V}}, \mathbf{z}_{s, 2}^{\mathrm{V}}, \cdots\}$。

## 3. Proposed multimodal SpeakerBeam

### 3.1. Attention-based fusion of audio-visual speaker clues

在该方法中，我们假设从观察到的语音混合中提取目标说话人的信号时，存在多模态说话人线索（即音频和视频）的情况，为了充分利用这两种类型的说话人线索的优点，提出了一种基于attention的融合机制来融合音频和视频说话人线索。注意attendtion的作用是在混合物的每个时间帧强调（即，轻轻地选择）提取目标说话人的更多信息性说话人线索。

图1-（b）是所提出的网络结构的示意图，它通过添加基于注意的融合机制扩展了说话人线索的特征提取过程。给定多个说话人线索，即音频线索$\mathbf{C}^A_s$和视频线索$\mathbf{C}^V_s$，特征提取模块分别将它们转换为中间特征序列$\mathbf{z}^A_s$和$\mathbf{Z}^V_s$，如等式（3）和（5）所述。然后，attention机制将这些说话者线索$\mathbf{z}^A_s$和$\mathbf{Z}^V_s$组合为视听说话者线索的中间特征序列$\mathbf{Z}_{s}^{\mathrm{AV}}=\{\mathbf{z}_{st}^{\mathrm{AV}};t=1,2,\ldots,T\}$。最后，网络以类似于第2.1节中所述的方式，基于视听说话者线索$\mathbf{Z}_{s}^{\mathrm{AV}}$生成时频掩码$\mathbf{M}_s^{\mathrm{AV}}$。

给定视听说话人线索$\mathbf{C}^A_s$和$\mathbf{C}^V_s$作为输入，基于视听说话人线索的综合特征$\mathbf{i}_{st}^{\mathrm{AV}}\in\mathbb{R}^{1\times H}$计算如下：
$$
\mathbf{z}_{s t}^{\mathrm{AV}}=\underbrace{\sum_{\psi \in\{\mathrm{A}, \mathrm{V}\}} a_{s t}^{\psi} \mathbf{z}_{s t}^{\psi}}_{\text {Attention }} \quad(t=1,2, \ldots, T) \tag{7}
$$
$$
\mathbf{i}_{s t}=\mathbf{z}_{t}^{\mathrm{M}} \odot \mathbf{z}_{st}^{\mathrm{AV}} \quad(t=1,2, \cdots, T) \tag{8}
$$
其中，$\{a_{s t}^{\psi}\}_{\psi \in\{\mathrm{A}, \mathrm{V}\}}$是目标说话人$s$在时间步$t$上的attention权重。

我们采用文献[15]中提出的加性attention机制来计算attention权重。注意权重$\{a_{s t}^{\psi}\}_{\psi \in\{\mathrm{A}, \mathrm{V}\}}$由混合$\mathbf{z}^{M}_t$的中间特征和说话人线索$\{\mathbf{z}_{s t}^{\psi}\}_{\psi \in\{\mathrm{A}, \mathrm{V}\}}$计算如下：
$$
e_{st}^{\psi}=\mathbf{w} \tanh (\mathbf{W} \mathbf{z}_{t}^{\mathrm{M}}+\mathbf{V} \mathbf{z}_{s t}^{\psi}+\mathbf{b}) \tag{9}
$$
$$
a_{s t}^{\psi}=\frac{\exp (\epsilon e^{\psi})}{\sum_{\psi \in\{\mathrm{A}, \mathrm{V}\}} \exp (\epsilon e^{\psi})} \tag{10}
$$
其中，$\mathbf{w}$，$\mathbf{W}$，$\mathbf{V}$，$\mathbf{b}$可训练重量和偏差参数，$\epsilon$是锐化因子[15]。在这里，对于音频说话人，我们使用所有时间段的时不变（全局）线索，即$\mathbf{z}_{st}^{\mathrm{A}}=\mathbf{z}_{s}^{\mathrm{A}}(t=1,2,\ldots,T)$。

### 3.2. Multitask learning-based training procedure

我们假设一组输入和目标特征$\{\mathbf{Y}, \mathbf{C}_{i}^{\mathrm{A}}, \mathbf{C}_{i}^{\mathrm{V}},\mathbf{X}_{i}\}_{i=1}^{I}$可用于训练模型，其中$\mathbf{X}_{i} \in \mathbb{C}^{T \times F}$是混合语音中第$i$个说话人的目标语音信号，$I$表示混合语音中的说话人数量。

我们建议使用多任务学习（MTL）使所提出的视听提取系统即使在没有音频或视频说话人线索的情况下也能工作。基于多任务学习的目标函数考虑三种情况，即1）有视听线索，2）只有音频线索，3）只有视觉线索：
$$
L_{\mathrm{MTL}}=\alpha L_{\mathrm{AV}}+\beta L_{\mathrm{A}}+\gamma L_{\mathrm{V}} \tag{11}
$$
$$
L_{\psi}=\frac{1}{I} \sum_{i=1}^{I} l(\mathbf{M}_{i}^{\psi} \odot|\mathbf{Y}|,|\mathbf{X}_{i}|) \tag{12}
$$

其中，$\psi\in\{\mathrm{AV},\mathrm{V},\mathrm{V}\}$，一组参数$\{\alpha,\beta,\gamma\}$是多任务权重，$l(\mathbf{A}, \mathbf{B})=\frac{1}{T F}\|\mathbf{A}-\mathbf{B}\|^{2}$是均方误差（MSE）准则。这里，$\mathbf{M}_{i}^{\mathrm{AV}}, \mathbf{M}_{i}^{\mathrm{A}}$ 和 $\mathbf{M}_{i}^{\mathrm{V}}$分别表示基于等式（3）（5）和（7）的中间特征$\mathbf{z}_{i}^{\mathrm{AV}}$，$\mathbf{z}_{i}^{\mathrm{A}}$和$\mathbf{z}_{i}^{\mathrm{V}}$估计的时频掩码。注意，当音频或视频线索单独使用时，即$\psi=\{A\}$或$\psi=\{V\}$，该类线索的attention权重变为1（见等式（10））。

## 4. Experiments

我们比较了我们提出的使用多说话人线索（SpeakerBeam-AV，SpeakerBeam-AV-MTL）的提取方法和使用单个说话人线索（Baseline-A[4]，Baseline-V[6，7]）的两种传统提取方法。SpeakerBeam-AV和SpeakerBeam-AV-MTL对应于所提出的具有音频和视觉提示的方法，其中我们将多任务权重设置为$\{\alpha=1.0,\beta=0.0,\gamma=0.0\}$和$\{\alpha=0.8,\beta=0.1,\gamma=0.1\}$。Baseline-A和Baseline-V分别对传统方法有音频或视频提示。注意，Baseline-A可被视为$\{\alpha=0.0,\beta=1.0,\gamma=0.0\}$，Baseline-V可被视为$\{\alpha=0.0,\beta=0.0,\gamma=0.1\}$。

### 4.1. Experimental conditions

#### 4.1.1 Data

为了评估我们提出的方法的有效性，我们建立了一个基于LipReading Sentences 3（LRS3-TED）视听语料库[16]的语音混合仿真数据集。我们的数据集由两个说话人混合产生，信噪比（SNR）在0和5 dB之间，类似于广泛使用的WSJ0-2mix语料库[1]。此外，我们还将语音降到8khz，以降低计算和存储成本。

培训集由500个说话人的50000个混合组成。开发集由300个说话人的10000个混合组成。训练和开发集所用的说话人是从LRS3-TED语料库的pre-train和train-val中随机抽取的。该测试集由基于LRS3-TED语料库中的测试集的295名说话人的5000个混合组成。数字“295”对应于在LRS3-TED语料库的测试集中有两个以上句子的说话人的数量。

为了获得视频的说话人线索，我们使用了与混合中每个说话人相对应的视频数据。对于音频说话人线索，我们在数据库中随机选择了同一个说话人的一个不用于生成混合语音的句子。

#### 4.1.2. Settings

作为音频特征，我们使用了由具有64 ms窗口长度和20 ms窗口偏移的STFT计算的幅度谱图。作为视觉特征，我们使用了基于Facenet的特征，这些特征是使用GitHub仓库[17]中提供的软件和预先训练的模型为每个视频帧（每秒25帧，即40毫秒的移动）提取的。

在所有的实验中，我们使用了一个3层的BLSTM网络，它有512个单元。每一个BLSTM层后接一个有512个单元的线性投影层，以组合前向和后向LSTM输出。我们使用一个全连接的层来输出一个由sigmoid激活函数估计的振幅mask。对于集成层，我们采用基于逐元素相乘的集成。集成层是在第一个BLSTM层之后插入的。

对于音频线索的特征提取网络（即等式（3）中的$\operatorname{SCnet}^{A}(\cdot)$），我们使用了一个网络，该网络包含2个具有200个单元和ReLU激活的全连接层，然后是1个具有512个单元的线性输出层。另一方面，对于视觉线索的特征提取网络（即等式（5）中的$\operatorname{SCnet}^{V}(\cdot)$），我们使用了具有3个具有256个通道的卷积层的网络（$\operatorname{filter}=7\times1,5\times1,5\times1,\operatorname{shift}=1\times1,1\times1,1\times1$），其中空间卷积是在[6]的启发下在时间轴上执行的，随后是具有512个单位的1个线性输出层。我们对每个卷积层都采用了ReLU激活和BN[18]。我们将注意力内积的维数（即方程（9）中的$\mathrm{w}$的维数）设置为200，将锐化因子$\epsilon$设置为2。

采用Adam算法[19]进行优化，初始学习率为0.0001，采用梯度裁剪[20]，经过200个epoch后停止训练。

我们用BSS Eval工具箱计算的信号失真率（SDR）来评估结果[21]。所有的实验结果都是通过平均两个说话人在混合物中的提取性能得到的。

### 4.2. Experimental results

#### 4.2.1. Evaluation: Single clue vs. Multiple clues

表1列出了未经处理的混合物，基线和提出的SpeakerBeam-AV的SDR得分。“Diff”和“Same”分别表示不同性别和相同性别的混合物的分数。“All”表示所有混合物的平均分数。

从表1中我们可以确认，在此实验设置中，传统的Baseline-A（音频提示）和Baseline-V（视频提示）具有可比性。此外，基于性别的结果显示，不同性别的人使用Baseline-A效果更好，而相同性别的人则使用Baseline-V效果更好。

提出的SpeakerBeam-AV（视听线索）成功优于使用单个说话人线索的传统Baseline-A和Baseline-V。具体来说，我们确认SpeakerBeam-AV可以显着改善同性别的提取性能。

#### 4.2.2. Analysis of performance improvement

我们更详细地研究了所提出方法的性能改进。图2显示了Baseline-A（音频），Baseline-V（视觉）和SpeakerBeam-AV（视听）的SDR改进的直方图，其中每个直方图单元代表2.5 dB间隔和垂直轴显示每个单元中已评估混合物的（标准化）计数。

着眼于低于2.5 dB的直方图单元，这表明提取性能较差（例如，系统提取了干扰说话人的语音信号，而不是目标说话人的语音信号），我们可以观察到与使用单个说话人线索的常规Baseline-A和Baseline-V相比，提出的使用多个说话人线索SpeakerBeam-AV大大降低了具有如此低SDR分数（即降低提取失败率）的句子数。这表明提出的SpeakerBeam-AV的行为更加稳定和强大。

#### 4.2.3. Evaluation of multitask learning effect

表2显示了使用建议的经过训练的单任务（SpeakerBeam-AV）和多任务（SpeakerBeam-AV-MTL）目标的系统获得的所有混合（即表1中的“all”）的平均SDR分数。 “Weightw”表示等式中的多任务权重。 （11）。 “Clues”表示在提取阶段使用的说话者线索；音频（A），视觉（V）和视听（AV）线索。

我们确认使用单个说话人提示（特别是Clues = V）时，SpeakerBeam-AV的性能会下降;另一方面，即使在这种情况下，SpeakerBeam-AV-MTL的性能可能会优于Baseline-A和Baseline-V（请参阅表1） ），也要保持与使用音频和视频提示（Clues= AV）的相当的性能。该结果表明，即使在没有音频或视频说话者线索的情况下，多任务目标也能有效地使所提出的视听提取系统工作。此外，结果表明，在训练阶段使用多个说话者线索可以有效地提高在提取阶段使用单个说话者线索的系统的提取性能。

## 5. Conclusion

本文提出了一种新颖的目标语音提取方案，该方案使用多个（视听）说话者线索。我们引入了一种基于attention的机制来整合音频和视觉线索以及基于多任务学习的训练过程。实验结果表明，与使用单个音频或视觉线索的常规基线相比，我们提出的使用视听线索的多模态说话人集束显着提高了提取性能。另外，我们观察到，当音频或视频说话者线索不可用时，所提出的多任务学习方案可以改善所提出的视听提取系统的性能。

在本文中，我们在控制的设置中评估了我们提出的方法，在该设置中，对于所有混合物始终可以使用高质量的音频和视觉线索。未来的工作将包括对基于attention的机制在每个时间范围内处理更多信息提示线索的有效性进行更详细的研究，以更具挑战性的（真实）设置，例如1）当音频提示线索嘈杂或简短时，或2）当由于面部移动或遮挡而缺少某些视觉线索时。另外，我们计划使用更大的数据集进行评估，并研究具有更大表示能力的网络架构。

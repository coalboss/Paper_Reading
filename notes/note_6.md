# The Conversation: Deep Audio-Visual Speech Enhancement

这是一篇来自Oxford 的研究者们的关于音视频语音增强的论文，被Interspeech 2018接收。

## Abstract

我们的目标是将单个说话人从视频中的多人同时讲话中分离出来。这一领域的现有研究集中在试图将话语与受控环境中的已知说话人分开。本文提出了一种深度视听语音增强网络，能够在相应视频中给定嘴唇区域的情况下，通过预测目标信号的幅度和相位，提取说话人的语音。该方法适用于训练过程中听不到和看不见的说话人，也适用于不受约束的环境，我们给出了很强的定量和定性结果，并给出了极具挑战性的真实世界的例子。

**Figure 1**：视听增强架构概述。它由两个模块组成：一个幅度子网和一个相位子网。第一个子网接收带噪信号的幅度频谱图和说话人视频作为输入，并输出一个软mask。然后，我们将输入幅度与mask逐元素相乘以生成滤波后的幅度频谱图。然后将预测幅度以及从带噪信号获得的相位频谱图一起馈入第二子网，该第二子网产生相位残差。残留物被添加到带噪相位中，从而产生增强的相位谱图。最终，增强的幅度和相位频谱被转换回时域，从而产生增强的信号。

**Figure 2**：视听增强网络。BN：批量归一化，C：通道数； K：内核宽度； S：步幅–小数表示转置的卷积。该网络由幅度和相位子网组成。基本构建单元是左侧显示的具有预激活的时间卷积块[37]。在每个卷积层之后都添加了跳过连接标识（加快了训练速度）。卷积层在幅度子网络中共有1536个通道，在相位子网络中具有1024个通道。使用深度可分离的卷积层[38]，它由沿着每个通道的时间维度的单独卷积组成，然后是在位置上投影到新的通道维度上（相当于内核宽度为1的卷积）

**Table 1**：对于不同说话者数量（以＃Spk表示）的场景，LRS2数据集上语音增强性能的评估。幅度（Mag）和相位（$\Phi$）列指定用于重构的频谱图是预测的还是 直接从混合或地面真实信号获得：混合：混合; Pr：预测; GT：地面真相; GL：格里芬-林; SIR：信号干扰比; SDR：信号失真比; PESQ：语音质量的感知评估 在0到4.5之间； （对于所有三个而言，越高越好）； WER：来自现成的ASR系统的字错误率（越低越好）。真实信号的wer是8.8%。

**Table 2**：对于3个同时讲话的人，在Vox-Celeb2数据集上评估语音增强性能的情况，表1的标题中描述了符号。此处使用的其他度量标准：SAR：信号与伪像比； STOI：短期目标可理解性，介于0和1之间； PESQ-NF：模型尚未在VoxCeleb上进行微调的PESQ得分；更高更好。

## 1. Introduction

在电影The Conversation(dir. Francis Ford Coppola, 1974)。由Gene Hackman扮演的主人公前往超长的地方，记录一对夫妇在拥挤的城市广场上的对话。尽管放置了许多巧妙的麦克风，但他并没有利用演讲者的嘴唇动作来压制附近其他人的讲话。在本文中，我们提出了一种新的视听语音增强模型，可供他使用。

更笼统地说，我们提出了一种视听神经网络，它可以利用目标讲话者嘴唇上的视觉信息将讲话者的声音与其他人隔离开：给定嘈杂的音频信号和相应的讲话者视频，我们产生的增强音频信号仅包含目标发言者的声音而其余发言者的声音和背景噪声得到抑制。

与其从头开始合成声音，这将是一项艰巨的任务，而是预测一个可以过滤输入带噪频谱图的mask。许多语音增强方法专注于仅细化带噪输入信号的幅度，并将带噪相位用于信号重建。这对于高信噪比的场景效果很好，但是随着SNR的降低，嘈杂的相位就变成了地面真实噪声的不良近似值[1]。相反，我们为幅度和相位都提出了校正模块。该结构在图1中进行了概述。在训练中，我们使用在单词级唇读任务上预先训练的网络来初始化可视流，但是在此之后，我们从未标记的数据中进行训练（第3.1节），其中没有在单词，字符或音素级别需要显式注释。

该模型有许多可能的应用。其中之一是自动语音识别（ASR）–虽然机器扫描在无噪声的环境中可以相对较好地识别语音，但是在嘈杂的环境中，用于识别的性能却显着下降[2]。我们提出的增强方法可以解决此问题，并改善例如拥挤环境中手机的ASR或YouTube视频的自动字幕。

实验上可以最多支持5个人分离出独自的音频，并且我们展示了强大的定性和定量性能。评估后的模型是在不受限制的“野外”环境中评估的，而说话人和语言在培训时是看不见的。据我们所知，我们是第一个在这种一般条件下实现改进的。我们通过<http://www.robots.ox.ac.uk/~vgg/demo/theconversation>上的交互式演示提供补充材料。

### 1.1 Related works

各种工作已经提出了分离多说话人同时语音的方法。其中大多数是基于只使用音频的方法，例如，通过使用已知说话人的声音特征[3，4，5，6，7]。与纯音频方法相比，我们不仅将语音分离出来，而且利用视觉信息将其恰当地分配给说话人。

语音增强方法传统上只涉及过滤频谱幅度，但是最近提出了许多方法来共同增强幅度频谱和相位频谱[1、8、9、10、11、12、13]。Griffinand Lim提出了一种从语音合成中根据给定幅度估计相位谱的方法[14]。

在进行深度学习之前，通过预测mask[15、16]或其他方式[17、18、19、20、21、22,23]，已经开发了许多用于增强视听语音的先前工作，并对音频-在[24]中提供了视觉源分离。但是，我们将在此集中介绍使用深度学习框架在这些方法上建立的方法。

在[25]中，开发了一种深度神经网络，可以从说话人的无声视频帧中生成语音。该模型在[26]中用于语音增强，其中预测的频谱图用作过滤带噪语音的mask。但是，带噪音频信号未在pipeline中使用，并且网络未针对语音任务进行训练。相反，[27]在混合语音输入和输入视频两者上合成了干净的信号。[28]也使用类似的视听融合方法，经过训练既可以产生干净的信号，又可以重建视频。这两篇论文均使用带噪输入信号的相位作为干净相位的近似值。但是，这些方法的局限性在于它们仅在受限条件下（例如[28]中由一组固定短语组成的话语）展示，或仅在培训过程中出现的少数说话者展示。

我们的方法在以下几个方面与这些工作有所不同：（i）我们不会将频谱图视为图像，而是将其视为时间信号，并将频点作为通道；这使我们能够建立一个具有大量的参数可以快速训练的更深的网络；（ii）我们生成了一个软mask进行过滤，而不是直接预测干净的幅度，因为我们发现这更有效； （iii）我们包括一个相位增强子网；最后，（iv）我们在以前听不见（和看不见的）的演讲者和通俗视频中进行测试。

在并发且独立的工作中，[29]开发了基于dilated卷积和双向LSTM的类似系统，在不受约束的环境中展示了良好的结果，而[30]则训练了用于视听同步的网络并成功使用其语音分离功能。

这里提出的增强方法是对唇读[31，32，33]的补充，该方法也已被证明可以改善嘈杂环境中的ASR性能[34，35]。

## 2. Architecture

本节介绍了视听语音增强网络的输入表示形式和体系结构。网络摄取音频视频数据的连续剪辑。图2中详细给出了模型架构。

### 2.1 Video representation

视觉特征是使用类似于[33]提出的时空残差网络从输入图像帧序列中提取的，其中网络在字级唇读任务上进行了预训练。该网络包括一个3D卷积层，然后是一个18层ResNet [36]。网络为每个视频帧输出一个紧凑的512维特征向量$f_0^v$（其中下标0表示视听网络中的层号）。由于我们在具有预先裁剪的面孔的数据集进行训练和评估，因此除了转换为灰度和适当的缩放比例外，我们不执行任何其他预处理。

### 2.2. Audio representation

使用具有Hann窗函数的短时傅立叶变换（STFT）从原始音频波形中提取声学表示，这会生成幅度和相位谱图。STFT参数的计算方式类似于[27]，因此输入序列的每个视频帧都对应于所生成频谱图的四个时间片。由于视频的速率为25fps（每帧40ms），因此我们选择在16Khz的采样率下，跳跃长度为10ms，窗口长度为40ms。所得频谱图的频率分辨率$F = 321$，表示从0到8 kHz的频率，时间分辨率$T\approx\frac{T_s}{hop}$，其中$T_s$是时域信号的长度。幅度谱图和相位谱图分别表示为$T\times321$和$T\times642$张量，其实部和虚部均沿频率轴连接。我们先将幅值转换为80个频点的mel频谱图，然后再将输入到幅值子网络中，然而我们会对原始的线性频谱图进行滤波。

### 2.3. Magnitude sub-network

视觉特征序列$f^v_0$由10个卷积块的残差网络处理。每个块由一个kernel size为5且stride为1的时序卷积组成，然后进行ReLU激活和bn。shortcut连接将块的输入添加到卷积的结果中。采用相似的5个卷积块的堆栈来处理音频流。卷积是沿着时间维度执行的，将有噪声的输入频谱图$M_n$的频率视为通道。两个中间块执行stride为2的卷积，将时间维度整体向下采样4，以便将其降低到视频流分辨率。这些层的skip连接通过使用stride为2进行average pooling来进行下采样。音频和视频流然后在通道维度上串联：$f^{av}_0 = [f^v_{10};f^a_5]$。融合张量通过另一个15个时间卷积块的堆栈。由于我们希望输出mask具有与输入幅度谱图相同的时间分辨率，因此我们包括两个转置卷积，每个卷积对时间维度的上采样系数为2，导致总因子为4。融合输出通过位置卷积投影到原始幅度谱图维度上，并通过SigMoid激活，以输出值为0到1的mask。结果张量逐元素乘以嘈杂的幅度谱图以产生增强的幅度：
$$
\hat{M}=\sigma(W_m^Tf_{15}^{av})\odot M_n
$$

### 2.4. Phase sub-network

我们对相位增强子网的设计的直觉是，语音中的结构会引起幅度和相位频谱图之间的相关性。与幅度一样，我们仅尝试预测细化噪声相位的残差，而不是尝试从头开始预测干净相位。因此，相位子网以噪声相位和幅度预测为条件。通过线性投影和串联将这两个输入融合在一起，然后由6个时间卷积块的堆栈进行处理，每个块有1024个通道。通过将结果投影到相位谱图的维上来形成相位残差，然后将其添加到噪声相位中。通过对结果进行L2归一化，最终获得预测的纯净相位：
$$
\begin{array}{l}
\phi_{6}=\underbrace{\text {ConvBlock}\left(\ldots \text {ConvBlock }\left(\left[W_{m \phi}^{T} \hat{M} ; W_{n \phi}^{T} \Phi_{n}\right]\right)\right)}_{\times 6} \\
\hat{\Phi}=\frac{\left(W_{\phi}^{T} \phi_{6}+\Phi_{n}\right)}{\left\|\left(W_{\phi}^{T} \phi_{6}+\Phi_{n}\right)\right\|_{2}}
\end{array}
$$
在训练中，使用小值和零偏差初始化层的权重，以便初始残差几乎为零，并且有噪相位传播到输出。

### 2.5 Loss Function

通过最小化预测幅度谱图和真实幅度之间的L1损失来训练幅度子网。通过最大化预测相位和真实相位之间的余弦相似度来训练相位子网，并根据真实幅度对其进行缩放。总体优化目标：
$$
\mathcal{L}=\left\|\hat{M}-M^{*}\right\|_{1}-\lambda \frac{1}{T F} \sum_{t, f} M_{t f}^{*}<\hat{\Phi}_{t f}, \Phi_{t f}^{*}> \tag{1}
$$

## 3. Experiments

### 3.1. Datasets

该模型在两个数据集上进行训练：第一个是BBC-牛津嘴唇阅读句子2（LRS2）数据集[34，39]，其中包含来自BBC节目（例如Doctors和EastEnders）的数千个句子；第二个是VoxCeleb2 [40]，其中有6,000多个不同的演讲者说出了超过一百万个句子。

LRS2数据集按广播日期分为训练集和测试集，以确保集之间没有重叠的视频。数据集涵盖了大量的说话者，这鼓励了训练后的模型与说话者无关。但是，由于数据集没有提供身份标签，因此在集合之间可能会有一些重叠的说话者。数据集提供了真实转录，这使我们能够对所生成音频的清晰度进行定量测试。

VoxCeleb2数据集缺少文本转录，但是通过识别将数据集分为训练集和测试集，这使我们能够明确测试模型是否与说话者无关。

这些数据集上的音频和视频已正确同步。通过使用[41]中描述的步骤进行预处理，可以对不是这种情况的视频进行评估（例如，TV broadcast），以检测和跟踪活动的说话人并同步视频和音频。

### 3.2. Experimental setup

我们研究了在干净信号上添加1至4个额外的干扰说话人的情况，因此我们生成的信号总共有2至5个说话人。应该注意的是，分离多个具有相等平均“响度”的说话人的语音的任务比分离语音信号和背景babble噪声的挑战更具挑战性。

### 3.3. Evaluation protocol

我们使用[42]中描述的盲源分离标准评估模型在感知语音质量方面的增强性能（我们使用[43]提供的实现）。信号干扰比（SIR）可以测量抑制不想要的信号的程度，信号伪像比（SAR）可以说明增强过程引入了伪像，而信号失真比（SDR）是整体质量衡量，两者都考虑在内。我们还报告了PESQ [44]的结果，PESQ [44]测量了整体感知质量，STOI [45]与信号的清晰度相关。从上面提出的指标来看，PESQ已被证明是与听力测试最相关的一种，可以说明相位失真[46]。此外，我们使用ASR系统来测试增强语音的清晰度。为此，我们使用GoogleSpeech识别界面，并在干净，混合和生成的音频样本上报告单词错误率（WER）。

### 3.4. Training

按照[33]，我们在单词级唇读任务上对时空视觉前端进行了预训练。这分为两个阶段：首先，对LRW数据集进行训练[31]，该数据集涵盖了近额姿势。然后在一个类似大小的内部多视图数据集上。为了加快后续训练过程，我们冻结了所有视频的前端，预先计算并保存了视觉特征，还计算并保存了纯净和噪声音频的幅值和相位谱图。

训练分三个阶段进行：首先，对幅度预测子网进行训练，按照从高SNR输入（即仅增加一说话言人）开始，然后逐步发展为具有更多说话人的更具挑战性的示例;其次，幅值子网络被冻结，只训练相位网络。最后，整个网络是端到端的微调。我们没有用超参数平衡幅度和相位loss项进行实验，而是将其设置为$\lambda=1$。

为了生成训练示例，我们首先通过随机采样一个60帧干净片段来选择视觉和音频特征$(v_r,a_r)$的参考对，确保音频和视觉特征相对应并正确对齐。然后，我们采样$N$个噪声频谱图$x_n,n\in[1,N]$，然后通过对在频域中复频谱求和将他们与参考频谱混合，得到混合频谱$a_m$。这是增加我们的训练数据的自然方法，因为每次都会对嘈杂的音频信号进行不同的组合采样。在添加噪声样本之前，我们将其能量标准化为参考信号的能量：$a_{m}=a_{r}+\sum_{n} \frac{r m s\left(a_{r}\right)}{r m s\left(a_{n}\right)} a_{n}$。

### 3.5. Results

LRS2。我们在表1中的LRS2数据集的测试集上总结了我们的结果。针对以下信号类型列出了在不同指标下的性能：混合信号作为基线，以及使用我们的网络预测的幅度和真实相位、预测的相位、使用GriffinLim算法近似的相位，混合信号相位获得的重构。从预测幅度和预测相位重建的信号就是我们认为网络的最终输出。
使用真实相位时的评估包括在相位预测的上限中。从混合信号的所有测量可以看出，随着添加更多说话人，任务变得越来越困难。总体而言，BSS指标和PESQ都与我们的观察结果紧密相关。有趣的是，虽然增加了更多的说话人，但SIR大致保持不变，但是引入了更大的失真。该模型在抑制输出中的串扰方面非常有效，但是这样做会牺牲目标语音的质量。
我们的网络预测的相位要比混合相位要好。即使改进的数量相对较小，但语音质量的差异也很明显，因为显着降低了具有不同步谐波的“robotic”效应。我们鼓励读者听补充材料中的样本，以便更好地理解这些差异。但是，与真实相位测试的性能存在较大差距，这表明相位网络还有很大的改进空间。
使用Google ASR的转录结果也符合这些发现。特别值得一提的是，我们的模型能够从人为或ASR系统无法理解的嘈杂音频中产生非常清晰的结果。
尽管内容主要由幅度决定，但是当使用更好的相位近似时，我们看到WER有了很大的改善。有趣的是，尽管使用Griffin Lim（GL）算法获得的相位在客观指标上的表现明显差强人意，但它展示了相对较强的WER结果，在5个人同时发言的情况下，甚至略微超出了预测相位。
VoxCeleb2。为了明确评估我们的模型是否可以概括为训练期间看不见的说话者，我们还使用与说话者身份无关的训练和测试集VoxCeleb2对网络进行微调和测试。结果总结在表2中，我们展示了三说话人场景下的实验。我们还包括使用SAR和STOI指标进行评估。总体而言，该性能与LRS2数据集相当，但略逊一筹，与定性性能一致。这可能归因于视觉功能未进行微调，以及VoxCeleb2中存在许多其他背景噪音。结果证实，该方法可以推广到看不见（和听不到）的说话者。
表格的最后一列显示了对在LRS2上训练的原始模型的PESQ评估，而没有对VoxCeleb进行任何微调。该性能比微调模型差，但是显然可以。由于LRS2仅限于讲英语的人，而VoxCeleb2包含多种语言，这表明该模型学会了归纳训练期间未见到的语言。

### 3.6. Discussion

相位细化：对我们的整个网络进行端到端培训可以减少相位损失，这可能表明包含视觉功能也可以改善相位增强。但是，要确定是否成立以及在何种程度上做到这一点，还需要进行深入的研究，以供将来工作。
AV同步。我们的方法对语音和视频之间的临时对齐非常敏感。我们使用SyncNet进行对齐，但是由于该方法在极端噪声下可能会失败，因此我们需要在模型中建立一些不变性。以后的工作将被纳入模型中。

## 4. Conclusion

在本文中，我们提出了一种使用目标说话人嘴唇上的视觉信息将目标说话人的语音信号与背景噪声和其他说话人分离的方法。深层网络通过预测目标信号的相位和幅度来产生逼真的语音片段；我们还证明了该网络能够从在“野外”环境中不受限制地录制的非常嘈杂的音频片段中生成可理解的语音。

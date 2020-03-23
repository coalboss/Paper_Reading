# My lips are concealed: Audio-visual speech enhancement through obstructions

## Abstract

我们的目标是建立一个视听模型，以将单一说话人与混合语音(例如其他说话人和背景噪声) 分开。此外，即使由于遮挡而暂时没有视觉提示时，我们也希望听到说话人的语音。

为此，我们引入了一个深层的视听语音增强网络，该网络可以通过已知说话人的嘴唇运动和/或他们声音的表示来区分说话人的声音。可以通过(i) enrollment或(ii) self-enrollment获得声音表示-在足够的无障碍视觉输入的情况下实时学习表示。该模型是通过混合音频，在嘴巴区域周围引入人工遮挡来防止视觉模态占主导地位而进行训练的。

该方法与说话人无关，并且我们在训练过程中未曾听说过(并且看不见) 的说话人的真实示例中进行了演示。该方法也对以前的模型进行了改进，特别是在视觉模态闭塞的情况下。

Figure 1：当嘴唇区域被麦克风等遮挡时，视听语音增强模型可能会失败。在这种情况下，输入音频通常被完全过滤掉，结果是在遮挡的帧上无声输出。我们的方法的目的是对各种遮挡具有鲁棒性。

Figure 2：视听语音增强网络的结构：有2个音频流。一个处理传入的带噪音频，而另一个则将注册音频样本作为输入，并创建一个说话人嵌入，以捕获说话人的声音特性。视觉流从输入视频中提取帧表示。视觉、说话人和音频嵌入被组合并馈入BLSTM，BLSTM输出一个乘法掩模，用于过滤带噪频谱图。当没有提供注册音频时，增强幅度（由纯视频创建）可以用作说话人嵌入网络的输入。

Table 1：架构细节。a）处理视频特征的1D ResNet。b）处理带噪音频频谱的1D ResNet。c）执行模态融合的BLSTM和FC层。表示法：$K$：核宽度；$S$：步长-分数步长表示转置卷积；$P$：填充；$Out$：层输出的时间维度。非转置卷积层都是深度可分的。在每一个卷积层后加入BN、ReLU激活和跳跃连接。

Figure 3：训练和评估期间使用的遮挡视频的示例帧

Table 2：对来自LRS3数据集的2个和3个同时说话人的样本的语音增强性能的评估。所有样本的长度为8秒，其中使用遮挡时遮挡6秒（两边各3秒）。Tr.Occ:（训练遮挡）表示模型已使用人工遮挡进行训练；T.Occ:（测试遮挡）表示使用遮挡进行评估；pre:预注册：注册音频来自目标说话人的另一个视频；self：自我注册；SDR：信号失真率（越高越好）；WER：现成ASR系统的字错误率（越低越好）

Figure 4：当遮挡2个说话人和3个说话人场景的视觉输入不同数量的帧时的增强性能。表2的标题中解释了模型符号。

## 1. Introduction

尽管近年来自动语音识别(ASR) 领域取得了长足进步，但仍然存在一些关键挑战，特别是在嘈杂的环境中或在多个人同时讲话的情况下如何理解语音。在这个方向上，在多说话人场景中隔离语音，增加嘈杂音频中的信噪比，或者两者的结合都是重要的任务。

直到最近，该领域的工作仅使用音频形式完成任务。但是，最近的工作表明，视频的使用可以极大地帮助解决问题[1,2,3]。

这些视听模型显示出令人印象深刻的结果，但是由于它们依赖于视觉输入，当说话人的手，麦克风(例如图1) 阻塞嘴巴区域或说话人转过头时，它们可能会失败。已经证明，说话人声音的embedding可以指导同时语音的分离[4]。

在本文中，我们提出将两种方法结合起来，即以包含说话人的嘴唇移动的视频输入和他们的声音embedding为条件，以使视听模型对遮挡具有鲁棒性。我们的假设是，视频在存在时可提供宝贵的判别信息，而当由于遮挡而视频不存在时，说话人embedding可以为模型提供帮助。在最简单的情况下，可以从预先注册的音频中获取声音embedding。

虽然可以仅使用音频[5、6]来分离同时发言的说话人，但时域中的排列问题仍然是一个未解决的问题。通过我们的方法，即使是部分遮挡的视频也可以提供有关说话人语音特性的信息，并解决了将分离的语音分配给说话人的歧义。

我们做出了以下贡献：(i) 我们展示了如何结合说话人embedding和视觉提示，以使单个说话人与混合语音分离，尽管视觉流(嘴唇) 被遮挡了; (ii) 我们提出了一种神经网络模型，该模型用于仅视频，用于仅注册数据或同时用于两种方法; (iii) 我们引入了一个递归模型，该模型可以在临时闭塞情况下引导说话人embedding的计算，而无需事先进行说话人embedding。

### 1.1 Related Work

**仅音频的增强和分离** 已经提出了各种方法来隔离多讲话人同时语音，其中大多数仅使用单声道音频，例如[7、8、9、10、11]。最近的许多工作已经解决了排列问题，以将看不见的说话人分开。深度聚类[5]使用经过训练的嵌入来生成理想的成对亲和力矩阵的低秩近似，而Yu等人则采用一个排列不变损失函数[6]。

**视听语音增强** 在深度学习出现之前，已经开发了许多用于视听语音增强的作品[12、13、14、15、16、17]。最近的几种方法已将深度学习框架用于同一任务-最著名的是[18，19，20]。但是，这些方法的局限性在于它们仅在受限条件下（例如，话语由固定的一组短语组成）或少数已知的讲话者才能证明有效。我们先前的工作[1]提出了一种深层的视听语音增强网络，该网络可以通过预测目标信号的幅度和相位来分离相应视频中给定嘴唇区域的说话人语音。Ephrat[2] 设计了一个网络，该网络以所有源说话人的视频输入为条件，并输出复数的mask，从而增强了幅度和相位。Owens和Efros[3]在视听同步方面训练了一个网络，并将学到的特征用于说话人分离。这些最后的作品在户外展示了泛化结果。

**通过仅以声音为条件进行增强** Wang等人[4]开发了一种方法，可以将以预先学习的说话人embedding为条件的语音进行分离，这表明仅声音特性就足以确定分离。但是，这依赖于预先训练的模型，并且不使用视频。

我们建议将这两种想法结合起来：使用来自目标说话人的视觉输入和声音embedding；我们的方法部分基于[1,2,4]。

## 2. Method

本节描述了图2所示的视听语音增强网络的体系结构。该网络接收三个输入：（i）要增强的嘈杂音频； （ii）相应的视频帧； （iii）包含来自目标说话人的参考音频。我们总结以下主要模块。表1中提供了该体系结构的详细信息。

**视频表示** 网络的输入是预先裁剪的图像帧，例如在LRS数据集中发现的面部数据[21,22]。使用[23]中描述的时空残差网络从图像帧序列中提取视觉特征。该网络包含一个3D卷积层，然后是一个普通的18层2D ResNet [24]。对于每个视频帧，它输出一个紧凑的512维特征向量。

**音频表示** 作为声学特征，我们使用通过短时傅立叶变换（STFT）从音频波形中提取的幅度和相位谱图，其窗长为25ms，帧移为10ms，采样率为16kHz。这将导致频谱图的时间维度是相应视频帧数的四倍。我们使用T/4和T分别表示视频帧的数量和频谱图的相应时间分辨率。

**说话人嵌入网络** 为了将参考音频片段嵌入到紧凑的说话人表示中，我们使用Xie等人的方法[25]。为了减少计算量，我们将所有2D空间卷积替换为将频点作为通道的1D时空变量，并在后续步骤中用VoxCeleb2 [26]数据集预先训练了修改后的架构[25]。

**模态组合** 如图2所示，通过浅的时序ResNet将带噪声的幅度谱图编码为音频特征向量。视频特征通过包含两个转置卷积层的网络进行上采样，以匹配频谱图的时间维度（4T）。从参考音频中提取的说活人嵌入会在时间上进行展平，然后添加到生成的视频嵌入中，以形成用于增强的条件向量。然后将此矢量与带噪的音频嵌入一起馈入一层双向LSTM，然后是两个完全连接的层。输出具有频谱图尺寸，并通过$\operatorname{Sigmoid}$激活以产生增强mask。

**相位子网** 为了调整带噪相位以适应增强的幅度，我们使用[1]的相位网络而没有任何变化。

**自我注册** 对于自我注册，幅度网络运行两次：在第一次通过时，没有将说话人嵌入添加到视觉对象中。然后，输出的幅度将用作说话人嵌入网络的输入，如红色反馈箭头所示，并且网络将第二次运行，这次伴随说话人嵌入。

我们最小化学习目标[1]：
$$
\mathcal{L}=\|\hat{M}-M^{*}\|_{1}-\frac{1}{T F} \sum_{t, f} M_{t f}^{*}<\hat{\Phi}_{t f}, \Phi_{t f}^{*}>
$$
其中$\hat{M}$，$\hat{\Phi}$和$M^*$，$\Phi^*$分别是预测的和基准真实振幅和相位谱图，以及$T$和$F$是它们的时间和频率分辨率。

## 3. Experimental Setup

**数据集** 网络在MV-LRS [27]，LRS2 [21]和LRS3 [22]数据集上进行训练，并在LRS3上进行了测试。MV-LRS和LRS2包含来自英国电视广播的资料，而LRS3是根据TED演讲的视频创建的。据我们所知，出现在LRS3中的发言人是在其他两个数据集中都没有看到的。数据集共享相同的格式和流水线（包括面部检测步骤），因此不需要预处理就可以将它们一起用于训练。我们从LRS3训练集中删除了也出现在测试集中的少数说话人，因此两者之间没有重叠的身份。因此，该测试集仅包含在训练过程中看不见和听不到的说话人，并且适合于对我们的方法进行说话人不可知性评估。此外，由于LRS3的测试集包含相对短的句子，因此为了进行测试，我们从生成LRS3测试集的原始材料中提取了更长的子序列进行。我们仅使用来自至少出现在2个不同视频（TED演讲）中的说话人样本，来确保以与目标不同的设置来注册录制的音频。这些额外的视频以及增加的噪音和遮挡已在项目网站上公开提供。

**合成数据** 我们首先通过从训练数据集中采样一个参考音视频句子，然后将其音频与干扰音频信号混合，来生成与其他作品[1、2、4]类似的合成示例。我们考虑了两个场景：2个说话人和3个说话人，其中一个和两个干扰语音分别添加到目标信号。

**注册** 在训练期间，我们不知道发言人的身份。因此，我们从同一视频但不同的非重叠时间段获得注册信号，这有效地减少了我们需要丢弃较短视频的数据量（例如，如果我们使用3秒，则至少需要视频有6秒长）。我们使用这种方法对说话人身份未知的数据集进行训练。

在评估过程中，我们尝试了两种注册方法：（i）预注册–我们从同一发言人的视频中抽样一个注册片段，该视频与用于创建目标样本的视频不同（我们确实具有测试集的身份标签）； （ii）自我注册-我们通过不使用说话人嵌入的网络获得注册附件，如第2节所述。

**遮挡** 为了进行训练，我们以随机补丁的形式人为地将遮挡添加到视频帧中，如图3a所示。我们随机遮挡15到25个连续帧的子序列，将清晰-遮挡的帧比例保持在1：3。这比简单地将传入的视觉帧归零更为现实，因为被遮挡的视频帧仍会产生有效的特征向量。为了进行评估，我们在视频上放置了抖动表情符号，而不是随机补丁，如图3b所示。在训练过程中没有看到这种视觉噪声。表情符号用于在头和尾遮挡视频，而句子的中间则保持清晰。

**训练** 时空视觉前端是在单词级唇读任务中预先训练的[23]。然后，我们冻结前端并预先计算视觉特征。这些特征是在我们添加了随机遮挡的视频版本中提取的。

训练分四个阶段进行。我们首先仅使用说话人嵌入输入对幅度子网络进行预训练，为此，我们首先使用两个说话人的混合然后是三个说话人的混合;其次，添加视觉模态，并针对三个说话人同时说话的场景的保存的视觉特征对幅度网络进行训练；第三，幅值网络被冻结，相位网络被训练；最终，整个网络进行端到端的训练。

## 4. Experiments

### 4.1. Evaluation protocol

为了评估我们模型的性能，我们使用信号失真比（SDR）[28]，这是一种通用指标，表示目标信号的能量与增强输出中包含的误差的能量之间的比率。此外，为了评估输出的可理解性，我们使用了Google Cloud ASR系统-我们计算了ASR系统对增强音频的预测与句子的真实转录之间的单词错误率（WER）。这被包含在评估的部分。我们对8秒（200帧）的固定长度视频片段进行评估。附录中报告了一些其他的性能度量。定性的例子可以在项目网站上找到<http://www.robots.ox.ac.uk/~vgg/research/concealed/>。

### 4.2. Baseline models

我们将我们提出的方法与下面的基线和消融进行比较，我们在有或无视觉遮挡的情况下对其进行训练和评估。

**PIT** 我们实现了一个盲源分离模型，该模型利用图2的带噪音频输入流，并在文[6]的基础上进行了排列不变损失函数的训练。这种模型是按预先确定的说话人数量定制的。

**V-Conv** 这是一条来自Afouraset[1]的卷积的，视觉条件的基线。该模型使用一系列的一维卷积块来融合音频和视频模态，而不是BLSTM。此外，在我们提出的模型中，视频流不会对视频特征进行上采样，而是在视频帧的时间分辨率下进行视听融合。然后，一维卷积堆将融合输入向上采样到谱图的维数。

**V-BLSTM** 该模型与我们提出的架构相似，但条件仅限于视频特性。

**VoiceFilter** 该模型只以说话人嵌入为条件，与训练过程第一阶段使用的子网络等效。它本质上是VoiceFilter[4]的实现，具有稍加修改的体系结构，在我们的数据集上进行训练。

**VS** 我们提出的架构，它同时接收视频和说话人嵌入输入。如第2节所述，我们调查了两种变体，VS-pre和VS-self，即与评估期间采用的不同注册方法相对应。

### 4.3. Results

我们将实验结果总结在表2中。当不使用遮挡时，v-blstm模型仅轻微的优于v-Conv的性能。在80%的视觉输入帧被遮挡时，没有经过遮挡训练的模型则失败。即使在v-Conv训练过程中包括遮挡，它也不能处理丢失的视觉信息，因为它的感受野是有限的（两边各1秒左右）。相反，V-blstm利用其记忆并学习处理局部闭塞。然而，总的来说，所提出的VS模型明确地以期望的说话人嵌入为条件，给出了最佳的性能。

结果进一步证实，当使用来自不同于目标源的注册信号进行评估时，voicefilter和vs-pre模型都表现良好，即使他们从未在该设置中接受过训练。

图4研究了遮挡不同数量的视觉输入的效果。当视频输入的很小一部分被遮挡时，在没有遮挡上训练的vblstm模型就不能很好地工作。当使用遮挡进行训练时，V-blstm变得更有弹性，但是，它仍然会为高遮挡率和整个视频被遮挡分别带来不良的结果和完全失败。

当一半或更多的输入被阻塞时，vs-pre模型的性能优于V-BLSTM，并且对于更干净的输入给出了类似的结果。

自注册：对于非常高的遮挡水平，VS-self的初始增强估计很差，显然无法捕获目标声音特征。但是，如果超过20％的帧是干净的，则自注册性能最佳。因此，除了较高的遮挡级别之外，具有自注册功能的VS与V-BLSTM相比更具优势。

## 5. Conclusion

本文提出了一种深度音视频语音增强网络，该网络通过以说话人的嘴唇运动和/或声音的表达为条件来分离说话人的声音。该网络对部分遮挡具有鲁棒性，当无法获得用于预注册的段时，声音表示可以从输入的未遮挡部分进行自注册。这些方法是在具有挑战性的LRS3数据集上进行评估的，当视频输入部分被遮挡时，它们的性能超过了先前的最新水平[1]。
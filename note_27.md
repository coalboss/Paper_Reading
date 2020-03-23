# Face Landmark-based Speaker-Independent Audio-Visual Speech Enhancement in Multi-Talker Environments

这是一篇由意大利University of Modena and Reggio Emilia和Istituto Italiano di Tecnologia的研究者发布的在受限大小的GRID和TED-TIMIT数据集上实现speaker-independent speech enhancement的首创性技术论文。

本文的主要贡献有2:

1. 第一个以GRID和TED-TIMIT这种大小受限数据集构建的speaker-independent multi-talker数据集上训练和测试模型并达到和之前报道的speaker-dependent的模型可比的性能。
2. 使用人脸特征点运动向量作为视觉特征。

数据集：
GRID:33个说话人被分成train：verify：test=25:4:4 （有一个说话人没有视频所以不用）
从每个说话人的1000句话中选出来200句，每句话与不同的说话人的3句话混合生成3个mixture样本。
TED-TIMIT：59个说话人被分成train：verify：test=51：4：4，每个说话人有98句话，全部用。和GRID类似的生成过程，但要求两句话的长度差不大于2s，且按照target speaker的长度对齐。

网络:

1. VL2M
2. VL2M_ref
3. AV concat
4. AV concat_ref

VL2M
5层BLSTM，每层250个节点
输入是从视频中提取的人脸特征点运动向量，输出是对Target Binary Mask的估计，loss是两个mask之间算交叉熵。

VL2M_ref


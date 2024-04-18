# Group_文献分享



[TOC]

## 目的

集思广益：收集同学们每周的文献分享内容👀  方便组内其他同学学习相应文献与交流



## 分享指南 

按照组内目前研究方向，大家可以把自己每周分享的内容或平时读到比较好的文献按照六个方向进行上传：

- **大模型安全**
- **大模型机理研究**
- **LLM4science**
- **大模型数据增强**
- **多模态大模型**
- **其他文献**



**分享建议** ：在上传时可以把自己想要上传的内容打包成文件夹（name: 文献/项目 名）push在对应的类别中，同时在相应类别内md文件中简要补全分享文献/项目的基本信息（方便其他同学快速定位查阅）：
![alt text](LLM_research.jpg)

文献名/背景/解决问题/文献（项目）领域贡献/实验设计/创新点/数据集（代码）/可借鉴点与不足【以上若没有则不填】

### example：Attention is All you Need

- 背景：介绍了一种基于注意力机制的新型神经网络模型——Transformer，可以用于序列转换问题，如机器翻译。与传统的基于循环神经网络和卷积神经网络的模型不同，Transformer完全依赖于注意力机制，消除了循环和卷积的使用，具有更高的并行性和更短的训练时间。
- 解决问题：解决了在机器翻译任务中，传统序列转换模型的计算效率较低、难以并行化的问题。作者提出了一种新的网络架构，即Transformer，仅基于注意力机制，不涉及循环和卷积操作。
- 领域贡献：提出了一种新的神经网络架构，即Transformer，它完全基于注意力机制，不涉及循环和卷积操作。Transformer在机器翻译任务上表现出了更好的性能，同时具有更高的并行性和训练效率
- 实验设计：实验主要是针对机器翻译任务进行的。作者在WMT 2014的英德和英法翻译任务上进行了实验，使用了基于BLEU分数（翻译质量评估指标）的评估方法。作者分别使用了base和big两种模型进行实验，其中big模型使用了更多的训练时间和计算资源。实验中还对不同的模型参数进行了调整和比较，如注意力头数、注意力维度、学习率等等，以确定最佳的模型配置。此外，作者还将Transformer模型应用于英语句法分析任务，以验证其在其他任务上的泛化能力。
- 数据集：用于定量评估的数据集是WMT 2014英德翻译任务的数据集，包含约450万个句子对。代码已经开源，https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor上找到。)
- 可借鉴点与不足：两个可以继续研究的方向，其中之一是对于具有大输入和输出的问题，如图像、音频和视频，进行局部的、限制性的注意力机制，以提高效率和减少计算时间。此外，可以尝试对于生成的输出进行预测，并在预测正确后进行下一个预测，以减少生成的时间。另一个可能的改进方向是对于输入和输出的结构进行更多的研究，以提高模型的性能和泛化能力。



**欢迎同学们贡献文献学习项目！**⚒️

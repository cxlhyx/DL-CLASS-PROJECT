# 深度学习课程项目

凌乐昊  黄堉轩  罗建林

### 选题八

VideoBooth源代码来自：https://github.com/Vchitect/VideoBooth

我们的工作：其中，fvd_clip_test.py文件是我们自己编写的测试指标代码，mydataset.ipynb文件是我们用来构造数据集
的代码，构造文本-图像-视频数据集的具体步骤在课程报告的4.1节中有提及。

我们还在models/attention.py中引入了相对位置编码，相关代码在26-49行，以及600-643行。

除此以外，我们还做了一些未在课程报告中提及的尝试，比如添加training-free的帧间增强模块Enhance_Video(原项目：https://github.com/NUS-HPC-AI-Lab/Enhance-A-Video)
相关代码在models/attention.py的383行到475行，以及enhence.py文件

### 预训练模型
由于GitHub内存有限，这里我们没有上传预训练权重。
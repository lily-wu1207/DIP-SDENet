# SDENet
### 说明：
本文基于CounTR,提出了基于尺度动态缩放的编码网络（Scale Deformable Embedding Network, SDENet）。SDENet网络采用尺度可动态缩放的自注意力模型作为查询图像的编码网络，对范例图像进行特征增强以适应类内差异，采用交互注意力作为相似度对比模块，并利用可学习的残差结构解决网络退化问题。
### 目录结构：
data: 数据预处理  
model: 存放模型，其中SDCAT_augment为最终版本  
其余作为ablation study: SDCAT不含范例框特征增强、SDCAT_noDAT不含动态尺度缩放注意力，SDCAT_noRes不含可学习残差连接。  
FSC_pretrain.py：预训练
train.py：训练
### 运行：
1. 下载数据集FSC147，在运行时并设置--dataset "路径名"  
2. 运行FSC_pretrain.py进行预训练  
3. 运行train.py进行训练，根据需要调整其中的参数。

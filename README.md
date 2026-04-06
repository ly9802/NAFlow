# Neuron Abandoning Attention Flow

This repository contains source code necessary to generate the visual attention flow throughout a CNN during decison-making in image classification task for this paper
[Neuron Abandoning Attention Flow: Visual Explanation of Dynamics inside CNN Models](https://ieeexplore.ieee.org/abstract/document/11342296) (Transaction on Pattern Analysis and Machine Intelligence, 2026). If you use the code in this paper, please considier citing:

<pre>
@article{yiliaoTPAMI2026NAFlow,
  author={Liao, Yi and Gao, Yongsheng and Zhang, Weichuan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Neuron Abandoning Attention Flow: Visual Explanation of Dynamics Inside CNN Models}, 
  year={2026},
  volume={48},
  issue={5},
  pages={5536-5551},
  doi={10.1109/TPAMI.2026.3651260}}
  publisher={IEEE}
}
</pre>

## The Environment Setup
To run the code successfully, please download the software **Anaconda3** from the [official website](https://www.anaconda.com/download/success) and install it in advance. 

1. After installation, open a **Command Prompt** (Windows) or **Terminal** (Linux), input the following command,
<pre>
conda create -n naflow python==3.8
</pre>

2. Please activate the enrionment `naflow` by running
<pre>
conda activate naflow
</pre>

3. Please install the pytorch framework and the below libraries by runing the following commands in sequence,
<pre>
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python -m pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
conda install -c anaconda pillow scikit-learn pandas seaborn cython dbf
conda install -c conda-forge h5py timm einops yacs cvxpy nested_dict
</pre>

4. Please navigate the directory where the main.py exists, and then run the program by the following command,
<pre>
python main.py
</pre>

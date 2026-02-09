(.venv) root@dsw-1618740-fb57ccd9f-zfbcm:/mnt/workspace# python --version
Python 3.11.11
(.venv) root@dsw-1618740-fb57ccd9f-zfbcm:/mnt/workspace# python - << 'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
PY
2.9.1+cu128
12.8
91002
(.venv) root@dsw-1618740-fb57ccd9f-zfbcm:/mnt/workspace# pip show mmengine
pip show mmsegmentation
Name: mmengine
Version: 0.10.7
Summary: Engine of OpenMMLab projects
Home-page: https://github.com/open-mmlab/mmengine
Author: MMEngine Authors
Author-email: openmmlab@gmail.com
License: 
Location: /mnt/workspace/tactile_topo_seg/.venv/lib/python3.11/site-packages
Requires: addict, matplotlib, numpy, opencv-python, pyyaml, rich, termcolor, yapf
Required-by: mmcv
Name: mmsegmentation
Version: 1.2.2
Summary: Open MMLab Semantic Segmentation Toolbox and Benchmark
Home-page: https://github.com/open-mmlab/mmsegmentation
Author: MMSegmentation Contributors
Author-email: openmmlab@gmail.com
License: Apache License 2.0
Location: /mnt/workspace/tactile_topo_seg/.venv/lib/python3.11/site-packages
Requires: matplotlib, numpy, packaging, prettytable, scipy
Required-by: 
(.venv) root@dsw-1618740-fb57ccd9f-zfbcm:/mnt/workspace# ## CUDA / Driver
nvidia-smi
Thu Feb  5 13:05:39 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.08             Driver Version: 550.127.08     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10                     Off |   00000000:00:08.0 Off |                 Off* |
|  0%   27C    P8             10W /  150W |       1MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
(.venv) root@dsw-1618740-fb57ccd9f-zfbcm:/mnt/workspace# cd /mnt/workspace




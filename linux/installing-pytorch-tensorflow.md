# Installing PyTorch and Tensorflow with CUDA, cuDDN

## ⚠️ CAUTION
- Ubuntu 20.04 version
- Anaconda must be installed

if you have error while installing NVIDIA driver or CUDA, please follow this step
1. ```sudo apt-get purge nvidia*```
2. ```sudo apt-get autoremove```
3. ```sudo apt-get autoclean```
4. ```sudo rm -rf /usr/local/cuda*```

## NVIDIA driver
⚠️Can be passed if you already installed driver(```nvidia-smi``` shows your GPU)

1. select your type and download file in [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)
   - RTX 4070 Ti SUPER
   - Linux 64bit

2. move to dir the file located and make authorization
   ```shell
   chmod +x ./NVIDIA-Linux-x86_64-550.78.run
   ```

3. install by running download file
   ```shell
   sudo sh ./NVIDIA-Linux-x86_64-550.78.run
   ```

4. check
   ```
   nvidia-smi
   ```

   if you see your GPU, you're good to go!
   ```
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 550.78                 Driver Version: 550.78         CUDA Version: 12.4     |
   |-----------------------------------------+------------------------+----------------------+
   | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
   |                                         |                        |               MIG M. |
   |=========================================+========================+======================|
   |   0  NVIDIA GeForce RTX 4070 ...    Off |   00000000:0A:00.0  On |                  N/A |
   |  0%   32C    P8             13W /  285W |     261MiB /  16376MiB |      0%      Default |
   |                                         |                        |                  N/A |
   +-----------------------------------------+------------------------+----------------------+
                                                                                            
   +-----------------------------------------------------------------------------------------+
   | Processes:                                                                              |
   |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
   |        ID   ID                                                               Usage      |
   |=========================================================================================|
   |    0   N/A  N/A      1251      G   /usr/lib/xorg/Xorg                             71MiB |
   |    0   N/A  N/A      1807      G   /usr/lib/xorg/Xorg                             81MiB |
   |    0   N/A  N/A      1937      G   /usr/bin/gnome-shell                           86MiB |
   +-----------------------------------------------------------------------------------------+
   ```

## CUDA
1. download cuda-11.8 version [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   ```shell
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   ```

2. install by running download file
   ```shell
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```
   
   make sure uncheck **```Driver```** below(cause we downloaded NVIDIA driver at first)
   
   ![](../asset/linux/setting-gpu-environment-1.png)
   ![](../asset/linux/setting-gpu-environment-2.png)
   ![](../asset/linux/setting-gpu-environment-3.png)

3. setting path to recognize CUDA
   ```shell
   sudo nano ~/.bashrc
   ```
   
   on the bottom line, enter the following three lines of code
   
   ```shell
   export PATH="/usr/local/cuda-11.8/bin:$PATH"
   export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
   export TF_CUDA_PATHS="/usr/local/cuda-11.8/lib64:$TF_CUDA_PATHS"
   ```
   
   apply setting
   ```shell
   source ~/.bashrc
   ```

4. check
   ```shell
   nvcc -V
   ```

   if you see something like this, you're good to go!
   ```
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2022 NVIDIA Corporation
   Built on Wed_Sep_21_10:33:58_PDT_2022
   Cuda compilation tools, release 11.8, V11.8.89
   Build cuda_11.8.r11.8/compiler.31833905_0
   ```

## cuDDN
1. select CUDA version and download [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)
   - cuDNN v8.6.0   (October 3rd, 2022), for CUDA 11.x - Local Installer for Linux x86_64 (Tar)

3. unzip file
   ```shell
   tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
   ```

4. setting path to copy cuDNN files into CUDA
   ```shell
   sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn*.h /usr/local/cuda-11.8/include
   ```
   
   ```shell
   sudo cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64
   ```
   
   ```shell
   sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
   ```

5. check
   ```shell
   cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
   ```

if you see something like this, you're good to go!
```shell
#define CUDNN_MAJOR 8
#define CUDNN_MINOR 6
#define CUDNN_PATCHLEVEL 0
```

## PyTorch
1. install with conda
   ```shell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
2. install with pip
   ```shell
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
   
3. check
   ```shell
   python
   
   >> import torch
   >> print(torch.cuda.is_available())
      True
   >> print(torch.cuda.device_count())
      1
   >> print(torch.cuda.get_device_name(torch.cuda.current_device()))
      NVIDIA GeForce RTX 4070 Ti SUPER
   ```

## Tensorflow
1. create conda
   ```shell
   conda create -n <env-name> python=3.8
   ```
   
2. install with pip
   ```shell
   pip install tensorflow==2.12
   ```

3. check
   ```shell
   python

   >> from tensorflow.python.client import device_lib
   >> device_lib.list_local_devices()
      [name: "/device:CPU:0"
      device_type: "CPU"
      memory_limit: 268435456
      locality {
      }
      incarnation: 2737954448086927755
      xla_global_id: -1
      , name: "/device:GPU:0"
      device_type: "GPU"
      memory_limit: 14345699328
      locality {
        bus_id: 1
        links {
        }
      }
      incarnation: 6898625890745723943
      physical_device_desc: "device: 0, name: NVIDIA GeForce RTX 4070 Ti SUPER, pci bus id: 0000:0a:00.0, compute capability: 8.9"
      xla_global_id: 416903419
      ]
   ```
   

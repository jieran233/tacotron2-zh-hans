# tacotron2-zh-hans (CPU training version)

http://download.pytorch.org/whl/cpu/torch/

## How to use

(For reference only, it is sorry I can't guarantee that following this step you will get it out of the box.)

- Clone this repo
  
  - ```
    git clone https://github.com/jieran233/tacotron2-zh-hans
    cd tacotron2-zh-hans
    git checkout CPU
    ```
  
- Create conda environment, and install dependencies in it.
  
  - ```
    conda create -n tacotron2-cpu python=3.7
    conda activate tacotron2-cpu
    ```
    
    ```
    # Download PyTorch CPU wheel from:
    # http://download.pytorch.org/whl/cpu/torch/
    
    # Linux http://download.pytorch.org/whl/cpu/torch-1.12.1%2Bcpu-cp37-cp37m-linux_x86_64.whl
    
    # Windows http://download.pytorch.org/whl/cpu/torch-1.12.1%2Bcpu-cp37-cp37m-win_amd64.whl
    
    # macOS(arm64) http://download.pytorch.org/whl/cpu/torch-1.12.1-cp37-none-macosx_11_0_arm64.whl
    # macOS(x86_64) http://download.pytorch.org/whl/cpu/torch-1.12.1-cp37-none-macosx_10_9_x86_64.whl
    
    # Install wheel
    pip3 install ./torch-*.whl
    ```
    
    ```
    # cd to this repo that you cloned just now
    cd tacotron2-zh-hans
    # Install dependencies (tensorflow-cpu~=1.15.0 are included)
    pip3 install -r requirements.txt
    ```

- MKL or OpenBLAS?
  
  - MKL (more suitable for Intel CPU)
    
    See [reference article](https://www.autodl.com/docs/perf/#numpy).
  
  - OpenBLAS (more suitable for AMD CPU)
    
    See [Tutorial for installing numpy with OpenBLAS on Windows - Stack Overflow](https://stackoverflow.com/a/67954011/16719590)
    
    ```bash
    #!/bin/bash
    conda install conda-forge::blas=*=openblas -y
    conda install -c conda-forge numpy -y
    ```

- Prepare your dataset and pertrained model.

- Modify config before training
  
  - File ./hparams.py line 10, 11
    
    Change the values to what you want.
    
    ```python
        epochs = 1000
        iters_per_checkpoint = 500
    ```
  
  - File ./hparams.py line 81
    
    Decrease the batch_size to 32 or lower. (16 is recommended)
    
    If VideoRAM BOMB(out of memory) again, decrease the value of batch_size again.
    
    ```
        batch_size = 16
    ```

- Start training without pretraint model
  
  ```
  python3 train.py --output_directory=./output --log_directory=./log
  ```
  
  - Or continue an unfinished work:
    
    Start training with pretraint model
    
    ```
    python3 train.py -c ckpt.pt --output_directory=./output --log_directory=./log
    ```

## Cleaners

(File ./hparams.py line 30)

'zh_hans_cleaners' (Currently only this one)

- Before:
  
  ```bash
  # cd tacotron2
  # conda activate tacotron2
  # python3 cleaners_test.py <text>
  (tacotron2) [tacotron2-zh-hans]$ python3 cleaners_test.py "CUDA（Compute Unified Device Architecture，统一计算架构[1]）是由英伟达NVIDIA所推出的一种集成技术，是该公司IA的GPU进行图像处理之外的运算，亦是首次可以利用GPU作为C-编译器的开发环境。"
  ```

- After:
  
  ```
  cuda(compute unified device architecture, tong3 yi1 ji4 suan4 jia4 gou4 [1]) shi4 you2 ying1 wei3 da2 nvidia suo3 tui1 chu1 de yi1 zhong3 ji2 cheng2 ji4 shu4 , shi4 gai1 gong1 si1 dui4 yu2 gpgpu de zheng4 shi4 ming2 cheng1 . tou4 guo4 zhe4 ge4 ji4 shu4 , yong4 hu4 ke3 li4 yong4 nvidia de gpu jin4 xing2 tu2 xiang4 chu3 li3 zhi1 wai4 de yun4 suan4 , yi4 shi4 shou3 ci4 ke3 yi3 li4 yong4 gpu zuo4 wei2 c- bian1 yi4 qi4 de kai1 fa1 huan2 jing4 .
  ```

<hr/>

## Training on linux-aarch64 (testing)

for example: `Ubuntu 18.04 LTS [running via Linux Deploy]`

```
sudo apt install wget git

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
chmod +x Miniconda3-latest-Linux-aarch64.sh
./Miniconda3-latest-Linux-aarch64.sh

conda create -n tacotron2-cpu python=3.7 -y
conda activate tacotron2-cpu

wget http://download.pytorch.org/whl/torch-1.12.1-cp37-cp37m-manylinux2014_aarch64.whl
pip3 install torch-1.12.1-cp37-cp37m-manylinux2014_aarch64.whl

wget https://github.com/KumaTea/tensorflow-aarch64/releases/download/v1.15/tensorflow-1.15.5-cp37-cp37m-manylinux_2_24_aarch64.whl
sudo apt install python3-h5py make automake gcc g++ subversion python3-dev pkg-config libhdf5-dev
pip3 install tensorflow-1.15.5-cp37-cp37m-manylinux_2_24_aarch64.whl

git clone https://github.com/jieran233/tacotron2-zh-hans
cd tacotron2-zh-hans
git checkout CPU

# download your dataset
wget https://221-206-125-10.d.123pan.cn:30443/123-847/241ac029/1812443978-0/241ac029b3eba0d1a831c52a6ae5eb4f?v=3&t=1661952511&s=bf07d6f5a3312b4520e28ec116e59f52&i=a8706a7&filename=wenzhi-tricolourlovestory.tar.gz&d=49ac3ed8
mv "241ac029b3eba0d1a831c52a6ae5eb4f?v=3&t=1661952511&s=bf07d6f5a3312b4520e28ec116e59f52&i=a8706a7&filename=wenzhi-tricolourlovestory.tar.gz&d=49ac3ed8" wenzhi-tricolourlovestory.tar.gz
tar -xzvf wenzhi-tricolourlovestory.tar.gz

sudo apt install nano
nano requirements.txt
#tensorflow-cpu~=1.15.0

sudo apt install libfreetype6-dev libpng-dev pkg-config
pip3 install -r requirements.txt

git clone https://github.com/libsndfile/libsndfile.git
sudo apt install autoconf autogen automake build-essential libasound2-dev \
  libflac-dev libogg-dev libtool libvorbis-dev libopus-dev libmp3lame-dev \
  libmpg123-dev pkg-config python
cd libsndfile
./autogen.sh
./configure --enable-werror
make
sudo make install

sudo ln -s /usr/local/lib/libsndfile.la /lib
sudo ln -s /usr/local/lib/libsndfile.so /lib
sudo ln -s /usr/local/lib/libsndfile.so.1 /lib
sudo ln -s /usr/local/lib/libsndfile.so.1.0.34 /lib

# https://github.com/scipy/scipy/issues/14541
pip3 uninstall numpy
pip3 install numpy

chmod +x *.sh
./run.sh
```



---

---

Reference: [NVIDIA/tacotron2](https://github.com/NVIDIA/tacotron2)

## How to use

1. Put raw Japanese texts in ./filelists
2. Put WAV files in ./wav
3. (Optional) Download NVIDIA's [pretrained model](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view?usp=sharing)
4. Open ./train.ipynb to install requirements and start training
5. Download NVIDIA's [WaveGlow model](https://drive.google.com/open?id=1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF) or [WaveGlow model](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EbyZnGnCJclGl5q_M3KGWTUBq4IIqSLiGznFdqHbv3WM5A?e=8c2aWE) based on Ayachi Nene
6. Open ./inference.ipynb to generate voice

## Cleaners

File ./hparams.py line 30

### 1. 'japanese_cleaners'

#### Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

#### After

nanikaacltaraitsudemohanashItekudasai.gakuiNnokotojanaku,shijinikaNsurukotodemonanidemo.

### 2. 'japanese_tokenization_cleaners'

#### Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

#### After

nani ka acl tara itsu demo hanashi te kudasai. gakuiN no koto ja naku, shiji nikaNsuru koto de mo naNdemo.

### 3. 'japanese_accent_cleaners'

#### Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

#### After

:na)nika a)cltara i)tsudemo ha(na)shIte ku(dasa)i.:ga(kuiNno ko(to)janaku,:shi)jini ka(Nsu)ru ko(to)demo na)nidemo.

### 4. 'japanese_phrase_cleaners'

#### Before

何かあったらいつでも話して下さい。学院のことじゃなく、私事に関することでも何でも

#### After

nanika acltara itsudemo hanashIte kudasai. gakuiNno kotojanaku, shijini kaNsuru kotodemo nanidemo.

## Models

Remember to change this line in ./inference.ipynb

```python
sequence = np.array(text_to_sequence(text, ['japanese_cleaners']))[None, :]
```

### Sanoba Witch

#### Ayachi Nene

* [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/ESltqOvyK3ZPsLMQwpv5FH0BoX8slLVsz3eUKwHHKkg9ww?e=vc5fdd) ['japanese_cleaners']

* [Model 2](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/ETNLDYH_ZRpMmNR0VGALhNQB5-LiJOqTaWQz8tXtbvCV-g?e=7nf2Ec) ['japanese_tokenization_cleaners']

* [Model 3](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/Eb0WROtOsYBInTmQQZHf36IBSXmyVd4JiCF7OnQjOZkjGg?e=qbbsv4) ['japanese_accent_cleaners']
  
  #### Inaba Meguru

* [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/Ed29Owd-E1NKstl_EFGZFVABe-F-a65jSAefeW_uEQuWxw?e=J628nT) ['japanese_tokenization_cleaners']

* [Model 2](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/ER8C2tiu4-RPi_MtQ3TCuTkBVRvO1MgJOPAKpAUD4ZLiow?e=ktT81t) ['japanese_tokenization_cleaners']
  
  ### Senren Banka
  
  #### Tomotake Yoshino

* [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EdfFetSH3tpMr7nkiqAKzwEBXjuCRICcvgUortEvE4pdjw?e=UyvkyI) ['japanese_tokenization_cleaners']

* [Model 2](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EeE4h5teC5xKms1VRnaNiW8BuqslFeR8VW7bCk7SWh2r8w?e=qADqbu) ['japanese_phrase_cleaners']
  
  #### Murasame

* [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EVXUY5tNA4JOqsVL7of8GrEB4WFPrcZPRWX0MP_7G0RXfg?e=5wzBlw) ['japanese_accent_cleaners']
  
  ### RIDDLE JOKER
  
  #### Arihara Nanami

* [Model 1](https://sjtueducn-my.sharepoint.com/:u:/g/personal/cjang_cjengh_sjtu_edu_cn/EdxWxcjx5XdAncOdoTjtyK0BUvrigdcBb2LPmzL48q4smw?e=OlAU66) ['japanese_accent_cleaners']

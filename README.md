# tacotron2-zh-hans (CPU training version)

http://download.pytorch.org/whl/cpu/torch/

## How to use

(For reference only, it is sorry I can't guarantee that following this step you will get it out of the box.)

- Clone this repo
  
  - ```
    git clone https://github.com/jieran233/tacotron2-zh-hans
    ```

- Create conda environment, and install dependencies in it.
  
  - ```
    conda create -n tacotron2-cpu python=3.7 -y
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
    conda install conda-forge::"blas=*=openblas" -y
    conda install -c conda-forge numpy -y
    ```
  
  - > [AMD CPU究竟可以不可以用MKL?教你用最好的配置加速numpy/sklearn/tensorflow - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/364051698)
    > 
    > ## 环境
    > 
    > MKL的环境我们用如下Anaconda指令来装：
    > 
    > ```text
    > $ conda create -n py38mkl python=3.8 && conda activate py38mkl
    > $ conda install numpy "blas=*=mkl"
    > $ conda install -c pytorch pytorch
    > $ conda install -c anaconda tensorflow-mkl
    > ```
    > 
    > OpenBLAS的环境我们用如下Anaconda指令来装：
    > 
    > ```text
    > $ conda create -n py38nomkl python=3.8 && conda activate py38nomkl
    > $ conda install nomkl
    > $ conda install numpy "blas=*=openblas"
    > $ pip install tensorflow
    > ```
    
    ```
    # mkl with debug
    conda create -n py37mkl-base python=3.7 -y && conda activate py37mkl-base
    conda install numpy "blas=*=mkl" -y
    
    # pytorch
    conda install -c pytorch pytorch -y
    # tensorflow
    conda install -c anaconda tensorflow-mkl -y
    
    conda install mkl=2020.0 intel-openmp=2020.0 -y
    ```
    
    ```
    conda create -n tacotron2-cpu --clone py37mkl-base -y
    conda activate tacotron2-cpu
    pip install -r requirements.txt
    
    # 这样安装下来虽然依赖关系会出问题但似乎不影响使用
    """
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    tensorflow 2.3.0 requires gast==0.3.3, but you have gast 0.2.2 which is incompatible.
    tensorflow 2.3.0 requires numpy<1.19.0,>=1.16.0, but you have numpy 1.19.2 which is incompatible.
    tensorflow 2.3.0 requires scipy==1.4.1, but you have scipy 1.6.2 which is incompatible.
    tensorflow 2.3.0 requires tensorboard<3,>=2.3.0, but you have tensorboard 1.15.0 which is incompatible.
    tensorflow 2.3.0 requires tensorflow-estimator<2.4.0,>=2.3.0, but you have tensorflow-estimator 2.6.0 which is incompatible.
    
    Successfully installed appdirs-1.4.4 audioread-3.0.0 cycler-0.11.0 decorator-5.1.1 gast-0.2.2 inflect-5.6.2 janome-0.4.2 joblib-1.1.0 kiwisolver-1.4.4 librosa-0.8.0 llvmlite-0.39.0 matplotlib-3.0.3 numba-0.56.0 packaging-21.0 pillow-9.1.1 pooch-1.6.0 protobuf-3.20.1 pyparsing-3.0.9 pypinyin-0.47.0 pysoundfile-0.9.0.post1 python-dateutil-2.8.1 resampy-0.4.0 scikit-learn-1.0.2 soundfile-0.10.3.post1 tensorboard-1.15.0 tensorflow-cpu-1.15.0 tensorflow-cpu-estimator-1.15.1 threadpoolctl-3.1.0 unidecode-1.3.4
    """
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

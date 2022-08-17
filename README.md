# tacotron2-zh-hans

## How to use

(For reference only, it is sorry I can't guarantee that following this step you will get it out of the box.)

- clone this repo
  
  - ```
    git clone https://github.com/jieran233/tacotron2-zh-hans
    ```

- create conda environment, and install dependencies in it.
  
  - ```
    conda create -n tacotron2 python=3.7
    conda activate tacotron2
    ```
  
  - ```
    # Pytorch with CUDA toolkit 11.3
    # Choose the version of CUDA toolkit that suits you
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    # Or use that command (It may download slowly)
    # pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    ```
  
  - ```
    # TensorFlow-GPU 1.15
    pip3 install tensorflow-gpu~=1.15
    ```
  
  - ```
    # cd to this repo that you cloned just now
    cd tacotron2-zh-hans
    # Install others dependencies
    pip3 install -r requirements.txt
    ```

-  Modify config before training
  
  - File ./hparams.py line 10, 11
    
    Change the values to what you want.
    
    ```python
        epochs = 100
        iters_per_checkpoint = 500
    ```
  
  - File ./hparams.py line 81
    
    Decrease the batch_size to 32 or lower.
    
    If VideoRAM BOMB(CUDA out of memory) again, decrease the value of batch_size again.
    
    ```
        batch_size = 64
    ```

- Start training without pretraint model
  
  ```
  python3 train.py --output_directory=/root/autodl-nas/output/ckpt --log_directory=/root/autodl-nas/output/log
  ```

- Or continue an unfinished work:
  
  Start training with pretraint model
  
  ```
  python3 train.py -c checkpoint.pt --output_directory=/root/autodl-nas/output/ckpt --log_directory=/root/autodl-nas/output/log
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

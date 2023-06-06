# MAUNSS



##Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
pip install -r requirements.txt
```

##Prepare Dataset:

Download cave_1024_28 ([One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--MAUNSS
    |--real
    	|-- test_code
    	|-- train_code
    |--simulation
    	|-- test_code
    	|-- train_code
    |--visualization
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

Following TSA-Net and DGSMP, we use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 

## Simulation Experiement:

### Training

```shell
cd ./simulation/train_code/

# MAUNSS_3stg
python train.py  --outf ./exp/MAUNSS_3stg/ --method OLU_3stg 

# MAUNSS_5stg
python train.py --outf ./exp/MAUNSS_5stg/ --method OLU_5stg  

# MAUNSS_7stg
python train.py --outf ./exp/MAUNSS_7stg/ --method OLU_7stg 

# MAUNSS_9stg
python train.py --outf ./exp/MAUNSS_9stg/ --method OLU_9stg 

```

The training log, trained model, and reconstrcuted HSI will be available in `./simulation/train_code/exp/` . 


### Testing	

Run the following command to test the model on the simulation dataset.

```shell
cd ./simulation/test_code/

# MAUNSS_3stg
python test.py  --outf ./exp/MAUNSS_3stg/ --method OLU_3stg --pretrained_model_path ./MAUNSS_3stg.pth

# MAUNSS_5stg
python test.py --outf ./exp/MAUNSS_5stg/ --method OLU_5stg  --pretrained_model_path ./MAUNSS_5stg.pth

# MAUNSS_7stg
python test.py --outf ./exp/MAUNSS_7stg/ --method OLU_7stg --pretrained_model_path ./MAUNSS_7stg.pth

# MAUNSS_9stg
python test.py --outf ./exp/MAUNSS_9stg/ --method OLU_9stg --pretrained_model_path ./MAUNSS_9stg.pth

```

- The reconstrcuted HSIs will be output into `MAUNSS/simulation/test_code/exp/`  




## Real Experiement:

### Training

```shell
cd ./real/train_code/

# MAUNSS_3stg
python train.py  --outf ./exp/MAUNSS_3stg/ --method OLU_3stg 


The training log, trained model, and reconstrcuted HSI will be available in `MAUNSS_3stg/real/train_code/exp/` . 
```

### Testing	

```shell
cd ./real/test_code/

# MAUNSS_3stg
python test.py  --outf ./exp/MAUNSS_3stg/ --method OLU_3stg  --pretrained_model_path ./MAUNSS_3stg.pth

The reconstrcuted HSI will be output into `MAUNSS_3stg/real/test_code/exp/`  
```
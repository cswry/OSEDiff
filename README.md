<div align="center">


<h1>One-Step Effective Diffusion Network for Real-World Image Super-Resolution</h1>

<div>
    <a href='https://scholar.google.com/citations?user=A-U8zE8AAAAJ&hl=zh-CN' target='_blank'>Rongyuan Wu<sup>1,2,*</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=ZCDjTn8AAAAJ&hl=zh-CN' target='_blank'>Lingchen Sun<sup>1,2,*</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=F15mLDYAAAAJ&hl=en' target='_blank'>Zhiyuan Ma<sup>1,*</sup></a>&emsp;
    <a href='https://www4.comp.polyu.edu.hk/~cslzhang/' target='_blank'>Lei Zhang<sup>1,2,† </sup></a>
</div>
<div>
    <sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute&emsp; 
</div>

[[paper]](https://arxiv.org/pdf/2406.08177v2)

---

</div>

## 🔥 News
- [2024.07] Release OSEDiff-SD21base.
- [2024.06] This repo is created.

## 🎬 Overview
![overview](asserts/framework.jpg)


## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/cswry/OSEDiff.git
    cd OSEDiff
    ```

2. Install dependent packages
    ```bash
    conda create -n OSEDiff python=3.10 -y
    conda activate OSEDiff
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. Download Models 
#### Dependent Models
* [SD21 Base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
* [RAM](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth)
* [DAPE](https://drive.google.com/file/d/1KIV6VewwO2eDC9g4Gcvgm-a0LDI7Lmwm/view?usp=drive_link)


## ⚡ Quick Inference
```
python test_osediff.py \
-i preset/datasets/test_dataset/input \
-o preset/datasets/test_dataset/output \
--osediff_path preset/models/osediff.pkl \
--pretrained_model_name_or_path SD21BASE_PATH \
--ram_ft_path DAPE_PATH \
--ram_path RAM_PATH
```

## 📷 Results
[<img src="asserts/compare_01.png" height="400px"/>](https://imgsli.com/Mjc1ODI1) [<img src="asserts/compare_02.png" height="400px"/>](https://imgsli.com/Mjc1ODMx)

![benchmark](asserts/benchmark.jpg)

<details>
<summary>Quantitative Comparisons (click to expand)</summary>

<p align="center">
  <img width="900" src="asserts/tab_main.png">
</p>

<p align="center">
  <img width="900" src="asserts/tab_main_gan.png">
</p>
</details>

<details>
<summary>Visual Comparisons (click to expand)</summary>

<p align="center">
  <img width="900" src="asserts/fig_main.png">
</p>

<p align="center">
  <img width="900" src="asserts/fig_main_gan.png">
</p>
</details>


## 📧 Contact
If you have any questions, please feel free to contact: `rong-yuan.wu@connect.polyu.hk`


## 🎓Citations
<!-- If our code helps your research or work, please consider citing our paper.
The following are BibTeX references: -->

```
@article{wu2024one,
  title={One-Step Effective Diffusion Network for Real-World Image Super-Resolution},
  author={Wu, Rongyuan and Sun, Lingchen and Ma, Zhiyuan and Zhang, Lei},
  journal={arXiv preprint arXiv:2406.08177},
  year={2024}
}
```

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=cswry/OSEDiff)

</details>



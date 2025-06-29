<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="./imgs/CEP_logo_hires.png" width="50%">
    <h3 align="center"><strong>CEP: Computational Endoscopy Platform (advanced deep learning toolset for analyzing endoscopy videos)</strong></h3>
    <p align="center">
    <a href="#miccai25-rt-gan-recurrent-temporal-gan-for-adding-lightweight-temporal-consistency-to-frame-based-domain-translation-approaches">RT-GAN MICCAI'25</a>    
    |
    <a href="#miccai22-clts-gan-color-lighting-texture-specular-reflection-augmentation-for-colonoscopy">CLTS-GAN MICCAI'22</a>
    |
    <a href="#miccai21-foldit-haustral-folds-detection-and-segmentation-in-colonoscopy-videos">FoldIt MICCAI'21</a>
    |
    <a href="#cvpr20-augmenting-colonoscopy-using-extended-and-directional-cyclegan-for-lossy-image-translation">XDCycleGAN CVPR'20</a>
    |
    <a href="#isbi21-visualizing-missing-surfaces-in-colonoscopy-videos-using-shared-latent-space-representations">MissedSurface ISBI'21</a>
    |
    <a href="https://github.com/nadeemlab/CEP/issues">Issues</a>
  </p>
</p>





Computational Endoscopy Platform (CEP) provides an exhaustive deep learning toolset to handle tasks such as haustral fold annotation (in colonoscopy videos), surface coverage visualization, depth estimation, color-lighting-texture-specular reflection augmentation, and more. All our code, AI-ready training/testing data, and pretrained models will be released here with detailed instructions along with easy-to-run docker containers and Google CoLab projects.

Internal Dataset            |  Public Dataset
:-------------------------:|:-------------------------:
<img src="imgs/internal_gifs/gifp1.gif" alt="FoldIt_Preview" width = 400 /> | <img src="imgs/public_gifs/pub_gifp1.gif" alt="FoldIt_Preview"  width = 400/> 
<img src="imgs/internal_gifs/gifp2.gif" alt="FoldIt_Preview" width = 400 /> | <img src="imgs/public_gifs/pub_gifp2.gif" alt="FoldIt_Preview"  width = 400/> 
<img src="imgs/internal_gifs/gifp3.gif" alt="FoldIt_Preview" width = 400 /> | <img src="imgs/public_gifs/pub_gifp3.gif" alt="FoldIt_Preview"  width = 400/> 





© This code is made available for non-commercial academic purposes.

## Updates:
- [x] RT-GAN **MICCAI'25** code released for adding lightweight temporal consistency to frame-based domain translation approaches. Please cite the following paper:
> Mathew S*, Nadeem S*, Alvin C. Goh, Kaufman A.
> RT-GAN: Recurrent Temporal GAN for Adding Lightweight Temporal Consistency to Frame-Based Domain Translation Approaches.
> *International Conference on Medical Imaging Computing and Computer Assisted Intervention (MICCAI)*, 2025. (* Equal Contribution) [Accepted]
> [[Paper Link]](https://arxiv.org/abs/2310.00868) [[Supplementary Video]](https://youtu.be/UMVP-uIXwWk)

- [x] CLTS-GAN **MICCAI'22** code released for color-lighting-texture-specular reflection augmentation in colonoscopy video frames. Please cite the following paper:
> Mathew S*, Nadeem S*, Kaufman A.
> CLTS-GAN: Color-Lighting-Texture-Specular Reflection Augmentation for Colonoscopy.
> *International Conference on Medical Imaging Computing and Computer Assisted Intervention (MICCAI)*, 2022. (* Equal Contribution)
> [[Paper Link]](https://arxiv.org/pdf/2206.14951.pdf)

- [x] AI-ready training and testing data released. This dataset is created from public [HyperKvasir](https://osf.io/mh9sj/) optical colonoscopy videos and [TCIA CT colonography](https://wiki.cancerimagingarchive.net/display/Public/CT+Colonography) repositories. Easy-to-run Docker containers and Google CoLab projects are also released. 

- [x] FoldIt **MICCAI'21** code released for haustral fold detection/segmentation in colonoscopy videos. Please cite the following paper:
> Mathew S*, Nadeem S*, Kaufman A.
> FoldIt: Haustral Folds Detection and Segmentation in Colonoscopy Videos.
> *International Conference on Medical Imaging Computing and Computer Assisted Intervention (MICCAI)*, 12903, 221-230, 2021. (* Equal Contribution) 
> [[Paper Link]](https://arxiv.org/abs/2106.12522) [[Supplementary Video]](https://www.youtube.com/watch?v=_iWBJnDMXjo) [[Reviews]](https://miccai2021.org/openaccess/paperlinks/2021/09/01/201-Paper1000.html)

- [x] XDCycleGAN **CVPR'20** code released for scale-consistent depth estimation for colonoscopy videos. Please cite the following paper:
> Mathew S*, Nadeem S*, Kumari S, Kaufman A. 
> Augmenting Colonoscopy using Extended and Directional CycleGAN for Lossy Image Translation.
> *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 4696-4705, 2021. (* Equal Contribution) 
> [[Paper Link]](https://openaccess.thecvf.com/content_CVPR_2020/html/Mathew_Augmenting_Colonoscopy_Using_Extended_and_Directional_CycleGAN_for_Lossy_Image_CVPR_2020_paper.html) [[Supplementary Video]](https://youtu.be/9JZdnwtsE6I)

- [ ] For surface coverage visualization, we will release our **ISBI 2021**:
> Mathew S*, Nadeem S*, Kaufman A.
> Visualizing Missing Surfaces In Colonoscopy Videos using Shared Latent Space Representations.
> *IEEE 18th International Symposium on Biomedical Imaging (ISBI)*, 329-333, 2021. (* Equal Contribution) 
> [[Paper Link]](https://arxiv.org/abs/2101.07280) [[Supplementary Video]](https://youtu.be/x1-wwCiYeC0)


## Prerequesites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
To install the CEP, this repo needs to be cloned
```
git clone https://github.com/nadeemlab/CEP.git
cd CEP
```

Once the repo is cloned, the python libraries can be installed 
  - via pip ``` pip install -r requirements.txt ```
  - via conda ``` conda env create -f environment.yml ```

### Docker
A dockerfile is provided as an additional way to install.
  - First, [Docker](https://docs.docker.com/engine/install/ubuntu/) needs to be installed along with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support
  - Build Docker Image ```docker build -t cep .```
  - Create and Run Docker Container ```docker run --gpus all --name CEP -it cep```

## [[MICCAI'25]](https://arxiv.org/abs/2310.00868) RT-GAN: Recurrent Temporal GAN for Adding Lightweight Temporal Consistency to Frame-Based Domain Translation Approaches
While developing new unsupervised domain translation methods for colonoscopy (e.g. to translate between real optical and virtual/CT colonoscopy), it is thus typical to start with approaches that initially work for individual frames without temporal consistency. Once an individual-frame model has been finalized, additional contiguous frames are added with a modified deep learning architecture to train a new model from scratch for temporal consistency. This transition to temporally-consistent deep learning models, however, requires significantly more computational and memory resources for training. In this paper, we present a lightweight solution with a tunable temporal parameter, RT-GAN (Recurrent Temporal GAN), for adding temporal consistency to individual frame-based approaches that reduces training requirements by a factor of 5. We demonstrate the effectiveness of our approach on two challenging use cases in colonoscopy: haustral fold segmentation (indicative of missed surface) and realistic colonoscopy simulator video generation. We also release a first-of-its kind temporal dataset for colonoscopy for the above use cases.

To train the RT-GAN model, run the following command. During the training process, results can be viewed via visdom. By default it is on http://localhost:8097.
``` 
python3 train.py --dataroot path_to_dataset -model rtgan -name "rtgan_model_name" 
```

To test your trained model, run the following command.

```
python3 test.py --dataroot path_to_dataset -model rtgan -name "rtgan_model_name"
```

### Public Dataset and Model
Both the dataset and the models can be found [here](https://zenodo.org/records/15460791).

## [[MICCAI'22]](https://arxiv.org/abs/2206.14951) CLTS-GAN: Color-Lighting-Texture-Specular Reflection Augmentation for Colonoscopy
Automated analysis of optical colonoscopy (OC) video frames (to assist endoscopists during OC) is challenging due to variations in color, lighting, texture, and specular reflections. Previous methods either remove some of these variations via preprocessing (making pipelines cumbersome) or add diverse training data with annotations (but expensive and time-consuming). We present CLTS-GAN, a new deep learning model that gives fine control over color, lighting, texture, and specular reflection synthesis for OC video frames. We show that adding these colonoscopy-specific augmentations to the training data can improve state-of-the-art polyp detection/segmentation methods as well as drive next generation of OC simulators for training medical students.

<p align="center">
  <img src="imgs/color_lighting_preview.png" alt="Color_Lighting_Preview" width="400"/> <img src="imgs/texture_specular_preview.png" alt="Texture_Specular_Preview" width="320"/>
</p>
<!-- ![Color_Lighting_Preview](imgs/color_lighting_preview.png =200x) -->

To train the CLTS model, run the following command. During the training process, results can be viewed via visdom. By default it is on http://localhost:8097.
``` 
python3 train.py --dataroot path_to_dataset -model cltsgan -name "cltsgan_model_name" 
```

To run your trained model to generate texture and colors, run the following command.

```
python3 test.py --dataroot path_to_dataset -model cltsTest -name "clts_model_name"
```
 The command above will create 5 images with different texture and color added to VC images. Here are some useful arugments:
- ```--freeze_color``` will fix the color information for each iteration of the images
- ```--freeze_texture``` will fix the texture information for each iteration of the images
- ```--augment``` will allow for augmentation of OC image input (image will passthrough both generator to allow which also works with the two argumnents above)

### Public Dataset
We augmented a portion of the polyp detection dataset from [PraNet](https://github.com/DengPingFan/PraNet). Both the model and augmented data can be found [here](https://zenodo.org/record/7036198).

## [[MICCAI'21]](https://arxiv.org/abs/2106.12522) FoldIt: Haustral Folds Detection and Segmentation in Colonoscopy Videos
Haustral folds are colon wall protrusions implicated for high polyp miss rate during optical colonoscopy procedures. If segmented accurately, haustral folds can allow for better estimation of missed surface and can also serve as valuable landmarks for registering pre-treatment virtual (CT) and optical colonoscopies, to guide navigation towards the anomalies found in pre-treatment scans. We present a novel generative adversarial network, FoldIt, for feature-consistent image translation of optical colonoscopy videos to virtual colonoscopy renderings with haustral fold overlays. A new transitive loss is introduced in order to leverage ground truth information between haustral fold annotations and virtual colonoscopy renderings. We demonstrate the effectiveness of our model on real challenging optical colonoscopy videos as well as on textured virtual colonoscopy videos with clinician-verified haustral fold annotations. In essence, the **FoldIt** model is a method for translating between domains when a shared common domain is available. We use the FoldIt model to learn a translation from optical colonoscopy to haustral fold annotation via a common virtual colonoscopy domain.

<p align="center">
  <img src="imgs/FoldIt_preview.PNG" alt="FoldIt_Preview" width="600"/>
</p>
<!-- ![FoldIt_Preview](imgs/FoldIt_preview.PNG =200x) -->


To train the FoldIt model, run the following command. During the training process, results can be viewed via visdom. By default it is on http://localhost:8097.
``` 
python3 train.py --dataroot path_to_dataset -model foldit -name "foldit_model_name" 
```

To test your trained model, run the following command.

```
python3 test.py --dataroot path_to_dataset -model foldit -name "foldit_model_name"
```

Our model weights and OC testing data is provided [here](https://zenodo.org/record/4993651). We trained our model with 80 generator filters so, when testing our model use the following command

```
python3 test.py --dataroot path_to_dataset -model foldit -name "foldit_model_name" --ngf 80
```

### Dataset Format
When training, the network will look for 'trainA', 'trainB', and 'trainC' folders each containing images from domains A, B and C in the dataroot folder. During testing time, 'testA', 'testB', and 'testC' subfolders should contain images for testing.

### Public Dataset
We trained our model on public data and found the results do not differ significantly. Both the model and data can be found [here](https://zenodo.org/record/5519974).

## [[CVPR'20]](https://openaccess.thecvf.com/content_CVPR_2020/html/Mathew_Augmenting_Colonoscopy_Using_Extended_and_Directional_CycleGAN_for_Lossy_Image_CVPR_2020_paper.html) Augmenting Colonoscopy Using Extended and Directional CycleGAN for Lossy Image Translation
Colorectal cancer screening modalities, such as optical colonoscopy (OC) and virtual colonoscopy (VC), are critical for diagnosing and ultimately removing polyps (precursors for colon cancer). The non-invasive VC is normally used to inspect a 3D reconstructed colon (from computed tomography scans) for polyps and if found, the OC procedure is performed to physically traverse the colon via endoscope and remove these polyps. In this paper, we present a deep learning framework, Extended and Directional CycleGAN, for lossy unpaired image-to-image translation between OC and VC to augment OC video sequences with scale-consistent depth information from VC and VC with patient-specific textures, color and specular highlights from OC (e.g. for realistic polyp synthesis). Both OC and VC contain structural information, but it is obscured in OC by additional patient-specific texture and specular highlights, hence making the translation from OC to VC lossy. The existing CycleGAN approaches do not handle lossy transformations. To address this shortcoming, we introduce an extended cycle consistency loss, which compares the geometric structures from OC in the VC domain. This loss removes the need for the CycleGAN to embed OC information in the VC domain. To handle a stronger removal of the textures and lighting, a Directional Discriminator is introduced to differentiate the direction of translation (by creating paired information for the discriminator), as opposed to the standard CycleGAN which is direction-agnostic. Combining the extended cycle consistency loss and the Directional Discriminator, we show state-of-the-art results on scale-consistent depth inference for phantom, textured VC and for real polyp and normal colon video sequences. We also present results for realistic pendunculated and flat polyp synthesis from bumps introduced in 3D VC models.


<p align="center">
  <img src="imgs/xdcyclegan_preview.PNG" alt="XDCycleGAN_Preview" width="600"/>
</p>

To train the XDCycleGAN model, run the following command. During the training process, results can be viewed via visdom. By default it is on http://localhost:8097.
``` 
python3 train.py --dataroot path_to_dataset -model xdcyclegan -name "xdcyclegan_model_name" 
```

To test your trained model, run the following command.

```
python3 test.py --dataroot path_to_dataset -model xdcyclegan -name "xdcyclegan_model_name"
```
The XDCycleGAN models trained on OC and depth/VC is provided [here](https://zenodo.org/record/5335909). We trained our model with 80 generator filters so, when testing our model use the following command

```
python3 test.py --dataroot path_to_dataset -model xdcyclegan -name "xdcyclegan_model_name" --ngf 80
```

<!--
Our model weights and OC testing data is provided [here](https://zenodo.org/record/4993651). We trained our model with 80 generator filters so, when testing our model use the following command

```
python3 test.py --dataroot path_to_dataset -model foldit -name "foldit_model_name" --ngf 80
```
-->

### Dataset Format
When training, the network will look for 'trainA' and 'trainB' folders each containing images from domains A and B in the dataroot folder. During testing time, 'testA' and 'testB' subfolders should contain images for testing.

### Public Dataset
We trained our model on public data and found the results do not differ significantly. The model and data for depth esimation can be found [here](https://zenodo.org/record/5520029). The model and data for VC can be found [here](https://zenodo.org/record/5520019).

## [[ISBI'21]](https://arxiv.org/abs/2101.07280) Visualizing Missing Surfaces In Colonoscopy Videos using Shared Latent Space Representations
Optical colonoscopy (OC), the most prevalent colon cancer screening tool, has a high miss rate due to a number of factors, including the geometry of the colon (haustral fold and sharp bends occlusions), endoscopist inexperience or fatigue, endoscope field of view, etc. We present a framework to visualize the missed regions per-frame during the colonoscopy, and provides a workable clinical solution. Specifically, we make use of 3D reconstructed virtual colonoscopy (VC) data and the insight that VC and OC share the same underlying geometry but differ in color, texture and specular reflections, embedded in the OC domain. A lossy unpaired image-to-image translation model is introduced with enforced shared latent space for OC and VC. This shared latent space captures the geometric information while deferring the color, texture, and specular information creation to additional Gaussian noise input. This additional noise input can be utilized to generate one-to-many mappings from VC to OC and OC to OC. 

<p align="center">
  <img src="imgs/isbi_preview.png" alt="ISBI_Preview" width="600"/>
</p>


## Issues
Please report all issues on the public forum.

## License
© [Nadeem Lab](https://nadeemlab.org/) - This code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 

## Acknowledgments
* This code is inspired by [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our papers:

```
@article{mathew2022cltsgan,
  title={RT-GAN: Recurrent Temporal GAN for Adding Lightweight Temporal Consistency to Frame-Based Domain Translation Approaches},
  author={Mathew, Shawn* and Nadeem, Saad* and Alvin Goh and Kaufman, Arie},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2025}
}

@article{mathew2022cltsgan,
  title={CLTS-GAN: Color-Lighting-Texture-Specular Reflection Augmentation for Colonoscopy},
  author={Mathew, Shawn* and Nadeem, Saad* and Kaufman, Arie},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year={2022}
}

@article{mathew2021foldit,
  title={FoldIt: Haustral Folds Detection and Segmentation in Colonoscopy Videos},
  author={Mathew, Shawn* and Nadeem, Saad* and Kaufman, Arie},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  volume={12903},
  pages={221--230},
  year={2021}
}

@article{mathew2021visualizing,
  title={Visualizing Missing Surfaces In Colonoscopy Videos using Shared Latent Space Representations},
  author={Mathew, Shawn* and Nadeem, Saad* and Kaufman, Arie},
  journal={International Symposium on Biomedical Imaging (ISBI)},
  pages={329--333},
  year={2021}
}

@inproceedings{mathew2020augmenting,
  title={Augmenting Colonoscopy using Extended and Directional CycleGAN for Lossy Image Translation},
  author={Mathew, Shawn* and Nadeem, Saad* and Kumari, Sruti and Kaufman, Arie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4696--4705},
  year={2020}
}
```

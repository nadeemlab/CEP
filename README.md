# CEP: Computational Endoscopy Platform

Computational Endoscopy Platform (CEP) provides an exhaustive deep learning toolset to handle tasks such as haustral fold annotation (in colonoscopy videos), surface coverage visualization, depth estimation and more. All our code, training data, and pretrained models will be released here with detailed instruction along with docker containers and Google CoLab projects.

© This code is made available for non-commercial academic purposes.

Currently, we are releasing code for our **MICCAI 2021** work, FoldIt, which is a haustral fold detection/segmentation algorithm for colonoscopy videos. Please cite the following paper:
> Mathew S*, Nadeem S*, Kaufman A. (* Equal Contribution) 
> FoldIt: Haustral Folds Detection and Segmentation in Colonoscopy Videos.
> International Conference on Medical Imaging Computing and Computer Assisted Intervention (MICCAI), 2021. (Provisionally Accepted)

In the coming weeks, we will also release depth estimation code and pretrained models for our **CVPR 2020** paper:
> Mathew S*, Nadeem S*, Kumari S, Kaufman A. 
> Augmenting Colonoscopy using Extended and Directional CycleGAN for Lossy Image Translation.
> Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4696--4705, 2021
> [[Paper and Supplementary Video Link]](https://openaccess.thecvf.com/content_CVPR_2020/html/Mathew_Augmenting_Colonoscopy_Using_Extended_and_Directional_CycleGAN_for_Lossy_Image_CVPR_2020_paper.html)

For surface coverage visualization, we will release our **ISBI 2021** and upcoming **TMI** code as well:
> Mathew S*, Nadeem S*, Kaufman A.
> Visualizing Missing Surfaces In Colonoscopy Videos using Shared Latent Space Representations.
> IEEE 18th International Symposium on Biomedical Imaging (ISBI), 329--333, 2021
> [[Paper Link]](https://arxiv.org/pdf/2101.07280.pdf) [[Supplementary Video]](https://tinyurl.com/y2dfwlc9)


## Prerequesites
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
To install the CEP, this repo needs to be cloned
```
git clone https://github.com/nadeemlab/CEP.git
cd CEP
```

Once the repo is cloned, the python libraries can be installed via ``` pip install -r requirements.txt ```


## FoldIt: Haustral Folds Detection and Segmentation in Colonoscopy Videos (MICCAI'21)

Haustral folds are colon wall protrusions implicated for high polyp miss rate during optical colonoscopy procedures. If segmented accurately, haustral folds can allow for better estimation of missed surface and can also serve as valuable landmarks for registering pre-treatment virtual (CT) and optical colonoscopies, to guide navigation towards the anomalies found in pre-treatment scans. We present a novel generative adversarial network, FoldIt, for feature-consistent image translation of optical colonoscopy videos to virtual colonoscopy renderings with haustral fold overlays. A new transitive loss is introduced in order to leverage ground truth information between haustral fold annotations and virtual colonoscopy renderings. We demonstrate the effectiveness of our model on real challenging optical colonoscopy videos as well as on textured virtual colonoscopy videos with clinician-verified haustral fold annotations. In essence, the **FoldIt** model is a method for translating between domains when a shared common domain is available. We use the FoldIt model to learn a translation from optical colonoscopy to haustral fold annotation via a common virtual colonoscopy domain.

To train the FoldIt model, run the following command:
```
python3 train.py --dataroot path_to_dataset -model foldit -name "foldit_model_name" 

```

## Issues
Please report all issues on the public forum.

## License
© [Nadeem Lab](https://nadeemlab.org/) - This code is distributed under **Apache 2.0 with Commons Clause** license, and is available for non-commercial academic purposes. 

## Acknowledgments
* This code is inspired by [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our papers:

```
@article{mathew2021foldit,
  title={FoldIt: Haustral Folds Detection and Segmentation in Colonoscopy Videos},
  author={Mathew, Shawn* and Nadeem, Saad* and Kaufman, Arie},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
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

# **Diffusion Visual Counterfactual Explanations**

Welcome to the codebase for our NeurIPS paper *Diffusion Visual Counterfactual Explanations.* We will show you how to generate DVCEs on the selected (and you can choose the targets yourselves) ImageNet images with the multiple norm robust model Madry + FT and two SOTA **non-robust** models Swin-T and ConvNeXt. 

## Examples of DVCEs for the **non-robust** ConvNeXt classifier

In the following, first image is the starting image, `GT` stands for the "ground-truth label", and the images in the second and third columns - are VCEs in the respective target classes displayed above. For each the achieved and the initial confidences (`i`) are displayed above.

<p align="center">
  <img src="image_examples/0.png" />
</p>
<p align="center">
  <img src="image_examples/2.png" />
</p>
<p align="center">
  <img src="image_examples/4.png" />
</p>
<p align="center">
  <img src="image_examples/8.png" />
</p>
<p align="center">
  <img src="image_examples/10.png" />
</p>

## Setup

Before we can start with the generation, we have to setup the project and install required packages.

* Start by extracting the content of the .zip file that also contains this readme.md somewhere on your computer. We will refer to the extraction directory as **project_path**.
* Navigate into the  **project_path**

* Download the weights for Madry + FT from [here](https://drive.google.com/file/d/1sUR81A5OckMS0maneU5KWOpc99rCtESR/view?usp=sharing) into your **project_path**
* Unzip the model file via `unzip MadryFT.zip` 

* Execute `mkdir checkpoints; cd checkpoints`
* and `wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt`


* Set variable *data_folder* to the folder that contains imagenet dataset
  For example, if your imagenet folder is located under '/scratch/datasets/imagenet', then use `data_folder='/scratch/datasets'`

* Create a new conda env via `conda env create -f python_38_dvces.yml`
* Activate the conda environment via `conda activate python_38_dvces`
* Install additionally robustbench via `pip install git+https://github.com/RobustBench/robustbench.git`


## Creating  DVCEs/SVCEs/blended diffusion based VCEs

In the following, we show, how to first set the parameters, and then - generate VCEs of the respective type for 6 selected targets. To choose your own image ids and targets, change `some_vces`, but consider targets that are semantically close to the original image, to ensure that meaningful explanations (VCEs) can be generated.

For any of the proposed parameter settings, feel free to adjust the values, but these are the ones we have used mostly in the paper.

* Generating DVCEs without cone projection for Madry + FT via
  `python imagenet_VCEs.py --data_folder $data_folder --num_imgs 12 --denoise_dist_input > logs/log`  

* Generating DVCEs with the cone projection for Madry + FT and respectively Swin-T (model id is 30) and ConvNeXt (model id is 31) via
  `second_classifier_ts=(31 30)`
  and then
  `for second_classifier_t in "${second_classifier_ts[@]}"; do python imagenet_VCEs.py --data_folder $data_folder --deg_cone_projection 30 --second_classifier_type $second_classifier_t --num_imgs 12 --denoise_dist_input --aug_num 16 > logs/log; done`

* Generating SVCEs for Madry + FT via
  `python imagenet_VCEs.py --data_folder $data_folder --num_imgs 12 --config 'svce.yml' > logs/log` 

* Generating blended diffusion based VCEs via
  `python imagenet_VCEs.py --data_folder $data_folder --num_imgs 12 --config 'blended.yml' --use_blended --background_preservation_loss > logs/log` 

The batchsize argument `--batch_size` is the number of samples per gpu, so if you encounter out-of-memory errors you can reduce it without altering results.

The resulting images can be found in `ImageNetVCEs/examples/`.

## Citation

If you find this useful in your research, please consider citing:

```bibtex
@inproceedings{Augustin2022Diffusion,
      title={Diffusion Visual Counterfactual Explanations},
      author={Maximilian Augustin and Valentyn Boreiko and  Francesco Croce  and Matthias Hein},
      booktitle={NeurIPS},
      year={2022}
}

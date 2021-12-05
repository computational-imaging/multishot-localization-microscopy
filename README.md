# Deep learning multi-shot 3D localization microscopy using hybrid optical-electronic computing
This repository contains code for the paper <em>Deep learning multi-shot 3D localization microscopy using hybrid optical-electronic computing</em> by [Hayato Ikoma](https://hayatoikoma.github.io/), [Takamasa Kudo](https://scholar.google.com/citations?user=d1ZrZ6YAAAAJ&hl=en), [Evan Peng](http://stanford.edu/~evanpeng/), [Michael Broxton](http://graphics.stanford.edu/~broxton/) and [Gordon Wetzstein](https://stanford.edu/~gordonwz/).

[[Project website]](https://www.computationalimaging.org/publications/localization-microscopy/)

[[Paper (to appear)]]()

## How to simulate the designed PSF
Run
```shell
python render_psf.py
```
This will create `result` directory and save a multi-stack tiff file. 

![Optimized PSF](img/psf.gif)


## How to set up a conda environement
Run `create_environment.sh` to create a conda environment for this repo.
You may need to change the version of cudatoolkit depending on your environment.


## Trained model and captured dataset
The trained model and the captured dataset is available [here](https://drive.google.com/file/d/1G94qKB2otmUhW2iKbmOWrL3v__HHfNVW/view?usp=sharing). Expand the downloaded zip file in `data` directory.
The zip file contains the trained phase mask and model for both fixed cells and live cells, and the captured PSF and raw data.



## How to run the training 
Our training framework is based on [Pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).
Run
```shell
python localizer_trainer.py --optimize_optics --gpus 1 --default_root_dir="data/logs"
```
The hyperparameters can be changed through flags. See [localizer.py](localizer.py), [module.microscope.py](module/microscope.py) and [pytorch_lightning.Trainer](https://pytorch-lightning.readthedocs.io/en/1.2.4/common/trainer.html#init) for available hyperparameters. Note that some available functionalities are not used in the paper.
You can change how many GPUs you want to use with `--gpus` flag.
The Tensorboard log will be saved in `--default_root_dir`.

## How to run the inference
Run the following to run the inference.
```shell
python infer.py \
    --img_path data/captured_data/fixed_cell.tif \
    --ckpt_path data/trained_model/fixed_cell.ckpt \
    --batch_sz 10 --save_dir result --gpus 1
```

## Contact
Hayato Ikoma (hikoma@stanford.edu)
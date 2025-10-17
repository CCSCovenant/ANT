## Get Start

This code was tested on `NVIDIA RTX 4090` and requires:

* conda3 or miniconda3
* python 3.8+
* pytorch 3.10+


a. Create a conda virtual environment and activate it.

```shell
bash setup_env.sh
```
b. Modify the `LayerNorm` module in clip for fp16 inference
```python
# miniconda3/envs/stablemofusion/lib/python3.8/site-packages/clip/model.py
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        if self.weight.dtype==torch.float32:

            orig_type = x.dtype
            ret = super().forward(x.type(torch.float32)) 
            return ret.type(orig_type)  
        else:
             return super().forward(x)
```
 
## Quick Start
1. Download pre-trained models from [Google Cloud](https://drive.google.com/drive/folders/1o3h0DHEz5gKG-9cTdl3lUEwjwW51Ay81?usp=sharing) and put them into ./ckeckpoints/ and arrange them in the following file structure:
```text
StableMoFusion
├── checkpoints
│   └── kit
│       └── ant_kit
│           ├── meta
│           │   ├── mean.npy
│           │   └── std.npy
│           ├── model
│           │   └── latest.tar
│           └── opt.txt
│   └── t2m
│       └── ant_t2m
│           ├── meta
│           │   ├── mean.npy
│           │   └── std.npy
│           ├── model
│           │   └── latest.tar
│           └── opt.txt
│   └── footskate
│       ├── underpressure_pretrained.tar
│       └── t2m_pretrained.tar  
```
2. Download the [UnderPressure code](https://github.com/InterDigitalInc/UnderPressure) and put them into ./UnderPressure/ like:
```
StableMoFusion
├── UnderPressure
│   ├── dataset
│   |   |── S1_HoppingLeftFootRightFoot.pth
│   |   └── ...
│   ├── anim.py
│   ├── data.py
│   ├── demo.py
│   └── ...
```
3. Updating import paths within `./Underpressure/*.py`. 
To ensure modules within the ./Underpressure/ can be imported and utilized seamlessly via python -m, it's necessary to update the import paths within the Python files located in ./Underpressure/*.py. For example:
* Replace `import util` with `from Underpressure import util` in UnderPressure/anim.py
* Replace `import anim, metrics, models, util`  with `from UnderPressure import anim, metrics, models, util` in UnderPressure/demo.py
4. run demo.py or scripts/generate.py 
```shell
# generate from a single prompta
# e.g. generate a 4-second wave motion . Unit of `--motion_length` is seconds.
python -m scripts.inference.generate --text_prompt "A man walks forward and picks up a toolbox." --motion_length 4 --opt_path checkpoints/t2m/ant_t2m/opt.txt

# Generate from your text file
# e.g. generate 5 motions by different prompts in .txt file, and set the motion frame length separately by .txt file. Unit of `--input_len` is frame.
python -m scripts.inference.generate  --opt_path checkpoints/t2m/ant_t2m/opt.txt
--input_text ./aaa_vvv.txt

```

**You may also define :**

* `--device` id.
* `--diffuser_name` sampler type in diffuser (e.g. 'ddpm','ddim','dpmsolver'), related settings see [./config/diffuser_params.yaml](config/diffuser_params.yaml)
* `--num_inference_steps` number of iterative denoising steps during inference
* `--seed` to sample different prompts.
* `--motion_length`  in seconds 
* `--opt_path` for loading model
* `--footskate_cleanup` to use footskate module in the diffusion framework

**You will get :**

* `output_dir/joints_npy/xx.npy` - xyz pose sequence of the generated motion
* `output_dir/xx.mp4` - visual animation for generated motion.

outputdir is located in the ckeckpoint dir like `checkpoints/t2m/t2m_condunet1d_batch64/samples_t2m_condunet1d_batch64_50173_seed0_a_person_waves_with_his_right_hand/`.

The visual animation will look something like this:

![example](./assets/wave.gif)

##  Train and Evaluation

### 1. Download datasets 

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./data/HumanML3D
```

**KIT** - Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) (no processing needed this time) and the place result in `./data/KIT-ML`
</details>

### 2. Download pretrained weights for evaluation
We use the same evaluation protocol as [this repo](https://github.com/EricGuo5513/text-to-motion). You should download pretrained weights of the contrastive models in [t2m](https://drive.google.com/file/d/1DSaKqWX2HlwBtVH5l7DdW96jeYUIXsOP/view) and [kit](https://drive.google.com/file/d/1tX79xk0fflp07EZ660Xz1RAFE33iEyJR/view) for calculating FID and precisions. To dynamically estimate the length of the target motion, `length_est_bigru` and [Glove data](https://drive.google.com/drive/folders/1qxHtwffhfI4qMwptNW6KJEDuT6bduqO7?usp=sharing) are required.

Unzipped all files and arrange them in the following file structure:

```text
StableMoFusion 
└── data
    ├── glove
    │   ├── our_vab_data.npy
    │   ├── our_vab_idx.pkl
    │   └── out_vab_words.pkl
    ├── pretrained_models
    │   ├── kit
    │   │   └── text_mot_match
    │   │       └── model
    │   │           └── finest.tar
    │   └── t2m
    │   │   ├── text_mot_match
    │   │   │   └── model
    │   │   │       └── finest.tar
    │   │   └── length_est_bigru
    │   │       └── model
    │   │           └── finest.tar
    ├── HumanML3D
    │   ├── new_joint_vecs
    │   │   └── ...
    │   ├── new_joints
    │   │   └── ...
    │   ├── texts
    │   │   └── ...
    │   ├── Mean.npy
    │   ├── Std.npy
    │   ├── test.txt
    │   ├── train_val.txt
    │   ├── train.txt
    │   └── val.txt
    ├── KIT-ML
    │   ├── new_joint_vecs
    │   │   └── ...
    │   ├── new_joints
    │   │   └── ...
    │   ├── texts
    │   │   └── ...
    │   ├── Mean.npy
    │   ├── Std.npy
    │   ├── test.txt
    │   ├── train_val.txt
    │   ├── train.txt
    │   └── val.txt
    |── kit_mean.npy
    |── kit_std.npy
    |── t2m_mean.npy
    |── t2m_std.npy
```

### 3. Train CondUnet1D Model
<details>
  <summary><b>HumanML3D</b></summary>

```shell 
bash train.sh
```
</details>


You may also define the `--config_file` for training on multi gpus.


### 4. Evaluate

<details>
  <summary><b>HumanML3D</b></summary>
```shell
bash eval.sh
```
</details>


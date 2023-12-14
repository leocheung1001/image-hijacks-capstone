# Understanding Defensive Strategies for Adversarial Attacks on Large Vision Language Models (AC297R Capstone Project)

This repository sources from https://github.com/euanong/image-hijacks and is used to generate adversarial image attacks over LLaVA-7b-chat model. 
## Setup

The code can be run under any environment with Python 3.9 and above. 

We use [poetry](https://python-poetry.org) for dependency management, which can be installed following the instructions [here](https://python-poetry.org/docs/#installation).

To build a virtual environment with the required packages, simply run

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```


To effectively run the code in this repository, the following resource configurations are recommended:
- Disk Storage: At least 35GB of available disk storage is required.
- GPU RAM: A minimum of 30GB GPU RAM is necessary. A GPU equivalent to or better than NVIDIA L40 should be sufficient.

## Train adversarial images for LLaVA-7b-chat by full patch noise perturbation
We show step-by-step how to generate adversarial images on LLaVA-7b-chat model. The pretrained lora weights can be downloaded from this [Hugging Face repo](https://huggingface.co/liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview). The default setting would be training the adversaries by adding noise to the whole patch by epsilon constraint.

To train these images, first download the LLaVA checkpoint:
```bash
poetry run python download.py models llava-llama-2-7b-chat
```

To get the list of jobs (with their job IDs) specified by this config file:
```bash
poetry run python experiments/exp_demo_imgs/config.py
```

To run job ID `N` without [wandb](https://wandb.ai/) logging:
```bash
# run w/o wandb
poetry run python run.py train \
--config_path experiments/exp_demo_imgs/config.py \
--log_dir experiments/exp_demo_imgs/logs \
--job_id N \
--playground
```

To run job ID `N` with [wandb](https://wandb.ai/) logging to `YOUR_WANDB_ENTITY/YOUR_WANDB_PROJECT`:
```bash
# log in HF using API key
pip install transformers[cli]
huggingface-cli login

# run w/ wandb
poetry run python run.py train \
--config_path experiments/exp_demo_imgs/config.py \
--log_dir experiments/exp_demo_imgs/logs \
--job_id N \
--wandb_entity YOUR_WANDB_ENTITY \
--wandb_project YOUR_WANDB_PROJECT \
--no-playground
```

## Train adversarial images for LLaVA-7b-chat by static and moving patches
To train the adversaries by static and moving patches, we simply need to change the following function [sweep_patches](https://github.com/leocheung1001/image-hijacks-capstone/blob/8293c03d5ddcf529df8d3f3c134413a3626dd5a2/experiments/exp_demo_imgs/config.py#L134).

```python
def sweep_patches(cur_keys: List[str]) -> List[Transform]:
    return [
        Transform(
            [
                cfg.proc_learnable_image,
                lambda c: cfg.set_input_image(c, EIFFEL_IMAGE),
            ],
            "pat_full",
        )
    ]
```

Here, we can change `cfg.proc_learnable_image` to `cfg.proc_patch_static` for static patches or `cfg.proc_patch_random_loc` for moving patches. These functions can be found in code [config.py](https://github.com/leocheung1001/image-hijacks-capstone/blob/8293c03d5ddcf529df8d3f3c134413a3626dd5a2/image_hijacks/config.py#L272).



## Reference
```bibtex
@misc{bailey2023image,
  title={Image Hijacks: Adversarial Images can Control Generative Models at Runtime}, 
  author={Luke Bailey and Euan Ong and Stuart Russell and Scott Emmons},
  year={2023},
  eprint={2309.00236},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

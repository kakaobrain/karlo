# Karlo-v1.0.alpha on COYO-100M and CC15M

Karlo is a text-conditional image generation model based on OpenAI's unCLIP architecture with the improvement over the standard super-resolution model from 64px to 256px, recovering high-frequency details only in the small number of denoising steps.

<p float="left">
  <img src="/assets/example.gif"/>
</p>

<details>
  <summary>"a portrait of an old monk, highly detailed."</summary>
  <p float="left">
    <img src="/assets/a portrait of an old monk, highly detailed.png"/>
  </p>
</details>
<details>
  <summary>"Photo of a business woman, silver hair"</summary>
  <p float="left">
    <img src="/assets/Photo of a business woman, silver hair.png"/>
  </p>
</details>
<details>
  <summary>"A teddy bear on a skateboard, children drawing style."</summary>
  <p float="left">
    <img src="/assets/A teddy bear on a skateboard, children drawing style..png"/>
  </p>
</details>
<details>
  <summary>"Goryeo celadon in the shape of bird"</summary>
  <p float="left">
    <img src="/assets/Goryeo celadon in the shape of bird.png"/>
  </p>
</details>

This alpha version of Karlo is trained on 115M image-text pairs, including [COYO](https://github.com/kakaobrain/coyo-dataset)-100M high-quality subset, CC3M, and CC12M. For those who are interested in a better version of Karlo trained on more large-scale high-quality datasets, please visit the landing page of our application [B^DISCOVER](https://bdiscover.kakaobrain.com/).

### Updates
* [2022-12-01] Karlo-v1.0.alpha is released!

## Model Architecture

### Overview
Karlo is a text-conditional diffusion model based on unCLIP, composed of prior, decoder, and super-resolution modules. In this repository, we include the improved version of the standard super-resolution module for upscaling 64px to 256px only in 7 reverse steps, as illustrated in the figure below:

<p float="left">
  <img src="/assets/improved_sr_arch.png"/>
</p>

In specific, the standard SR module trained by DDPM objective upscales 64px to 256px in the first 6 denoising steps based on the respacing technique. Then, the additional fine-tuned SR module trained by [VQ-GAN](https://compvis.github.io/taming-transformers/)-style loss performs the final reverse step to recover high-frequency details. We observe that this approach is very effective to upscale the low-resolution in a small number of reverse steps.

### Details
We train all components from scratch on 115M image-text pairs including COYO-100M, CC3M, and CC12M. In the case of Prior and Decoder, we use ViT-L/14 provided by OpenAI’s [CLIP repository](https://github.com/openai/CLIP). Unlike the original implementation of unCLIP, we replace the trainable transformer in the decoder into the text encoder in ViT-L/14 for efficiency. In the case of the SR module, we first train the model using the DDPM objective in 1M steps, followed by additional 234K steps to fine-tune the additional component. The table below summarizes the important statistics of our components:

| | Prior | Decoder | SR |
|:------|----:|----:|----:|
| CLIP | ViT-L/14 | ViT-L/14 | - |
| #param | 1B | 900M | 700M + 700M |
| #optimization steps | 1M | 1M | 1M + 0.2M |
| #sampling steps | 25 | 50 (default), 25 (fast) | 7 |
|Checkpoint links| [ViT-L-14](https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/096db1af569b284eb76b3881534822d9/ViT-L-14.pt), [ViT-L-14 stats](https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th), [model](https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/efdf6206d8ed593961593dc029a8affa/decoder-ckpt-step%3D01000000-of-01000000.ckpt) | [model](https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/85626483eaca9f581e2a78d31ff905ca/prior-ckpt-step%3D01000000-of-01000000.ckpt) | [model](https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/4226b831ae0279020d134281f3c31590/improved-sr-ckpt-step%3D1.2M.ckpt) |

In the checkpoint links, ViT-L-14 is equivalent to the original version, but we include it for convenience. We also remark that ViT-L-14-stats is required to normalize the outputs of the prior module.

### Evaluation
We quantitatively measure the performance of Karlo-v1.0.alpha in the validation split of CC3M and MS-COCO. The table below presents CLIP-score and FID. To measure FID, we resize the image of the shorter side to 256px, followed by cropping it at the center. We set classifier-free guidance scales for prior and decoder to 4 and 8 in all cases. We observe that our model achieves reasonable performance even with 25 sampling steps of decoder. 

CC3M
| Sampling step | CLIP-s (ViT-B/16) | FID (13k from val)|
|:------|----:|----:|
| Prior (25) + Decoder (25) + SR (7) | 0.3081 | 14.37 |
| Prior (25) + Decoder (50) + SR (7) | 0.3086 | 13.95 |

MS-COCO
| Sampling step | CLIP-s (ViT-B/16) | FID (30k from val)|
|:------|----:|----:|
| Prior (25) + Decoder (25) + SR (7) | 0.3192 | 15.24 |
| Prior (25) + Decoder (50) + SR (7) | 0.3192 | 14.43 |


For more information, please refer to the upcoming technical report.


## Environment Setup
We use a single V100 of 32GB VRAM for sampling under PyTorch >= 1.10 and CUDA >= 11. The following commands install additional python packages and get pretrained model checkpoints. Or, you can simply install the package and download the weights via [setup.sh](setup.sh)
- Additional python packages
```
pip install -r requirements.txt
```
- Model checkpoints
```
wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/096db1af569b284eb76b3881534822d9/ViT-L-14.pt -P $KARLO_ROOT_DIR  # same with the official ViT-L/14 from OpenAI
wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/0b62380a75e56f073e2844ab5199153d/ViT-L-14_stats.th -P $KARLO_ROOT_DIR
wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/efdf6206d8ed593961593dc029a8affa/decoder-ckpt-step%3D01000000-of-01000000.ckpt -P $KARLO_ROOT_DIR
wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/85626483eaca9f581e2a78d31ff905ca/prior-ckpt-step%3D01000000-of-01000000.ckpt -P $KARLO_ROOT_DIR
wget https://arena.kakaocdn.net/brainrepo/models/karlo-public/v1.0.0.alpha/4226b831ae0279020d134281f3c31590/improved-sr-ckpt-step%3D1.2M.ckpt -P $KARLO_ROOT_DIR
```

## Sampling

### Gradio demo (T2I and Image variation)
The following command launches gradio demo for text-to-image generation and image variation. We notice that the second run in the gradio is unexpectedly slower than the usual case in PyTorch>=1.12. We guess that this happens because launching the cuda kernels takes some time, usually up to 2 minutes.
```
python demo/product_demo.py --host 0.0.0.0 --port $PORT --root-dir $KARLO_ROOT_DIR
```

Samples below are non-cherry picked T2I and image variation examples of random seed 0.
In each case, the first row shows T2I samples and the second shows the image variation samples of the leftmost image in the first row.

<details>
  <summary>[T2I + Image variation] "A man with a face of avocado, in the drawing style of Rene Magritte."</summary>
  <p float="left">
    <img src="/assets/A man with a face of avocado, in the drawing style of Rene Magritte..png"/>
    <img src="/assets/variation_A man with a face of avocado, in the drawing style of Rene Magritte..png"/>
  </p>
</details>

<details>
  <summary>[T2I + Image variation] "a black porcelain in the shape of pikachu"</summary>
  <p float="left">
    <img src="/assets/a black porcelain in the shape of pikachu.png"/>
    <img src="/assets/variation_a black porcelain in the shape of pikachu.png"/>
  </p>
</details>


### T2I command line example
Here, we include the command line example of T2I. For image variation, you can refer to [karlo/sampler/i2i.py](karlo/sampler/i2i.py) on how to replace the prior into the clip image feature.
```python
python example.py --root-dir=$KARLO_ROOT_DIR \
                  --prompt="A man with a face of avocado, in the drawing style of Rene Magritte" \
                  --output-dir=$OUTPUT_DIR \
                  --max-bsz=2 \
                  --sampling-type=fast
```

## Licence and Disclaimer
This project including the weights are distributed under [CreativeML Open RAIL-M license](LICENSE), equivalent version of [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE). You may use this model in commercial applications, but it is highly recommended to adopt a powerful safe checker as a post-processing. We also remark that we are not responsible for any kinds of use of the generated images.

## BibTex
If you find this repository useful in your research, please cite:
```
@misc{kakaobrain2022karlo-v1-alpha,
  title         = {Karlo-v1.0.alpha on COYO-100M and CC15M},
  author        = {Donghoon Lee, Jiseob Kim, Jisu Choi, Jongmin Kim, Minwoo Byeon, Woonhyuk Baek and Saehoon Kim},
  year          = {2022},
  howpublished  = {\url{https://github.com/kakaobrain/karlo}},
}
```

## Acknowledgement
We deeply appreciate all the contributors to OpenAI’s [Guided-Diffusion](https://github.com/openai/guided-diffusion) project.

## Contact
If you would like to collaborate with us or share a feedback, please e-mail to us, contact@kakaobrain.com

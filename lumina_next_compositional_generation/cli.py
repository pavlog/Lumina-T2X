import argparse
import builtins
import json
import math
import os
import random
import socket
import traceback

import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image

import models
from transport import Sampler, create_transport


class ModelFailure:
    pass


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

@torch.no_grad()
class LuminaComposition:
    def __init__(self,precision="bf16",hf_token=None,checkpoint="D:/checkpoints/Lumina-Next-T2I/",useEma=True,solver="midpoint",path_type="Linear",prediction="velocity",sampling_steps=30):
        # import here to avoid huggingface Tokenizer parallelism warnings
        from diffusers.models import AutoencoderKL
        from transformers import AutoModel, AutoTokenizer

        # override the default print function since the delay can be large for child process
        original_print = builtins.print

        # Redefine the print function with flush=True by default
        def print(*args, **kwargs):
            kwargs.setdefault("flush", True)
            original_print(*args, **kwargs)

        # Override the built-in print with the new version
        builtins.print = print

        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        os.environ["MASTER_PORT"] = str(12354)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

        dist.init_process_group("gloo")
        # set up fairscale environment because some methods of the Lumina model need it,
        # though for single-GPU inference fairscale actually has no effect
        fs_init.initialize_model_parallel(1)
        torch.cuda.set_device(0)

        self.train_args = torch.load(checkpoint+"model_args.pth")
        print("Loaded model arguments:", json.dumps(self.train_args.__dict__, indent=2))

        print(f"Creating lm: Gemma-2B")

        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        self.text_encoder = AutoModel.from_pretrained(
            "google/gemma-2b", torch_dtype=self.dtype, device_map="cuda", token=hf_token
        ).eval()
        cap_feat_dim = self.text_encoder.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=hf_token)
        self.tokenizer.padding_side = "right"

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae",
            torch_dtype=torch.float32,
        ).cuda()

        print(f"Creating DiT: {self.train_args.model}")
        # latent_size = train_args.image_size // 8
        self.model = models.__dict__[self.train_args.model](
            qk_norm=self.train_args.qk_norm,
            cap_feat_dim=cap_feat_dim,
        )
        self.model.eval().to("cuda", dtype=self.dtype)

        ckpt = load_file(
            os.path.join(
                checkpoint,
                f"consolidated{'_ema' if useEma else ''}.{0:02d}-of-{1:02d}.safetensors",
            ),
            device="cpu",
        )
        self.model.load_state_dict(ckpt, strict=True)

        # begin sampler
        self.transport = create_transport(
            path_type,
            prediction,
            None, #args.loss_weight,
            None,
            None,
        )
        self.sampler = Sampler(self.transport)
        self.solver = solver
        self.sampling_steps = sampling_steps
        # end sampler


    @torch.no_grad()
    def generate(self, cap1,
                    cap2,
                    cap3,
                    cap4,
                    neg_cap,
                    resolution,
                    cfg_scale,
                    t_shift,
                    seed,
                    scaling_method,
                    scaling_watershed,
                    proportional_attn):

        with torch.autocast("cuda", self.dtype):

            sample_fn = self.sampler.sample_ode(
                sampling_method=self.solver,
                num_steps=self.sampling_steps,
                atol=1e-6,
                rtol=1e-3,
                reverse=False,
                time_shifting_factor=4,
                )

            metadata = dict(
                cap1=cap1,
                cap2=cap2,
                cap3=cap3,
                cap4=cap4,
                neg_cap=neg_cap,
                resolution=resolution,
                num_sampling_steps=self.sampling_steps,
                cfg_scale=cfg_scale,
                solver=solver,
                t_shift=t_shift,
                seed=seed,
                scaling_method=scaling_method,
                scaling_watershed=scaling_watershed,
                proportional_attn=proportional_attn,
            )
            print("> params:", json.dumps(metadata, indent=2))

            do_extrapolation = "Extrapolation" in resolution
            split = resolution.split(" ")[1].replace("(", "")
            w_split, h_split = split.split("x")
            resolution = resolution.split(" ")[0]
            w, h = resolution.split("x")
            w, h = int(w), int(h)
            latent_w, latent_h = w // 8, h // 8
            if int(seed) != 0:
                torch.random.manual_seed(int(seed))
            z = torch.randn([1, 4, latent_h, latent_w], device="cuda").to(self.dtype)
            z = z.repeat(2, 1, 1, 1)

            cap_list = [cap1, cap2, cap3, cap4]
            global_cap = " ".join(cap_list)
            with torch.no_grad():
                if neg_cap != "":
                    cap_feats, cap_mask = encode_prompt(
                        cap_list + [neg_cap] + [global_cap], self.text_encoder, self.tokenizer, 0.0
                    )
                else:
                    cap_feats, cap_mask = encode_prompt(
                        cap_list + [""] + [global_cap], self.text_encoder, self.tokenizer, 0.0
                    )

            cap_mask = cap_mask.to(cap_feats.device)

            model_kwargs = dict(
                cap_feats=cap_feats[:-1],
                cap_mask=cap_mask[:-1],
                global_cap_feats=cap_feats[-1:],
                global_cap_mask=cap_mask[-1:],
                cfg_scale=cfg_scale,
                h_split_num=int(h_split),
                w_split_num=int(w_split),
            )
            if proportional_attn:
                model_kwargs["proportional_attn"] = True
                model_kwargs["base_seqlen"] = (self.train_args.image_size // 16) ** 2
            else:
                model_kwargs["proportional_attn"] = False
                model_kwargs["base_seqlen"] = None

            if do_extrapolation and scaling_method == "Time-aware":
                model_kwargs["scale_factor"] = math.sqrt(w * h / self.train_args.image_size**2)
                model_kwargs["scale_watershed"] = scaling_watershed
            else:
                model_kwargs["scale_factor"] = 1.0
                model_kwargs["scale_watershed"] = 1.0

            if dist.get_rank() == 0:
                print(f"> caption: {global_cap}")
                print(f"> num_sampling_steps: {self.sampling_steps}")
                print(f"> cfg_scale: {cfg_scale}")

            print("> start sample")
            samples = sample_fn(z, self.model.forward_with_cfg, **model_kwargs)[-1]
            samples = samples[:1]

            factor = 0.18215 if self.train_args.vae != "sdxl" else 0.13025
            print(f"> vae factor: {factor}")
            samples = self.vae.decode(samples / factor).sample
            samples = (samples + 1.0) / 2.0
            samples.clamp_(0.0, 1.0)

            return to_pil_image(samples[0].float())




if __name__ == "__main__":

    negative_prompt = "aerial,bird view,(((close view))),text,unnatural,noise,fish eye, spherical lens,lens flare,glare,panfuturism,window,border,stylized digital illustration,old cgi,dark,poor details,anime,cartoon,Ugly,"
    negative_prompt+= "(((blur))),(((blurry))),low resolution,animals,DOF,(((depth of field)),(((fog))),(((haze))),people,labels,text,logo,vignetting,letters,captions,copyright,watermark,unaestheticXLv31,"
    negative_prompt+= "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, duplicate, morbid, mutilated, mutation, deformed, disfigured, gross proportions,"
    negative_prompt+= "username, signature"


    solver = "midpoint"#,"euler", "midpoint", "rk4"
    path_type = "Linear" # the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).
    prediction = "velocity" #=["velocity", "score", "noise"],
    composition = LuminaComposition(solver=solver,path_type=path_type,prediction=prediction)
    image = composition.generate(
        cap1="cave",
        cap2="canyon",
        cap3="shallow river, trees, big rocks",
        cap4="shallow river with stones on sides",
        neg_cap=negative_prompt,
        resolution="1024x1024 (2x2 Grids)",
        cfg_scale=4,
        t_shift = 6,
        scaling_method=  "Time-aware",
        scaling_watershed=  0.3,
        proportional_attn=True,
        seed=1,
        )
    image.save("D:/Projects/sdexperiments/PointsInpaint/Lumina/b.png")

#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import sys
sys.path.insert(0,"D:/Projects/diffusers/src")
import argparse
import json
import os
import socket
import sys
import PIL

from diffusers.models import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch as pt
import torch.distributed as dist
from torchvision.utils import save_image

import models
from transport import Sampler, create_transport
from diffusers.utils import load_image
from torchvision import transforms

def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class SamplePipeline:



    def __init__( self, ckpt="D:/Projects/Lumina-T2X/Next-DiT-ImageNet/results2/checkpoints/",precision="tf32",ema=True, mode="ODE",path_type="Linear",prediction="velocity",
                atol=1e-6,rtol=1e-3,sampling_method="dopri5",likelihood=False, initMP=True):
        
        with pt.no_grad():
            self.likelihood = likelihood
            self.path_type = path_type
            self.prediction = prediction
            self.reverse = False
            #group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
            #group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
            #group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
            #group.add_argument("--sample-eps", type=float)
            #group.add_argument("--train-eps", type=float)

            self.atol = atol
            self.rtol = rtol
            self.mode = mode
            self.sampling_method = sampling_method

            self.image_normalize = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            num_gpus = 1
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            # Setup PyTorch:

            rank = 0

            if initMP:
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(num_gpus)
                os.environ["MASTER_PORT"] = str(find_free_port())
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ['TORCHDYNAMO_DISABLE']='1'

                dist.init_process_group("gloo")
                fs_init.initialize_model_parallel(num_gpus)
                torch.cuda.set_device(rank)

            # assert train_args.model_parallel_size == args.num_gpus
            dirs = os.listdir(ckpt)
            dirs = sorted(dirs, key=lambda x: int(x))
            path = dirs[-1] if len(dirs) > 0 else None
            ckpt_path = os.path.join(ckpt, path)

            self.train_args = torch.load(os.path.join(ckpt_path, "model_args.pth"))

            if dist.get_rank() == 0:
                print("Model arguments used for inference:", json.dumps(self.train_args.__dict__, indent=2))

            # Load model:
            self.image_size = self.train_args.image_size
            self.latent_size = self.image_size // 8
            #latent_size = image_size // 4
            self.model = models.__dict__[self.train_args.model](
                input_size=self.latent_size,
                num_classes=self.train_args.num_classes,
                qk_norm=self.train_args.qk_norm,
            )

            
            self.torch_dtype = {
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[precision]
            self.model.to(self.torch_dtype).cuda()
            if precision == "tf32":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            ckpt = torch.load(
                os.path.join(
                    ckpt_path,
                    f"consolidated{'_ema' if ema else ''}." f"{rank:02d}-of-{num_gpus:02d}.pth",
                ),
                map_location="cpu",
            )
            self.model.load_state_dict(ckpt, strict=True)

            self.train_steps = 0
            with open(os.path.join(ckpt_path, "resume_step.txt")) as f:
                self.train_steps = int(f.read().strip())

            self.model.eval()  # important!

            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.train_args.vae}").cuda()


    def sample(self,imageFilename,class_label,num_sampling_steps=250,seed=0,args=None):

        with pt.no_grad():
        
            transport = create_transport(self.path_type, self.prediction, None, None,None ) #args.loss_weight, args.train_eps, args.sample_eps)
            sampler = Sampler(transport)

            if self.mode == "ODE":
                if self.likelihood:
                    assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
                    sample_fn = sampler.sample_ode_likelihood(
                        sampling_method=self.sampling_method,
                        num_steps=num_sampling_steps,
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                else:
                    sample_fn = sampler.sample_ode(
                        sampling_method=self.sampling_method,
                        num_steps=num_sampling_steps,
                        atol=self.atol,
                        rtol=self.rtol,
                        reverse=self.reverse,
                    )

            elif self.mode == "SDE":
                sample_fn = sampler.sample_sde(
                    sampling_method=self.sampling_method,
                    diffusion_form=args.diffusion_form,
                    diffusion_norm=args.diffusion_norm,
                    last_step=args.last_step,
                    last_step_size=args.last_step_size,
                    num_steps=args.num_sampling_steps,
                )

            imageOrg = load_image(imageFilename)
            #
            #    #"D:\Projects\InpaintDataBase\Batch001_Train\cave_0006_0063.png")
            #"D:/Projects/sdexperiments/PointsInpaint/Test1/river19_0000_0002.png"
            ##"sample_61000_1a.png"
            #)
            imageOrg = self.image_normalize(imageOrg)
            imageOrg = imageOrg.unsqueeze(0)
            #imageOrg = imageOrg.repeat(n, 1, 1, 1) 
            imageOrg = imageOrg.to( device="cuda")
            org_latents = self.vae.encode(imageOrg).latent_dist.mode() #sample().mul_(vae.config.scaling_factor)

            # Create sampling noise:
            n = 1
            torch.manual_seed(seed)
            z = torch.randn(
                n,
                4,
                self.latent_size,
                self.latent_size,
                dtype=self.torch_dtype,
                device="cuda",
            )
            y = torch.tensor([class_label], device="cuda")

            input_latents = org_latents

            if class_label==6:
                input_latents = torch.zeros_like(z).to(z.device)

            # Setup classifier-free guidance:
            #z = torch.cat([z, z], 0)
            #y_null = torch.tensor([1000] * n, device="cuda")
            #y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y,input_latents = input_latents)#,cfg_scale=args.cfg_scale)

            # Sample images:
            #samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples = sample_fn(z, self.model.forward, **model_kwargs)[-1]
            # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            # samples = vae.decode(samples / 0.18215).sample
            dist.barrier()

            samples = self.vae.decode(samples / 0.18215).sample
            samples = samples.squeeze(0)

            ndarr = samples.clamp_(-1,1).sub_(-1).div_(max(1.0 - (-1.0), 1e-5)).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = PIL.Image.fromarray(ndarr)
            return im
'''
def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument(
        "--diffusion-form",
        type=str,
        default="sigma",
        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing"],
        help="form of diffusion coefficient in the SDE",
    )
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument(
        "--last-step",
        type=none_or_str,
        default="Mean",
        choices=[None, "Mean", "Tweedie", "Euler"],
        help="form of last step taken in the SDE",
    )
    group.add_argument("--last-step-size", type=float, default=0.04, help="size of the last step taken")
'''

if __name__ == "__main__":

    sampler = SamplePipeline(ckpt="D:/Projects/Lumina-T2X/Next-DiT-ImageNet/results3/checkpoints/")

    '''
    sys.argv.extend([
                #"--model=DiT_Llama_7B_patch2_Actions",
                #"--ckpt=D:/Projects/Lumina-T2X/Next-DiT-ImageNet/results/checkpoints/",
                #"--image_save_path=D:/Projects/Lumina-T2X/Next-DiT-ImageNet/results",

                #"--image_save_path=D:/Projects/Lumina-T2X/Next-DiT-ImageNet/results2",
                #"--ckpt=D:/Projects/Lumina-T2X/Next-DiT-ImageNet/results2/checkpoints/",
                #"--model=DiT_Llama_600_patch2_Actions2",

                ])


    parser = argparse.ArgumentParser()
    mode = sys.argv[1]
    if mode not in ["ODE", "SDE"]:
        mode = "ODE"

    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--class_labels",
        type=int,
        nargs="+",
        help="Class labels to generate the images for.",
        default=[1],
    )
    
    parser.add_argument(
        "--local_diffusers_model_root",
        type=str,
        help="Specify the root directory if diffusers models are to be loaded "
        "from the local filesystem (instead of being automatically "
        "downloaded from the Internet). Useful in environments without "
        "Internet access.",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument(
        "--image_save_path",
        type=str,
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    # args = parser.parse_args()
    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."
    main(args, 0, master_port)
    '''

    image_save_path="D:/Projects/Lumina-T2X/Next-DiT-ImageNet/results3"

    classes = [7]

    import shutil
    
    for class_label in classes:
        shutil.copy("D:/Projects/sdexperiments/PointsInpaint/Test1/river19_0000_0002.png",image_save_path+f"/sample_{sampler.train_steps}_{class_label}_0.png")
        shutil.copy("D:/Projects/sdexperiments/PointsInpaint/Test5/controlHorizon.png",image_save_path+f"/sample_{sampler.train_steps}_{class_label}_0.png")
        shutil.copy("D:/Projects/InpaintDataBase/Batch3D_001/AUG1/river19_0003_0017_1_seg.png",image_save_path+f"/sample_{sampler.train_steps}_{class_label}_0.png")
        

        #shutil.copy("D:/Projects/InpaintDataBase/Batch3D_002/river98_0001.png",args.image_save_path+f"/sample_{train_steps}_{args.class_labels[0]}_0.png")
        
        #shutil.copy("results/testNoise.png",image_save_path+f"/sample_{sampler.train_steps}_{class_label}_0.png")
        #shutil.copy("D:/Projects/InpaintDataBase/Batch3D_002/05SL/river75_0008_05SL_rti.png",args.image_save_path+f"/sample_{train_steps}_{args.class_labels[0]}_0.png")
        
        for i in range(10):
            imageFile = image_save_path+f"/sample_{sampler.train_steps}_{class_label}_{i}.png"
            image = sampler.sample(imageFile,class_label,seed=i)
            image.save(image_save_path+f"/sample_{sampler.train_steps}_{class_label}_{i+1}.png")
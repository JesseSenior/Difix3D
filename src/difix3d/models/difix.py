import torch
from diffusers import AutoencoderKL, DDPMScheduler
from einops import rearrange, repeat
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


class Difix(torch.nn.Module):
    def __init__(
        self,
        pretrained_name=None,
        pretrained_path=None,
        ckpt_folder="checkpoints",
        mv_unet=False,
        timestep=999,
        verbose=False,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")

        if mv_unet:
            from difix3d.models.mv_unet import UNet2DConditionModel
        else:
            from diffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if pretrained_path is not None:
            sd = torch.load(pretrained_path, map_location="cpu")
            _sd_unet = unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            unet.load_state_dict(_sd_unet)

        elif pretrained_name is None and pretrained_path is None:
            print("Initializing model with random weights")

        # unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")

        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([timestep], device="cuda").long()
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        if verbose:
            # print number of trainable parameters
            print("=" * 50)
            print(
                f"Number of trainable parameters in UNet: {sum(p.numel() for p in unet.parameters() if p.requires_grad) / 1e6:.2f}M"
            )
            print("=" * 50)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.unet.requires_grad_(True)

    def forward(self, x, timesteps=None, prompt=None, prompt_tokens=None):
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"
        assert (timesteps is None) != (self.timesteps is None), "Either timesteps or self.timesteps should be provided"

        if prompt is not None:
            # encode the text prompt
            caption_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.cuda()
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        num_views = x.shape[1]
        x = rearrange(x, "b v c h w -> (b v) c h w")
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        caption_enc = repeat(caption_enc, "b n c -> (b v) n c", v=num_views)

        unet_input = z

        model_pred = self.unet(
            unet_input,
            self.timesteps,
            encoder_hidden_states=caption_enc,
        ).sample
        z_denoised = self.sched.step(model_pred, self.timesteps, z, return_dict=True).prev_sample
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        output_image = rearrange(output_image, "(b v) c h w -> b v c h w", v=num_views)

        return output_image

    def sample(self, image, width, height, ref_image=None, timesteps=None, prompt=None, prompt_tokens=None):
        target_width = width - width % 8
        target_height = height - height % 8

        T = transforms.Compose(
            [
                transforms.Resize((target_height, target_width), interpolation=Image.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        if ref_image is None:
            x = T(image).unsqueeze(0).unsqueeze(0).cuda()
        else:
            x = torch.stack([T(image), T(ref_image)], dim=0).unsqueeze(0).cuda()

        output_image = self.forward(x, timesteps, prompt, prompt_tokens)[:, 0]
        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

        if target_width != width or target_height != height:
            output_pil = output_pil.resize((width, height), Image.LANCZOS)

        return output_pil

    def save_model(self, outf, optimizer):
        sd = {}
        sd["state_dict_unet"] = self.unet.state_dict()
        sd["optimizer"] = optimizer.state_dict()

        torch.save(sd, outf)

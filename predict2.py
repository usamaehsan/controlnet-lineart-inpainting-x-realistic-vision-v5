from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
from controlnet_aux import LineartDetector
from compel import Compel
import os
from PIL import Image

MODEL_NAME = "SG161222/Realistic_Vision_V5.0_noVAE"
MODEL_CACHE = "cache"
VAE_CACHE = "vae-cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # vae = AutoencoderKL.from_single_file(
        #     "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
        #     cache_dir=VAE_CACHE
        # )
        #inpaint controlnet
        # controlnet = ControlNetModel.from_pretrained(
        #     "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
        # )
        lineart_controlnet = ControlNetModel.from_pretrained("ControlNet-1-1-preview/control_v11p_sd15_lineart", torch_dtype=torch.float16)
        self.lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")

        controlnets= [lineart_controlnet]

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.0_noVAE", controlnet=controlnets, torch_dtype=torch.float16,
            # vae=vae,
        )
        self.compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        self.pipe = pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()


    def resize_image(self, image, max_width, max_height):
        """
        Resize an image to a specific height while maintaining the aspect ratio and ensuring
        that neither width nor height exceed the specified maximum values.

        Args:
            image (PIL.Image.Image): The input image.
            max_width (int): The maximum allowable width for the resized image.
            max_height (int): The maximum allowable height for the resized image.

        Returns:
            PIL.Image.Image: The resized image.
        """
        # Get the original image dimensions
        original_width, original_height = image.size

        # Calculate the new dimensions to maintain the aspect ratio and not exceed the maximum values
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height

        # Choose the smallest ratio to ensure that neither width nor height exceeds the maximum
        resize_ratio = min(width_ratio, height_ratio)

        # Calculate the new width and height
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        return resized_image


    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def closest_multiple_of_8(self, width, height):
        # Calculate the closest multiple of 8 for width
        closest_width = ((width + 7) // 8) * 8

        # Calculate the closest multiple of 8 for height
        closest_height = ((height + 7) // 8) * 8

        return closest_width, closest_height

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = "(a tabby cat)+++, high resolution, sitting on a park bench",
        # mask: Path = Input(description="Mask image"),
        negative_prompt: str = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck",
        strength: float = Input(description="control strength/weight", ge=0, le=2, default=0.8),
        max_height: float = Input(description="max height of mask/image", ge=128, default=612),
        max_width: float = Input(description="max width of mask/image", ge=128, default=612),
        steps: int = Input(description=" num_inference_steps", ge=0, le=100, default=20),
        seed: int = Input(description="Leave blank to randomize",  default=None),
        guidance_scale: int = Input(description="guidance_scale", ge=0, le=30, default=10),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed == 0) or (seed == None):
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)

        init_image = Image.open(image).convert("RGB")
        init_image = self.resize_image(init_image, max_width, max_height)
        width, height = init_image.size
        width,height= self.closest_multiple_of_8( width, height)
        init_image= init_image.resize((width,height))
        # mask_image = Image.open(mask).convert("L").resize((width,height))
        # inpainting_control_image = self.make_inpaint_condition(init_image, mask_image)
        lineart_control_image = self.lineart_processor(init_image)
        lineart_control_image= lineart_control_image.resize((width,height))
        
        images= [lineart_control_image]
        
        image = self.pipe(
            prompt_embeds=self.compel_proc(prompt),
            negative_prompt_embeds=self.compel_proc(negative_prompt),
            num_inference_steps=steps,
            generator=generator,
            eta=1,
            image=images,
            # mask_image=mask_image,
            # control_image=images,
            controlnet_conditioning_scale= strength,
            guidance_scale= guidance_scale
        ).images[0]

        out_path = Path(f"/tmp/output.png")
        image.save(out_path)
        return  out_path



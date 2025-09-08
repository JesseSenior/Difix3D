import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import functional as TF


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.dataset_root = Path(dataset_path).parent
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

    def transform(self, input_img, output_img, ref_img=None):
        # Resize to slightly larger size for random crop
        resize_h, resize_w = int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1)
        resize = v2.Resize(size=(resize_h, resize_w))
        input_img = resize(input_img)
        output_img = resize(output_img)
        if ref_img is not None:
            ref_img = resize(ref_img)

        # Random crop with same parameters
        i, j, h, w = v2.RandomCrop.get_params(input_img, output_size=self.image_size)
        input_img = TF.crop(input_img, i, j, h, w)
        output_img = TF.crop(output_img, i, j, h, w)
        if ref_img is not None:
            ref_img = TF.crop(ref_img, i, j, h, w)

        # Random rotation with same angle
        angle = v2.RandomRotation.get_params(degrees=[-10, 10])
        input_img = TF.rotate(input_img, angle)
        output_img = TF.rotate(output_img, angle)
        if ref_img is not None:
            ref_img = TF.rotate(ref_img, angle)

        # Random horizontal flipping
        if random.random() > 0.5:
            input_img = TF.hflip(input_img)
            output_img = TF.hflip(output_img)
            if ref_img is not None:
                ref_img = TF.hflip(ref_img)

        # Convert to tensor and normalize
        transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
        ])
        
        input_img = transforms(input_img)
        output_img = transforms(output_img)
        if ref_img is not None:
            ref_img = transforms(ref_img)
            
        return input_img, output_img, ref_img

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        input_img_path = self.dataset_root / self.data[img_id]["image"]
        output_img_path = self.dataset_root / self.data[img_id]["target_image"]
        ref_img_path = self.dataset_root / self.data[img_id]["ref_image"] if "ref_image" in self.data[img_id] else None
        caption = self.data[img_id]["prompt"]

        try:
            input_img = Image.open(input_img_path)
            output_img = Image.open(output_img_path)
            if ref_img_path is not None:
                ref_img = Image.open(ref_img_path)
        except Exception:
            print(f"Error loading image: {input_img_path}|{output_img_path}|{ref_img_path}. Use next idx")
            return self.__getitem__(idx + 1)

        # Apply same random transforms to all images
        if ref_img_path is not None:
            input_img, output_img, ref_img = self.transform(input_img, output_img, ref_img)
            input_img = torch.stack([input_img, ref_img], dim=0)
            output_img = torch.stack([output_img, ref_img], dim=0)
        else:
            input_img, output_img, _ = self.transform(input_img, output_img)
            input_img = input_img.unsqueeze(0)
            output_img = output_img.unsqueeze(0)

        out = {
            "output_pixel_values": output_img,
            "conditioning_pixel_values": input_img,
            "caption": caption,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids
            out["input_ids"] = input_ids

        return out

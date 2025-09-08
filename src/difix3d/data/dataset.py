import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=576, width=1024, tokenizer=None):
        super().__init__()
        with open(dataset_path, "r") as f:
            self.data = json.load(f)[split]
        self.dataset_root = Path(dataset_path).parent
        self.img_ids = list(self.data.keys())
        self.image_size = (height, width)
        self.tokenizer = tokenizer

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

        transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(self.image_size),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        input_img = transforms(input_img)
        output_img = transforms(output_img)

        if ref_img_path is not None:
            ref_img = transforms(Image.open(ref_img))

            input_img = torch.stack([input_img, ref_img], dim=0)
            output_img = torch.stack([output_img, ref_img], dim=0)
        else:
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

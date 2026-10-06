"""Decoder-only fine-tuning utilities for Ultralytics SAM Base (ViT-B).

SAM-B is promptable segmentation, not an object detector. Evaluation in this
module uses one ground-truth bounding-box prompt per instance.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoSamPromptDataset(Dataset):
    """COCO instances converted to SAM image, box-prompt and low-res mask tensors."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        image_size: int = 1024,
        mask_size: int = 256,
        training: bool = False,
        box_jitter_fraction: float = 0.03,
    ) -> None:
        self.split_dir = Path(dataset_dir) / split
        self.coco = COCO(str(self.split_dir / "_annotations.coco.json"))
        self.image_ids = sorted(self.coco.getImgIds())
        self.image_size = image_size
        self.mask_size = mask_size
        self.training = training
        self.box_jitter_fraction = box_jitter_fraction

    def __len__(self) -> int:
        return len(self.image_ids)

    def _jitter_boxes(self, boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        boxes = boxes.copy()
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        noise = np.random.uniform(
            -self.box_jitter_fraction,
            self.box_jitter_fraction,
            size=boxes.shape,
        ).astype(np.float32)
        boxes += noise * np.concatenate([wh, wh], axis=1)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
        return boxes

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]
        info = self.coco.loadImgs([image_id])[0]
        image_path = self.split_dir / info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        annotations = [
            ann
            for ann in self.coco.loadAnns(ann_ids)
            if ann.get("segmentation") and ann["bbox"][2] > 0 and ann["bbox"][3] > 0
        ]
        boxes = np.asarray(
            [
                [
                    ann["bbox"][0],
                    ann["bbox"][1],
                    ann["bbox"][0] + ann["bbox"][2],
                    ann["bbox"][1] + ann["bbox"][3],
                ]
                for ann in annotations
            ],
            dtype=np.float32,
        ).reshape(-1, 4)

        hflip = self.training and random.random() < 0.5
        vflip = self.training and random.random() < 0.25
        if hflip:
            image = np.ascontiguousarray(image[:, ::-1])
            old_x0, old_x1 = boxes[:, 0].copy(), boxes[:, 2].copy()
            boxes[:, 0], boxes[:, 2] = width - old_x1, width - old_x0
        if vflip:
            image = np.ascontiguousarray(image[::-1, :])
            old_y0, old_y1 = boxes[:, 1].copy(), boxes[:, 3].copy()
            boxes[:, 1], boxes[:, 3] = height - old_y1, height - old_y0
        if self.training and len(boxes):
            boxes = self._jitter_boxes(boxes, width, height)

        scale = self.image_size / max(height, width)
        resized_h = max(1, int(round(height * scale)))
        resized_w = max(1, int(round(width * scale)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        resized_tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
        resized_tensor = (resized_tensor - pixel_mean) / pixel_std
        input_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        input_tensor[:, :resized_h, :resized_w] = resized_tensor

        boxes *= scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, resized_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, resized_h - 1)

        target_h = max(1, int(round(resized_h * self.mask_size / self.image_size)))
        target_w = max(1, int(round(resized_w * self.mask_size / self.image_size)))
        target_masks = torch.zeros(
            (len(annotations), self.mask_size, self.mask_size), dtype=torch.float32
        )
        for mask_index, annotation in enumerate(annotations):
            mask = self.coco.annToMask(annotation)
            if hflip:
                mask = np.ascontiguousarray(mask[:, ::-1])
            if vflip:
                mask = np.ascontiguousarray(mask[::-1, :])
            resized_mask = cv2.resize(
                mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
            target_masks[mask_index, :target_h, :target_w] = torch.from_numpy(
                (resized_mask > 0).astype(np.float32)
            )

        return {
            "image": input_tensor,
            "boxes": torch.from_numpy(boxes),
            "masks": target_masks,
            "image_id": image_id,
            "image_path": str(image_path),
            "original_size": (height, width),
            "resized_size": (resized_h, resized_w),
        }


def single_item_collate(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError(
            "SAM prompt training requires batch_size=1; prompt_chunk_size batches instances."
        )
    return batch[0]


def freeze_for_decoder_finetune(sam_model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for parameter in sam_model.parameters():
        parameter.requires_grad = False
    for parameter in sam_model.mask_decoder.parameters():
        parameter.requires_grad = True
    sam_model.image_encoder.eval()
    sam_model.prompt_encoder.eval()
    sam_model.mask_decoder.train()
    return [parameter for parameter in sam_model.parameters() if parameter.requires_grad]


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def binary_mask_metrics(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    predictions = logits.sigmoid() > 0.5
    truth = targets > 0.5
    intersection = (predictions & truth).flatten(1).sum(1).float()
    union = (predictions | truth).flatten(1).sum(1).float()
    pred_area = predictions.flatten(1).sum(1).float()
    true_area = truth.flatten(1).sum(1).float()
    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_area + true_area + eps)
    return iou, dice


def forward_prompt_chunk(
    sam_model: torch.nn.Module,
    image_embedding: torch.Tensor,
    boxes: torch.Tensor,
):
    with torch.no_grad():
        sparse, dense = sam_model.prompt_encoder(points=None, boxes=boxes, masks=None)
        image_pe = sam_model.prompt_encoder.get_dense_pe()
    low_res_masks, quality_scores = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return low_res_masks[:, 0], quality_scores[:, 0]


def run_epoch(
    sam_model: torch.nn.Module,
    loader,
    device: torch.device,
    amp_dtype: torch.dtype,
    prompt_chunk_size: int,
    training: bool,
    optimizer=None,
    scaler=None,
    trainable_parameters=None,
) -> dict:
    sam_model.image_encoder.eval()
    sam_model.prompt_encoder.eval()
    sam_model.mask_decoder.train(training)
    totals = defaultdict(float)

    for sample in loader:
        boxes = sample["boxes"].to(device, non_blocking=True)
        targets = sample["masks"].to(device, non_blocking=True)
        if len(boxes) == 0:
            continue
        image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=True):
            image_embedding = sam_model.image_encoder(image)

        if training:
            optimizer.zero_grad(set_to_none=True)
        chunk_count = math.ceil(len(boxes) / prompt_chunk_size)
        for start in range(0, len(boxes), prompt_chunk_size):
            stop = min(start + prompt_chunk_size, len(boxes))
            box_chunk = boxes[start:stop]
            target_chunk = targets[start:stop]
            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context, torch.autocast("cuda", dtype=amp_dtype, enabled=True):
                logits, quality = forward_prompt_chunk(
                    sam_model, image_embedding, box_chunk
                )
                bce = F.binary_cross_entropy_with_logits(logits, target_chunk)
                dice_loss = soft_dice_loss(logits, target_chunk)
                true_iou, true_dice = binary_mask_metrics(
                    logits.detach(), target_chunk
                )
                quality_loss = F.mse_loss(
                    quality.float(), true_iou.detach().float()
                )
                loss = bce + dice_loss + 0.1 * quality_loss
            if training:
                scaler.scale(loss / chunk_count).backward()
            count = stop - start
            totals["loss"] += float(loss.detach()) * count
            totals["bce"] += float(bce.detach()) * count
            totals["dice_loss"] += float(dice_loss.detach()) * count
            totals["iou"] += float(true_iou.sum())
            totals["dice"] += float(true_dice.sum())
            totals["instances"] += count

        if training:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

    denominator = max(1.0, totals["instances"])
    metrics = {
        key: value / denominator
        for key, value in totals.items()
        if key != "instances"
    }
    metrics["instances"] = int(totals["instances"])
    return metrics


def save_ultralytics_sam_checkpoint(
    path: str | Path,
    sam_model: torch.nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    metadata: dict,
) -> None:
    """Save a nested model dict accepted by Ultralytics' SAM checkpoint loader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {
        key: value.detach().cpu() for key, value in sam_model.state_dict().items()
    }
    torch.save(
        {
            "model": cpu_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_type": "sam_b",
            "training_scope": "mask_decoder_only",
            "metadata": metadata,
        },
        path,
    )

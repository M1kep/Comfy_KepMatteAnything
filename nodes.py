import os

import PIL
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from groundingdino.config import GroundingDINO_SwinT_OGC
from groundingdino.util.inference import Model as DinoModel
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T
from torch import Tensor

import folder_paths
import comfy.model_management

folder_paths.folder_names_and_paths["sams"] = ([os.path.join(folder_paths.models_dir, "sams")],
                                               folder_paths.supported_pt_extensions)


def to_dino_format(image):
    dino_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = dino_transform(Image.fromarray(image), None)
    return image_transformed

# https://github.com/WASasquatch/was-node-suite-comfyui/blob/4413ecd432aa1f13d8ad01a4f32fe8fe506e7051/WAS_Node_Suite.py#L374
# Tensor to SAM-compatible NumPy
def tensor2sam(image):
    # Convert tensor to numpy array in HWC uint8 format with pixel values in [0, 255]
    sam_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    # Transpose the image to HWC format if it's in CHW format
    if sam_image.shape[0] == 3:
        sam_image = np.transpose(sam_image, (1, 2, 0))
    return sam_image

class SAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("sams"), ),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("SAM_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Matte Anything"

    def load_model(self, model_name, device_mode="auto"):
        modelname = folder_paths.get_full_path("sams", model_name)

        if 'vit_h' in model_name:
            model_kind = 'vit_h'
        elif 'vit_l' in model_name:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam = sam_model_registry[model_kind](checkpoint=modelname)
        # Unless user explicitly wants to use CPU, we use GPU
        device = comfy.model_management.get_torch_device() if device_mode == "Prefer GPU" else "CPU"

        if device_mode == "Prefer GPU":
            sam.to(device=device)

        sam.is_auto_mode = device_mode == "AUTO"
        print(f"Loads SAM model: {modelname} (device:{device_mode})")
        return (sam,)

class InitSAMPredictor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "sam_model": ("SAM_MODEL", {}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("SAM_PREDICTOR",)
    FUNCTION = "init_predictor"

    CATEGORY = "Matte Anything"

    def init_predictor(self, image, sam_model):
        predictor = SamPredictor(sam_model)
        predictor.set_image(tensor2sam(image))
        return (predictor,)

# folder_paths.folder_names_and_paths["dinos"] = ([os.path.join(folder_paths.models_dir, "dinos")]
class LoadDINOModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("dino"), ),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("DINO_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Matte Anything"

    def load_model(self, model_name, device_mode="auto"):
        model = DinoModel(
            model_config_path=GroundingDINO_SwinT_OGC.__file__,
            model_checkpoint_path=folder_paths.get_full_path("dino", model_name),
            device="cpu"
        )
        return (model,)

def pil2cv(image: Image) -> np.ndarray:
    mode = image.mode
    new_image: np.ndarray
    if mode == '1':
        new_image = np.array(image, dtype=np.uint8)
        new_image *= 255
    elif mode == 'L':
        new_image = np.array(image, dtype=np.uint8)
    elif mode == 'LA' or mode == 'La':
        new_image = np.array(image.convert('RGBA'), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == 'RGB':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == 'RGBA':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    elif mode == 'LAB':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_LAB2BGR)
    elif mode == 'HSV':
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)
    elif mode == 'YCbCr':
        # XXX: not sure if YCbCr == YCrCb
        new_image = np.array(image, dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_YCrCb2BGR)
    elif mode == 'P' or mode == 'CMYK':
        new_image = np.array(image.convert('RGB'), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif mode == 'PA' or mode == 'Pa':
        new_image = np.array(image.convert('RGBA'), dtype=np.uint8)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f'unhandled image color mode: {mode}')

    return new_image

def tensor2pil(image: Tensor) -> PIL.Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: PIL.Image.Image) -> Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
class DinoBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "dino_model": ("DINO_MODEL", {}),
                "caption": ("STRING", {}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE", "DINO_BOXES")
    FUNCTION = "get_boxes"

    CATEGORY = "Matte Anything"

    def get_boxes(self, image, dino_model: DinoModel, caption: str):
        point_coords, point_labels = None, None
        image_cv = pil2cv((tensor2pil(image)))
        detections, labels = dino_model.predict_with_caption(
            image=image_cv,
            caption=caption,
        )
        # Detection is
        """
        Data class containing information about the detections in a video frame.
        Attributes:
            xyxy (np.ndarray): An array of shape `(n, 4)` containing the bounding boxes coordinates in format `[x1, y1, x2, y2]`
            mask: (Optional[np.ndarray]): An array of shape `(n, W, H)` containing the segmentation masks.
            confidence (Optional[np.ndarray]): An array of shape `(n,)` containing the confidence scores of the detections.
            class_id (Optional[np.ndarray]): An array of shape `(n,)` containing the class ids of the detections.
            tracker_id (Optional[np.ndarray]): An array of shape `(n,)` containing the tracker ids of the detections.
        """
        image_pil = tensor2pil(image)
        img_draw = ImageDraw.Draw(image_pil)
        for detection in detections:
            xyxy = detection[0]
            img_draw.rectangle(xyxy, outline="red", width=4)
        return (pil2tensor(image_pil),detections)

class SAMMaskFromBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_predictor": ("SAM_PREDICTOR", {}),
                "boxes": ("DINO_BOXES", {}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("MASK",)
    FUNCTION = "get_mask"

    CATEGORY = "Matte Anything"

    def get_mask(self, sam_predictor: SamPredictor, boxes):
        boxes_stacked = torch.stack([torch.tensor(box[0]) for box in boxes])
        masks, scores, logits = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes_stacked,
            multimask_output=False
        )

        return (torch.squeeze(masks),)

def generate_trimap(mask, erode_kernel_size=10, dilate_kernel_size=10):
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erode_kernel, iterations=5)
    dilated = cv2.dilate(mask, dilate_kernel, iterations=5)
    trimap = np.zeros_like(mask)
    trimap[dilated==255] = 128
    trimap[eroded==255] = 255
    return trimap

class MaskToTrimap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "erode_kernel_size": ("INT", {"min": 1, "step": 1}),
                "dilate_kernel_size": ("INT", {"min": 1, "step": 1}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("TRIMAP","MASK")
    FUNCTION = "get_trimap"

    CATEGORY = "Matte Anything"

    def get_trimap(self, mask: Tensor, erode_kernel_size: int, dilate_kernel_size: int):
        mask = mask.cpu().detach().numpy().astype(np.uint8)*255
        trimap = generate_trimap(mask, erode_kernel_size, dilate_kernel_size).astype(np.float32)
        trimap[trimap==128] = 0.5
        trimap[trimap==255] = 1
        trimap = torch.from_numpy(trimap).unsqueeze(0)
        # mask_np = mask.cpu().numpy()
        # mask_np = np.clip(mask_np, 0, 1)
        # mask_np = mask_np.astype(np.uint8)
        # mask_np = mask_np * 255
        # mask_np = cv2.GaussianBlur(mask_np, (5, 5), 0)
        # mask_np = mask_np.astype(np.float32) / 255.0
        # trimap = torch.from_numpy(mask_np).unsqueeze(0)
        return (trimap,trimap)


vitmatte_config = {
    'vit_b': './custom_nodes/Comfy_MatteAnything/Matte_Anything/configs/matte_anything.py',
}
def init_vitmatte(model_type, model_path):
    """
    Initialize the vitmatte with model_type in ['vit_s', 'vit_b']
    """
    cfg = LazyConfig.load(os.path.abspath(vitmatte_config[model_type]))
    vitmatte = instantiate(cfg.model)
    device = comfy.model_management.get_torch_device()
    vitmatte.to(device)
    vitmatte.eval()
    DetectionCheckpointer(vitmatte).load(model_path)

    return vitmatte


class LoadVITMatteModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("matte"), ),
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("VIT_MATTE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Matte Anything"

    def load_model(self, model_name, device_mode="auto"):
        model_path = folder_paths.get_full_path("matte", model_name)
        vitmatte = init_vitmatte('vit_b', model_path)
        return (vitmatte,)

class GenerateVITMatte:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "trimap": ("TRIMAP", {}),
                "vit_matte_model": ("VIT_MATTE_MODEL", {}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_matte"

    CATEGORY = "Matte Anything"

    def generate_matte(self, image, trimap, vit_matte_model):
        image_in = pil2cv(tensor2pil(image))
        device = comfy.model_management.get_torch_device()
        input = {
            "image": torch.from_numpy(image_in).permute(2, 0, 1).unsqueeze(0).to(device)/255,
            "trimap": trimap.unsqueeze(0).to(device),
            # "trimap": torch.from_numpy(trimap).unsqueeze(0).unsqueeze(0),
        }
        alpha = vit_matte_model(input)['phas'].flatten(0, 2)
        alpha = alpha.detach().cpu().numpy()
        # Converts alpha matte to RGBA image using the image tensor
        image = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGBA)
        alpha = np.clip(255. * alpha, 0, 255).astype(np.uint8)
        image[:, :, 3] = alpha
        image = Image.fromarray(image)
        image = pil2tensor(image)
        return (image,)

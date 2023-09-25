import os

import folder_paths
from folder_paths import supported_pt_extensions
from .nodes import (
    SAMLoader,
    InitSAMPredictor,
    LoadDINOModel,
    DinoBoxes,
    SAMMaskFromBoxes,
    MaskToTrimap,
    LoadVITMatteModel,
    GenerateVITMatte,
)

# Doesn't work as it doesn't add supported file extensions
# folder_paths.add_model_folder_path("dino", os.path.join(folder_paths.models_dir, "dino"))
folder_paths.folder_names_and_paths["dino"] = ([os.path.join(folder_paths.models_dir, "dino")], supported_pt_extensions)
folder_paths.folder_names_and_paths["matte"] = ([os.path.join(folder_paths.models_dir, "matte")], supported_pt_extensions)

NODE_CLASS_MAPPINGS = {
    "MatteAnything_SAMLoader": SAMLoader,
    "MatteAnything_InitSamPredictor": InitSAMPredictor,
    "MatteAnything_LoadDINO": LoadDINOModel,
    "MatteAnything_DinoBoxes": DinoBoxes,
    "MatteAnything_SAMMaskFromBoxes": SAMMaskFromBoxes,
    "MatteAnything_ToTrimap": MaskToTrimap,
    "MatteAnything_LoadVITMatteModel": LoadVITMatteModel,
    "MatteAnything_GenerateVITMatte": GenerateVITMatte
}

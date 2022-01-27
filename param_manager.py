"""
The parameter manager of this registration framework.
Please do not change the parameters in it if not necessary.
"""
import os

import Parameter
import paths

Parameter._init()  # Initialize the global parameter dict

### General params
Parameter.set_value("register_levels", [3, 4, 5, 6])  # Image levels used for this framework
Parameter.set_value("similarity_metric", "ncc")  # Metric of the image similarity

### Raw data pre-processing params
Parameter.set_value("max_patch_size", 2048)  # Maximum patch size allowed for the local image reading (must be a power of 2)

### Background segmentation params
Parameter.set_value("bs_image_level", 3)  # Image level used for background segmentation step
Parameter.set_value("bs_model_path", os.path.join(paths.bs_model_folder, "segmentation_model_512"))  # Path of the background segmentation model
Parameter.set_value("bs_input_path", paths.grayscale_pair_path)  # Path of the background segmentation outputs
Parameter.set_value("bs_output_path", paths.bs_output_folder)  # Path of the background segmentation outputs

### Image registration params
# Parameter.set_value("ir_input_path", paths.bs_output_folder)
Parameter.set_value("ir_output_path", paths.ir_output_folder)
# Rigid registration params
Parameter.set_value("rr_image_level", 6)  # Image level used for rigid registration step
Parameter.set_value("angle_step", 1)  # Angle step of rotation transformation
# Global affine registration params
Parameter.set_value("ga_image_level", 5)  # Image level used for global affine registration step
Parameter.set_value("ga_model_path", os.path.join(paths.ga_model_folder, "09-13-11_52_55/global_affine_model_7"))  # Path of the global affine registration model
# Local affine registration params
Parameter.set_value("la_image_level", 4)  # Image level used for local affine registration step
Parameter.set_value("la_patch_size", 224)  # Input patch size of the local affine registration network
Parameter.set_value("la_stride", 112)  # Stride of the displacement field patches
Parameter.set_value("la_magnifications_num", 2)  # Number of magnification layers in the local affine registration network
Parameter.set_value("la_model_path", os.path.join(paths.la_model_folder, "09-22-13_47_40/local_affine_model_1"))  # Path of the local affine registration model
# Non-rigid registration params
Parameter.set_value("nr_image_level", 4)  # Image level used for non-rigid registration step
Parameter.set_value("nr_patch_size", 224)  # Input patch size of the non-rigid registration network
Parameter.set_value("nr_stride", 112)  # Stride of the displacement field patches
Parameter.set_value("nr_magnifications_num", 2)  # Number of magnification layers in the non-rigid registration network
Parameter.set_value("nr_model_path", os.path.join(paths.nr_model_folder, "09-24-11_12_26/mm_nonrigid_model_3"))  # Path of the non-rigid registration model

### Image deformation params
Parameter.set_value("id_input_path", paths.rgba_pair_path)  # Path of the image deformation outputs
Parameter.set_value("id_output_path", paths.id_output_folder)  # Path of the image deformation outputs
















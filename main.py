import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import param_manager
import data_preprocess as dp
import background_segmentation as bs
import image_registration as ir
import image_deformation as ide

import utils
import paths
import Parameter



########################################################################################################################
#                                                       Settings                                                       #
########################################################################################################################

# 1. [Raw Data Pre-processing]
# In the pre-processing step, the raw data is first manually cropped to remove most of the background areas.
# Then the whole-slide image is segmented into image patches by local reading.
# The image patches are saved as RGBA (used for generation of final results) and grayscale (used for image registration) respectively.
# Paths: (1) grayscale image pairs: paths.grayscale_pair_path
#        (2) rgba image pairs: paths.rgba_pair_path

pre_processing = True
levels = [2, 4]  # Image levels of the generated registration results (deformed images)
manual_boundary_list = {
    # Each tuple (x, y, width, height) describes the valid area on the corresponding whole-slide image,
    # (x, y) giving the top left pixel in the level 8 reference frame, (width, height) giving the region size.
    'Test1_HE1': (63, 179, 210, 234),
    'Test1_CD8_2': (63, 179, 210, 234),
    'Test1_HE_3': (63, 179, 210, 234),
    'Test2_CD8_1': (109, 186, 210, 234),
    'Test2_HE_2': (109, 186, 210, 234),
    # '1_HE': (36, 187, 310, 391),
    # '1_CD8': (36, 187, 310, 391),
    # '2_HE': (81, 149, 238, 459),
    # '2_CD8': (81, 149, 238, 459),
    # '3_HE': (25, 160, 320, 415),
    # '3_CD8': (25, 160, 320, 415),
    # '4_HE': (20, 215, 318, 335),
    # '4_CD8': (20, 215, 318, 335),
    # '5_HE': (85, 275, 230, 340),
    # '5_CD8': (85, 275, 230, 340),
    # '6_HE': (55, 220, 295, 410),
    # '6_CD8': (55, 220, 295, 410),
    # '7_HE': (58, 270, 301, 263),
    # '7_CD8': (58, 270, 301, 263),
    # '8_HE': (22, 277, 280, 286),
    # '8_CD8': (22, 277, 280, 286),
    # '9_HE': (81, 224, 265, 399),
    # '9_CD8': (81, 224, 265, 399),
    # '10_HE': (45, 261, 316, 354),
    # '10_CD8': (45, 261, 316, 354),
    # '11_HE': (55, 205, 305, 335),
    # '11_CD8': (55, 205, 305, 335),
    # '12_HE': (18, 187, 337, 376),
    # '12_CD8': (18, 187, 337, 376),
    # '13_HE': (25, 155, 335, 460),
    # '13_CD8': (25, 155, 335, 460),
    # '14_HE': (15, 170, 305, 420),
    # '14_CD8': (15, 170, 305, 420),
    # '15_HE': (45, 194, 247, 286),
    # '15_CD8': (45, 194, 247, 286),
    # '16_HE': (36, 179, 301, 384),
    # '16_CD8': (36, 179, 301, 384),
    # '17_HE': (18, 231, 328, 317),
    # '17_CD8': (18, 231, 328, 317),
    # '18_HE': (20, 245, 350, 360),
    # '18_CD8': (20, 245, 350, 360),
}  # Image boundaries for the manual cropping
image_pair_list = [
    # Each tuple (x, y) describes a pair of images to register, where x is the source image and y is the target image.
    ('Test1_CD8_2', 'Test1_HE1'),
    ('Test2_CD8_1', 'Test1_HE1'),
    ('Test1_CD8_2', 'Test1_HE_3'),
    ('Test1_CD8_2', 'Test2_HE_2'),
    ('Test2_CD8_1', 'Test1_HE_3'),
    # ('1_CD8, 1_HE'),
    # ('2_CD8, 2_HE'),
    # ('3_CD8, 3_HE'),
    # ('4_CD8, 4_HE'),
    # ('5_CD8, 5_HE'),
    # ('6_CD8, 6_HE'),
    # ('7_CD8, 7_HE'),
    # ('8_CD8, 8_HE'),
    # ('9_CD8, 9_HE'),
    # ('10_CD8, 10_HE'),
    # ('11_CD8, 11_HE'),
    # ('12_CD8, 12_HE'),
    # ('13_CD8, 13_HE'),
    # ('14_CD8, 14_HE'),
    # ('15_CD8, 15_HE'),
    # ('16_CD8, 16_HE'),
    # ('17_CD8, 17_HE'),
    # ('18_CD8, 18_HE'),
]  # The image pairs to register


# 2. [Background Segmentation]
# In the background segmentation step, the background artifacts are removed from the grayscale image pairs.
# Paths: (1) background segmentation result: paths.bs_output_folder
#        (2) background segmentation preview: paths.preview_folder

background_segmentation = False


# 3. [Image Registration]
# In the image registation step, you are free to choose different combinations of registration steps to perform.
# The calculated displacement field is saved after each step.
# Paths: (1) calculated displacement fields: paths.ir_output_folder
#        (2) image registration preview: paths.preview_folder
#
# This step only performs the calculation of the displacement field and does not perform the image deformation.
# You can try different registration combinations, observe the registration results by the preview figures, and then
# choose the best one to generate the final registered image.

image_registration = True

reg_params = dict()  # Select the registration steps to perform
reg_params["rigid_registration"] = True  # rigid registration
reg_params["global_affine_registration"] = False  # global affine registration
reg_params["local_affine_registration"] = True  # local affine registration (patch-based)
reg_params["nonrigid_registration"] = True  # non-rigid registration (patch-based)

iter_params = dict()
if reg_params["local_affine_registration"]:
    iter_params["la_iteration"] = 1  # Number of times to perform local affine registration
if reg_params["nonrigid_registration"]:
    iter_params["nr_iteration"] = 1  # Number of times to perform non-rigid registration


# 4. [Image Deformation]
# In the image deformation step, the source image is deformed based on the displacement fields calculated by the registration step.
# Paths: (1) deformed image pairs: paths.id_output_folder

image_deformation = True


########################################################################################################################
#                                                Algorithm starts here                                                 #
########################################################################################################################

register_levels = Parameter.get_value("register_levels")  # Image levels used for this registration framework

if pre_processing:  # data pre-processing step
    print("Start data pre-processing...\n")
    for filename in manual_boundary_list.keys():
        manual_boundary = manual_boundary_list[filename]
        dp.pre_processing(filename, levels, manual_crop=True, manual_boundary=manual_boundary, format=['mha', 'jpg'], show=False)
    dp.generate_image_pairs(image_pair_list)

if background_segmentation:  # background segmentation step
    print("Start background segmentation...\n")
    bs_input_path = Parameter.get_value("bs_input_path")
    for id in os.listdir(bs_input_path):
        # load the source and target path
        print(f"Current image pair: {id}")
        img_pair_path = os.path.join(bs_input_path, id)
        source_path, target_path = utils.generate_image_path(img_pair_path)
        # perform the background segmentation
        bs.background_segmentation(id, source_path, target_path, register_levels, save=True, plot=True, device=device)
        print()

if image_registration:  # image registration step
    print("Start image registration...\n")
    if background_segmentation:
        ir_input_path = paths.bs_output_folder
    else:
        ir_input_path = paths.grayscale_pair_path
    for id in os.listdir(ir_input_path):
        print("-" * 100)
        print(f"Current image pair: {id}")
        # load the source and target path
        img_pair_path = os.path.join(ir_input_path, id)
        source_path, target_path = utils.generate_image_path(img_pair_path)
        # register the source and target
        ir.registration(id, source_path, target_path, reg_params, iter_params, plot=True, device=device)
        print()

if image_deformation:  # image deformation step
    print("Start image deformation...\n")
    id_input_path = Parameter.get_value("id_input_path")
    for id in os.listdir(id_input_path):
        print("-" * 100)
        print(f"Current image pair: {id}")
        # load the source and target path
        img_pair_path = os.path.join(id_input_path, id)
        source_path, target_path = utils.generate_image_path(img_pair_path)
        # load all calculated displacement fields
        displacement_fields_folder = os.path.join(Parameter.get_value("ir_output_path"), id)
        displacement_fields_path = utils.generate_displacement_field_path(displacement_fields_folder)
        # displacement_fields_path = [path for path in displacement_fields_path if "rr" in path and "la" in path and "nr" in path] ####
        for displacement_field_path in displacement_fields_path:
            print(f"Current displacement field path: {displacement_field_path}")
            displacement_field = torch.load(displacement_field_path, map_location=device).to(device)
            source_level = int(displacement_field_path[-4])
            case_folder = "_".join((displacement_field_path.split("/")[-1]).split("_")[:-3])
            # deform the source image based on the current displacement field
            for target_level in levels:
                print(f"Generating level-{target_level} image pair...")
                ide.generate_deformed_image(source_path, target_path, displacement_field, (source_level, target_level), case_folder, device=device)
        print()

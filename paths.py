# Original data paths
raw_data_path = "../data/DATA_MRXS"
rgba_data_path = "../data/DATA_RGBA"
grayscale_data_path = "../data/DATA_GRAYSCALE"

# Image pair paths
rgba_pair_path = "../data/rgba_image_pair"
grayscale_pair_path = "../data/grayscale_image_pair"

# Output paths
bs_output_folder = "../output/background_segmentation"
ir_output_folder = "../output/image_registration"
id_output_folder = "../output/image_deformation"

# Preview paths
preview_folder = "../preview"

# Model paths
bs_model_folder = "model/segmentation_network"
ga_model_folder = "model/global_affine_network"
la_model_folder = "model/local_affine_network"
nr_model_folder = "model/nonrigid_network"

# Training/validation data paths for different networks
ga_train_folder = "../data/training/global_affine_registration/train"
ga_val_folder = "../data/training/global_affine_registration/val"
la_train_folder = "../data/training/local_affine_registration/train"
la_val_folder = "../data/training/local_affine_registration/val"
nr_train_folder = "../data/training/nonrigid_registration/train"
nr_val_folder = "../data/training/nonrigid_registration/val"

### logging paths
logging_folder = "logging"


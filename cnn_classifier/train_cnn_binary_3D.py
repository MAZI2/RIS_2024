# %% [markdown]
# # Images

# %% [markdown]
# ## Reading

# %%
import os
import zipfile
import numpy as np
import tensorflow as tf  # for data preprocessing
import nibabel as nib
from scipy import ndimage

import keras
from keras import layers
import gc
from tensorflow.keras import Input, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense

# AFTER:
from tensorflow.keras.callbacks import ModelCheckpoint
import os


# tf.enable_eager_execution()


# %%
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    try:
        scan = nib.load(filepath)
        # Get raw data
        scan = scan.get_fdata()
        return scan
    except Exception as e:
        print(e)
        return None


# %%
def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 250
    desired_width = 350
    desired_height = 350
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    img = img.astype(np.float16)
    return img


def process_scan(scan):
    """Read and resize volume"""
    # Normalize
    volume = normalize(scan)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


scan_paths_main = [
    os.path.join("/d/hpc/projects/training/RIS/data/RIS", x)
    for x in os.listdir("/d/hpc/projects/training/RIS/data/RIS")
]

print("all paths: " + str(len(scan_paths_main)))
scan_paths_main = sorted(scan_paths_main)

scan_paths_main = scan_paths_main
print("used paths: " + str(len(scan_paths_main)))
combined_scans_main = []
pos_indices_main = []

for scan_path_idx in range(0, len(scan_paths_main), 10):
    scan_paths = scan_paths_main[scan_path_idx : scan_path_idx + 10]
    # %%
    # Scan masks to sort between positive and negative samples
    mask_scans_list = [read_nifti_file(path + "/MASK.nii.gz") for path in scan_paths]
    ok_idx_mask = [i for i, x in enumerate(mask_scans_list) if x is not None]
    ct_scans_list = [read_nifti_file(path + "/CT.nii.gz") for path in scan_paths]
    ok_idx_ct = [i for i, x in enumerate(ct_scans_list) if x is not None]
    pet_scans_list = [read_nifti_file(path + "/PET.nii.gz") for path in scan_paths]
    ok_idx_pet = [i for i, x in enumerate(pet_scans_list) if x is not None]

    # Filter for corrupted files
    ok_indices = set(ok_idx_mask) & set(ok_idx_ct) & set(ok_idx_pet)
    print(ok_indices)

    ct_scans_list = [ct_scans_list[i] for i in ok_indices]
    mask_scans_list = [mask_scans_list[i] for i in ok_indices]
    pet_scans_list = [pet_scans_list[i] for i in ok_indices]

    # %%
    # find positive and negative samples: if the mask has any pixel > 0 it is a positive sample
    pos_indices = np.where([np.max(mask) > 0 for mask in mask_scans_list])[0]
    print(pos_indices)
    pos_indices_main.extend(pos_indices)
    del mask_scans_list

    # %% [markdown]
    # ## Processing

    # %%
    # mask_scans_processed = np.array([process_scan(scan) for scan in mask_scans_list])
    ct_scans_processed = np.array([process_scan(scan) for scan in ct_scans_list])
    combined_scans = 0.02 * ct_scans_processed
    del ct_scans_processed
    del ct_scans_list

    pet_scans_processed = np.array([process_scan(scan) for scan in pet_scans_list])
    combined_scans += 0.98 * pet_scans_processed
    combined_scans_main.extend(list(combined_scans))
    del pet_scans_processed
    del pet_scans_list
    gc.collect()

# %%
combined_scans = np.array(combined_scans_main)
del combined_scans_main
print("pos indices main: ", pos_indices_main)
print("length of pos indices main: ", len(pos_indices_main))

# create a list of neg indices
neg_indices_main = [i for i in range(len(combined_scans)) if i not in pos_indices_main]
print("length of neg indices: ", len(neg_indices_main))

# choose only first len(pos_indices_main) neg indices so to balance the dataset
neg_indices_main = neg_indices_main[: len(pos_indices_main)]

# choose 15% of neg and pos indices and save them to a file, remove them from arrays
pos_indices_testing = np.random.choice(
    pos_indices_main, int(0.15 * len(pos_indices_main)), replace=False
)
neg_indices_testing = np.random.choice(
    neg_indices_main, int(0.15 * len(neg_indices_main)), replace=False
)
# remove pos and neg indices testing from main
pos_indices_main = [i for i in pos_indices_main if i not in pos_indices_testing]
neg_indices_main = [i for i in neg_indices_main if i not in neg_indices_testing]

# save testing indices in a txt file
with open("testing_indices.txt", "w") as f:
    f.write("pos indices testing: " + str(pos_indices_testing) + "\n")
    f.write("neg indices testing: " + str(neg_indices_testing) + "\n")

# cut np array combined scans to pos and neg indices
combined_scans = np.array(
    [combined_scans[i] for i in pos_indices_main + neg_indices_main]
)


# %% [markdown]
# ## Preparing

# %%
X = combined_scans[..., np.newaxis]  # Add channel dimension
y = np.array(
    [1 if i in pos_indices else 0 for i in range(len(combined_scans))]
)  # Assuming pos_indices identifies positive samples
del combined_scans

# split data into training, validation and test data, while remembering the indices
# that went into each one


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# # Model

# %%


# Adjusted model definition using an Input layer
model = Sequential(
    [
        Input(shape=(350, 350, 250, 1)),  # Specify input shape here
        Conv3D(
            8, kernel_size=(3, 3, 3), activation="relu"
        ),  # Removed input_shape from here
        MaxPooling3D(pool_size=(2, 2, 2)),
        Conv3D(16, kernel_size=(3, 3, 3), activation="relu"),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)


# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Checkpoint directory
checkpoint_dir = "checkpoints/"

# Create directory if it does not exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Specify the checkpoint file path with .h5 extension for weights
checkpoint_path = "checkpoints/model_{epoch:02d}-{val_loss:.2f}.weights.h5"

# Create a ModelCheckpoint callback to save only the weights
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,  # Save only the weights
    save_best_only=True,  # Save only the best model based on the monitored metric
    monitor="val_loss",  # Monitor the validation loss
    mode="min",  # Objective is to minimize the validation loss
    verbose=1,  # Print out when a model is being saved
)

# %%
# Pass this callback to the `fit` method
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    verbose=1,
    batch_size=2,
    callbacks=[checkpoint],  # Include the callback here
)

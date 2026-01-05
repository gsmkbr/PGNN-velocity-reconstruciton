#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# Set the path relative to the current notebook location
# ------------------------------------------------------------------
# Assuming your data folder is inside the GitHub repository
base_path = Path("github/data")  # relative path from your notebook

# ------------------------------------------------------------------
# Load U and V velocity fields from compressed .npz files
# ------------------------------------------------------------------
u_dns = np.load(base_path / "u_dns.npz")["arr_0"]
v_dns = np.load(base_path / "v_dns.npz")["arr_0"]

# ------------------------------------------------------------------
# Verify the shapes of the loaded arrays
# ------------------------------------------------------------------
print("u_dns shape:", u_dns.shape)
print("v_dns shape:", v_dns.shape)


# In[2]:


print(u_dns.shape)
print(v_dns.shape)

u_velocity = u_dns
v_velocity = v_dns


# In[3]:


import matplotlib.pyplot as plt

# Number of snapshots to display
num_snapshots = 4

# Create figure and axes (2 rows: U and V)
fig, axes = plt.subplots(
    nrows=2,
    ncols=num_snapshots,
    figsize=(18, 6)
)

# Plot U velocity snapshots
for i in range(num_snapshots):
    im_u = axes[0, i].contourf(
        u_velocity[i],
        levels=50,
        cmap="viridis"
    )
    axes[0, i].set_title(f"U {i + 1}", fontsize=14)
    axes[0, i].axis("off")

    cbar_u = fig.colorbar(im_u, ax=axes[0, i])
    cbar_u.set_label("U Velocity", fontsize=14)

# Plot V velocity snapshots
for i in range(num_snapshots):
    im_v = axes[1, i].contourf(
        v_velocity[i],
        levels=50,
        cmap="viridis"
    )
    axes[1, i].set_title(f"V {i + 1}", fontsize=14)
    axes[1, i].axis("off")

    cbar_v = fig.colorbar(im_v, ax=axes[1, i])
    cbar_v.set_label("V Velocity", fontsize=14)

# Adjust layout
plt.tight_layout()
plt.show()


# In[4]:


# This preprocessing step extracts a centered region downstream of the cylinder
# from all velocity snapshots. The selected subdomain is used to train and evaluate
# the neural network, focusing on the most physically relevant flow features
# in the wake region behind the cylinder.

# Select the first snapshot
snapshot = u_velocity[0]

# Get snapshot dimensions
num_rows, num_cols = snapshot.shape

# Compute center indices
center_row = num_rows // 2
center_col = num_cols // 2

print("Center of snapshot 1:")
print(f"Row index: {center_row}, Column index: {center_col}")

# Define extraction window size (half-size on each side)
rows_to_extract = 64
cols_to_extract = 64

# Get full data dimensions
num_snapshots, num_rows, num_cols = u_velocity.shape

# Compute slicing indices
row_start = center_row - rows_to_extract
row_end   = center_row + rows_to_extract
col_start = center_col - cols_to_extract
col_end   = center_col + cols_to_extract

# Extract centered regions from all snapshots
extracted_u_regions = u_velocity[:, row_start:row_end, col_start:col_end]
extracted_v_regions = v_velocity[:, row_start:row_end, col_start:col_end]

# Check extracted shapes
print(f"Extracted U regions shape: {extracted_u_regions.shape}")
print(f"Extracted V regions shape: {extracted_v_regions.shape}")


# In[5]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------------------------------------------
# Global plot settings
# ------------------------------------------------------------------
plt.rcParams.update({"font.size": 18})

# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------
snapshot_indices = [20, 60]
rows_to_extract = 64
cols_to_extract = 64
row_labels = ["a)", "b)", "c)", "d)"]

# ------------------------------------------------------------------
# Domain and extraction indices
# ------------------------------------------------------------------
num_snapshots, num_rows, num_cols = u_velocity.shape

center_row = num_rows // 2
center_col = num_cols // 2

row_start = center_row - rows_to_extract
row_end   = center_row + rows_to_extract
col_start = center_col - cols_to_extract
col_end   = center_col + cols_to_extract

# ------------------------------------------------------------------
# Figure setup
# ------------------------------------------------------------------
fig_width  = 4 * len(snapshot_indices)
fig_height = 4 * 1.5

fig, axes = plt.subplots(
    nrows=4,
    ncols=len(snapshot_indices),
    figsize=(fig_width, fig_height)
)

# ------------------------------------------------------------------
# Plot snapshots
# ------------------------------------------------------------------
for i, snap_idx in enumerate(snapshot_indices):

    # Extract subdomains
    u_ext = u_velocity[snap_idx, row_start:row_end, col_start:col_end]
    v_ext = v_velocity[snap_idx, row_start:row_end, col_start:col_end]

    # ---- Row 1: Full U velocity ----
    im_u_full = axes[0, i].imshow(u_velocity[snap_idx], cmap="viridis", aspect="auto")
    axes[0, i].axis("off")

    rect_u = patches.Rectangle(
        (col_start, row_start),
        col_end - col_start,
        row_end - row_start,
        linewidth=4,
        edgecolor="red",
        linestyle="dotted",
        facecolor="none"
    )
    axes[0, i].add_patch(rect_u)
    fig.colorbar(im_u_full, ax=axes[0, i], fraction=0.046, pad=0.04)

    # ---- Row 2: Extracted U velocity ----
    im_u_ext = axes[1, i].imshow(u_ext, cmap="viridis", aspect="auto")
    axes[1, i].axis("off")
    fig.colorbar(im_u_ext, ax=axes[1, i], fraction=0.046, pad=0.04)

    # ---- Row 3: Full V velocity ----
    im_v_full = axes[2, i].imshow(v_velocity[snap_idx], cmap="viridis", aspect="auto")
    axes[2, i].axis("off")

    rect_v = patches.Rectangle(
        (col_start, row_start),
        col_end - col_start,
        row_end - row_start,
        linewidth=4,
        edgecolor="red",
        linestyle="dotted",
        facecolor="none"
    )
    axes[2, i].add_patch(rect_v)
    fig.colorbar(im_v_full, ax=axes[2, i], fraction=0.046, pad=0.04)

    # ---- Row 4: Extracted V velocity ----
    im_v_ext = axes[3, i].imshow(v_ext, cmap="viridis", aspect="auto")
    axes[3, i].axis("off")
    fig.colorbar(im_v_ext, ax=axes[3, i], fraction=0.046, pad=0.04)

# ------------------------------------------------------------------
# Row labels (a), b), c), d))
# ------------------------------------------------------------------
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].text(
        -0.15, 0.5,
        label,
        transform=axes[row_idx, 0].transAxes,
        fontsize=18,
        fontweight="bold",
        va="center",
        ha="right"
    )

# ------------------------------------------------------------------
# Final layout
# ------------------------------------------------------------------
plt.tight_layout()
plt.show()


# In[6]:


vector3d_U = extracted_u_regions
vector3d_V = extracted_v_regions


# In[7]:


# ------------------------------------------------------------------
# Add Gaussian noise to the velocity fields for model evaluation
# This simulates different noise levels to test the model's robustness
# ------------------------------------------------------------------
import numpy as np

def add_gaussian_noise(data, mean=0.0, std=0.0):
    """
    Add Gaussian noise to a given data array.

    Parameters:
        data (np.ndarray): Input data array
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise

    Returns:
        np.ndarray: Noisy data
    """
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

# Add noise (currently std=0.0; can be increased for testing)
u_velocity_noisy = add_gaussian_noise(vector3d_U, mean=0.0, std=0.0)
v_velocity_noisy = add_gaussian_noise(vector3d_V, mean=0.0, std=0.0)

# ------------------------------------------------------------------
# Check range of noisy data
# ------------------------------------------------------------------
print(f"Min and Max of noisy U: {u_velocity_noisy.min()}, {u_velocity_noisy.max()}")
print(f"Min and Max of noisy V: {v_velocity_noisy.min()}, {v_velocity_noisy.max()}")

# ------------------------------------------------------------------
# Quick check of specific elements
# ------------------------------------------------------------------
print("Sample values from noisy U:")
print(u_velocity_noisy[0, 10, 10:20])

print("Corresponding values from vector3d_U:")
print(vector3d_U[0, 10, 10:20])


# In[9]:


# ------------------------------------------------------------------
# Compute the mean percentage of noise added to the velocity field
# ------------------------------------------------------------------
import numpy as np

# Original and noisy velocity fields (assumed to be numpy arrays)
original_values = vector3d_U
noisy_values    = u_velocity_noisy

# Avoid division by zero by adding a small epsilon
epsilon = 1e-12
percent_noise = (np.abs(noisy_values - original_values) / (original_values + epsilon)) * 100

# Compute mean percentage noise across all elements
mean_percent_noise = np.mean(percent_noise)

# Display result
print("Mean percentage noise added:", mean_percent_noise)


# In[10]:


vector3d_U = u_velocity_noisy
vector3d_V = v_velocity_noisy


# In[11]:


import matplotlib.pyplot as plt

# Number of snapshots to display
num_snapshots = 4

# Set up a 2x5 grid for the subplots
fig, axes = plt.subplots(2, num_snapshots, figsize=(18, 6))  # 2 rows, num_snapshots columns

# Plot U velocity (Original)
for i in range(num_snapshots):
    im_u_orig = axes[0, i].contourf(vector3d_U[i], cmap='viridis', levels=50)
    axes[0, i].set_title(f'U {i+1}', fontsize=14)
    axes[0, i].axis('off')
    cbar_u_orig = fig.colorbar(im_u_orig, ax=axes[0, i], orientation='vertical')
    cbar_u_orig.set_label('U Velocity', fontsize=14)

# Plot V velocity (Original)
for i in range(num_snapshots):
    im_v_orig = axes[1, i].contourf(vector3d_V[i], cmap='viridis', levels=50)
    axes[1, i].set_title(f'V {i+1}', fontsize=14)
    axes[1, i].axis('off')
    cbar_v_orig = fig.colorbar(im_v_orig, ax=axes[1, i], orientation='vertical')
    cbar_v_orig.set_label('V Velocity', fontsize=14)

# Adjust layout to make sure everything fits
plt.tight_layout()
plt.show()


# In[12]:


# ------------------------------------------------------------------
# Introduce artificial gaps (missing regions) into the velocity fields
# This step simulates incomplete data in the flow fields.
# The neural network will be trained and evaluated based on these gaps,
# learning to reconstruct or predict the missing regions from the surrounding data.
# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------------
# Function to generate a single artificial mask image
# ------------------------------------------------------------------
def generate_image(
    image_size=(128, 128), 
    min_rectangles=15, 
    max_rectangles=45, 
    target_zero_area_range=(0.1, 0.9)
):
    """
    Generates a 2D image with random black rectangles (zeros) on a white background (ones).
    Ensures that the total zero-area falls within a specified fraction of the image.

    Parameters:
        image_size (tuple): Size of the image (rows, cols)
        min_rectangles (int): Minimum number of rectangles to draw
        max_rectangles (int): Maximum number of rectangles to draw
        target_zero_area_range (tuple): Min and max fraction of zero-area in the image

    Returns:
        np.ndarray: Generated image with zeros (black) and ones (white)
    """
    total_cells = image_size[0] * image_size[1]
    min_zero_area = int(total_cells * target_zero_area_range[0])
    max_zero_area = int(total_cells * target_zero_area_range[1])

    # Random number of rectangles for this image
    num_rectangles = random.randint(min_rectangles, max_rectangles)

    max_attempts = 100  # Prevent infinite loops
    attempt = 0

    while attempt < max_attempts:
        # Start with a white background
        image = np.ones(image_size)
        
        # Draw random rectangles
        for _ in range(num_rectangles):
            x_start = random.randint(0, image_size[0] - 1)
            y_start = random.randint(0, image_size[1] - 1)
            rect_width = random.randint(5, 20)
            rect_height = random.randint(5, 20)

            # Ensure rectangle stays inside image
            x_end = min(x_start + rect_width, image_size[0])
            y_end = min(y_start + rect_height, image_size[1])

            # Draw black rectangle (zero pixels)
            image[x_start:x_end, y_start:y_end] = 0

        # Check zero-area percentage
        current_zero_area = np.sum(image == 0)
        if min_zero_area <= current_zero_area <= max_zero_area:
            break  # Accept this image if within target range

        attempt += 1

    return image

# ------------------------------------------------------------------
# Set random seed for reproducibility
# ------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ------------------------------------------------------------------
# Generate multiple fixed mask images
# ------------------------------------------------------------------
num_images = 151
masks = [generate_image() for _ in range(num_images)]

# ------------------------------------------------------------------
# Example: Plot a few generated masks
# ------------------------------------------------------------------
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axs):
    ax.imshow(masks[i], cmap='viridis', origin='lower')
    ax.axis('off')
plt.show()

# ------------------------------------------------------------------
# Optional: Check zero-value percentages for the first few images
# ------------------------------------------------------------------
for i, image in enumerate(masks[:5]):
    zero_area_percentage = np.sum(image == 0) / (128 * 128)
    print(f"Image {i + 1} zero area percentage: {zero_area_percentage * 100:.2f}%")

# ------------------------------------------------------------------
# Save masks to file
# ------------------------------------------------------------------
np.save("fixed_masks.npy", np.array(masks))


# In[14]:


# ------------------------------------------------------------------
# Apply artificial masks to the velocity fields and replace masked areas with NaN
# This simulates missing regions in the flow data for training/evaluation
# ------------------------------------------------------------------

# Masks generated previously (0 = gap, 1 = valid)
mask_nan = masks  # shape: (num_snapshots, 128, 128)

# Apply masks to U and V velocity components
u_component = vector3d_U * mask_nan
v_component = vector3d_V * mask_nan

# Optional: check shapes
print("Shape of u_component:", u_component.shape)
print("Shape of v_component:", v_component.shape)

# Replace zeros in masked regions with NaN
u_nan = np.where(u_component == 0, np.nan, u_component)
v_nan = np.where(v_component == 0, np.nan, v_component)

# ------------------------------------------------------------------
# Calculate percentage of NaN values
# ------------------------------------------------------------------
u_nan_percent = np.isnan(u_nan).sum() / u_nan.size * 100
v_nan_percent = np.isnan(v_nan).sum() / v_nan.size * 100

print(f"NaN percentage in u_component: {u_nan_percent:.2f}%")
print(f"NaN percentage in v_component: {v_nan_percent:.2f}%")


# In[15]:


import tensorflow as tf

# ------------------------------------------------------------------
# Function: top_hat
# Computes local mean in sliding windows while masking regions with
# too many zeros (replacing them with NaN)
# ------------------------------------------------------------------
def top_hat(snapshot, kernel_size):
    """
    Apply top-hat (mean filtering) with masking of windows that have
    more than 50% zeros. Returns the mean values with NaNs in masked regions.

    Parameters:
        snapshot (tf.Tensor): Shape (num_snapshots, height, width)
        kernel_size (tuple): (window_height, window_width)

    Returns:
        tf.Tensor: Shape (num_snapshots, new_height, new_width) with NaNs in masked windows
    """
    window_height, window_width = kernel_size
    snapshot_exp = snapshot[..., tf.newaxis]  # Add channel dimension

    # Extract sliding windows
    windows = tf.image.extract_patches(
        snapshot_exp,
        sizes=[1, window_height, window_width, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    windows = tf.reshape(windows, (-1, window_height * window_width))

    # Count zeros in each window
    zero_counts = tf.reduce_sum(tf.cast(tf.equal(windows, 0), tf.float32), axis=1)

    # Reshape to 2D spatial grid
    output_shape = (snapshot.shape[1] - window_height + 1,
                    snapshot.shape[2] - window_width + 1)
    zero_counts = tf.reshape(zero_counts, (-1, *output_shape))

    # Compute mean window via convolution
    mean_kernel = tf.ones((window_height, window_width), dtype=tf.float32) / (window_height * window_width)
    mean_kernel = mean_kernel[:, :, tf.newaxis, tf.newaxis]  # Shape for conv2d
    mean_window = tf.nn.conv2d(snapshot_exp, mean_kernel, strides=[1, 1, 1, 1], padding='VALID')
    
    # Mask windows with more than 50% zeros
    mask = zero_counts > (window_height * window_width) / 2
    mask_broadcasted = mask[..., tf.newaxis]
    mean_window_with_nan = tf.where(mask_broadcasted, tf.constant(float('nan'), dtype=tf.float32), mean_window)

    # Reshape back to (num_snapshots, new_height, new_width)
    num_snapshots = tf.shape(snapshot)[0]
    height = tf.shape(snapshot)[1] - window_height + 1
    width = tf.shape(snapshot)[2] - window_width + 1
    mean_window_with_nan = tf.reshape(mean_window_with_nan, (num_snapshots, height, width))

    return mean_window_with_nan

# ------------------------------------------------------------------
# Function: barycentric_snapshot
# Computes the anisotropic stress tensor components for U and V snapshots
# ------------------------------------------------------------------
def barycentric_snapshot(input_snapshots_u, input_snapshots_v, custom_kernel_size):
    """
    Calculate the anisotropic stress tensor a_ij for U and V snapshots
    using local mean filtering with top_hat.

    Parameters:
        input_snapshots_u (tf.Tensor): U velocity snapshots, shape (N, H, W)
        input_snapshots_v (tf.Tensor): V velocity snapshots, shape (N, H, W)
        custom_kernel_size (tuple): Window size for top_hat filtering

    Returns:
        tf.Tensor: Stress tensor, shape (N, H', W', 2, 2)
    """
    num_snapshots = tf.shape(input_snapshots_u)[0]

    # Local mean with masking
    mean_u = top_hat(input_snapshots_u, custom_kernel_size)
    mean_v = top_hat(input_snapshots_v, custom_kernel_size)

    # Compute second-order moments
    u_squared = input_snapshots_u ** 2
    mean_u_squared = top_hat(u_squared, custom_kernel_size)

    v_squared = input_snapshots_v ** 2
    mean_v_squared = top_hat(v_squared, custom_kernel_size)

    u_times_v = input_snapshots_u * input_snapshots_v
    mean_u_times_v = top_hat(u_times_v, custom_kernel_size)

    # Stress tensor components
    tau_xx = mean_u_squared - mean_u * mean_u
    tau_xy = mean_u_times_v - mean_u * mean_v
    tau_yx = tau_xy
    tau_yy = mean_v_squared - mean_v * mean_v

    # Mean stress (trace / 2)
    tau_ii = (tau_xx + tau_yy) / 2

    # Delta tensor
    delta = 0.5 * tf.ones_like(tau_ii, dtype=tf.float32)

    # Anisotropic stress tensor components
    a_xx = tau_xx - delta * tau_ii
    a_xy = tau_xy - delta * tau_ii
    a_yx = tau_yx - delta * tau_ii
    a_yy = tau_yy - delta * tau_ii

    # Stack into final tensor (num_snapshots, H', W', 2, 2)
    a_tensor = tf.stack([a_xx, a_xy, a_yx, a_yy], axis=-1)
    a_tensor = tf.reshape(a_tensor, (num_snapshots, tf.shape(a_xx)[1], tf.shape(a_xx)[2], 2, 2))

    return a_tensor


# In[16]:


import tensorflow as tf
u_velocity_tensor = vector3d_U
v_velocity_tensor = vector3d_V
#custom kernel size
custom_kernel_size = (5, 5)
a_tensor = barycentric_snapshot(u_velocity_tensor, v_velocity_tensor, custom_kernel_size)


# In[17]:


from scipy.interpolate import griddata

# ------------------------------------------------------------------
# Grid coordinates for interpolation
# ------------------------------------------------------------------
grid_x, grid_y = np.meshgrid(np.arange(128), np.arange(128))

# ------------------------------------------------------------------
# Function: interpolate_component
# Fill missing values (NaN) in each snapshot using cubic interpolation
# ------------------------------------------------------------------
def interpolate_component(component):
    """
    Interpolates NaN values in 3D velocity component array (snapshots x height x width)
    using cubic grid interpolation.

    Parameters:
        component (np.ndarray): Shape (num_snapshots, height, width) with NaNs for missing values

    Returns:
        np.ndarray: Component array with NaNs replaced by interpolated values
    """
    num_snapshots, height, width = component.shape

    for snapshot_idx in range(num_snapshots):
        snapshot_data = component[snapshot_idx]

        # Mask of NaN values
        nan_mask = np.isnan(snapshot_data)

        if np.any(nan_mask):  # Only interpolate if there are NaNs
            # Coordinates of known (non-NaN) points
            points = np.column_stack([grid_x[~nan_mask], grid_y[~nan_mask]])

            # Values at known points
            values = snapshot_data[~nan_mask]

            # Interpolate missing values on the grid
            interpolated_values = griddata(points, values, (grid_x, grid_y), method='cubic')

            # Replace NaNs with interpolated values
            snapshot_data[nan_mask] = interpolated_values[nan_mask]

        # Update the snapshot in the component array
        component[snapshot_idx] = snapshot_data

    return component

# ------------------------------------------------------------------
# Apply cubic interpolation to U and V components
# This provides an initial guess for the missing regions (NaNs)
# before training the model.
# ------------------------------------------------------------------
u_component = interpolate_component(u_nan)
v_component = interpolate_component(v_nan)


# In[18]:


# ------------------------------------------------------------------
# Grid coordinates for interpolation
# ------------------------------------------------------------------
grid_x, grid_y = np.meshgrid(np.arange(128), np.arange(128))

# ------------------------------------------------------------------
# Function: interpolate_component
# Fill missing values (NaN) in each snapshot using nearest-neighbor interpolation
# ------------------------------------------------------------------
def interpolate_component(component):
    """
    Interpolates NaN values in a 3D velocity component array (snapshots x height x width)
    using nearest-neighbor grid interpolation as an initial guess.

    Parameters:
        component (np.ndarray): Shape (num_snapshots, height, width) with NaNs

    Returns:
        np.ndarray: Component array with NaNs replaced by interpolated values
    """
    num_snapshots, height, width = component.shape

    for snapshot_idx in range(num_snapshots):
        snapshot_data = component[snapshot_idx]

        # Mask of NaN values
        nan_mask = np.isnan(snapshot_data)

        if np.any(nan_mask):  # Only interpolate if there are missing values
            # Coordinates of known points
            points = np.column_stack([grid_x[~nan_mask], grid_y[~nan_mask]])

            # Values at known points
            values = snapshot_data[~nan_mask]

            # Interpolate missing values using nearest neighbor
            interpolated_values = griddata(points, values, (grid_x, grid_y), method='nearest')

            # Replace NaNs with interpolated values
            snapshot_data[nan_mask] = interpolated_values[nan_mask]

        # Update the snapshot in the component array
        component[snapshot_idx] = snapshot_data

    return component

# ------------------------------------------------------------------
# Apply interpolation to U and V components
# Provides an initial guess for missing regions before training the model
# ------------------------------------------------------------------
u_component = interpolate_component(u_component)
v_component = interpolate_component(v_component)


# In[20]:


# ------------------------------------------------------------------
# Set a fixed random seed for reproducibility
# ------------------------------------------------------------------
np.random.seed(42)

# ------------------------------------------------------------------
# Prepare data components
# ------------------------------------------------------------------
# Downsampled velocity regions
vector3d_U_t = vector3d_U
vector3d_V_t = vector3d_V

# Masks as NumPy array
masks = np.array(masks)

# Stack U and V components along the last axis for model input (x) and output (y)
x = np.stack((u_component[:150], v_component[:150]), axis=-1)  # Input with gaps interpolated
y = np.stack((vector3d_U_t[:150], vector3d_V_t[:150]), axis=-1)  # Ground truth / target

print("x shape:", x.shape)  # Expected: (150, height, width, 2)
print("y shape:", y.shape)  # Expected: (150, height, width, 2)

# ------------------------------------------------------------------
# Split data into training and testing sets
# ------------------------------------------------------------------
total_samples = x.shape[0]

# Randomly permute indices for splitting
indices = np.random.permutation(total_samples)

train_indices = indices[:140]  # First 140 samples for training
test_indices = indices[140:150]  # Last 10 samples for testing

# Split input (x), target (y), and masks
x_train = x[train_indices]
y_train = y[train_indices]
masks_a_train = masks[train_indices]

x_test = x[test_indices]
y_test = y[test_indices]
masks_a_test = masks[test_indices]

# Optionally keep the full U and V test data for evaluation
vector3d_U_test = vector3d_U[test_indices]
vector3d_V_test = vector3d_V[test_indices]

# ------------------------------------------------------------------
# Print shapes for verification
# ------------------------------------------------------------------
print("Shapes after splitting:")
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("masks_a_train shape:", masks_a_train.shape)
print("masks_a_test shape:", masks_a_test.shape)


# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

# ------------------------------------------------------------------
# Custom loss function incorporating physical stress components
# ------------------------------------------------------------------
def custom_mse_loss(y_true, y_pred):
    """
    Custom MSE loss function for the autoencoder, incorporating:
    - Standard MSE on U and V components (masked where missing)
    - Physical anisotropic stress tensor a_ij (from barycentric_snapshot)

    Parameters:
        y_true: Ground truth velocity fields, shape (batch, H, W, 2)
        y_pred: Predicted velocity fields, shape (batch, H, W, 2)

    Returns:
        Total loss (tf.Tensor scalar)
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Split channels into U and V components
    y_true_u, y_true_v = tf.split(y_true, num_or_size_splits=2, axis=-1)
    y_pred_u, y_pred_v = tf.split(y_pred, num_or_size_splits=2, axis=-1)

    # Masks to ignore missing values (gaps)
    mask_u_a = tf.cast(y_true_u != 0, tf.float32)
    mask_v_a = tf.cast(y_true_v != 0, tf.float32)

    # Standard MSE for U and V components
    mse_u = tf.reduce_mean(tf.square(y_true_u - y_pred_u) * mask_u_a)
    mse_v = tf.reduce_mean(tf.square(y_true_v - y_pred_v) * mask_v_a)

    # Define kernel size for top_hat/barycentric stress calculation
    custom_kernel_size = (5, 5)

    # Reshape to (batch, H, W) for barycentric_snapshot
    u_tensor = tf.reshape(y_true_u, [-1, 128, 128])
    v_tensor = tf.reshape(y_true_v, [-1, 128, 128])
    u_pred_tensor = tf.reshape(y_pred_u * mask_u_a, [-1, 128, 128])
    v_pred_tensor = tf.reshape(y_pred_v * mask_v_a, [-1, 128, 128])

    # Compute physical anisotropic stress tensor
    a_true = barycentric_snapshot(u_tensor, v_tensor, custom_kernel_size)
    a_pred = barycentric_snapshot(u_pred_tensor, v_pred_tensor, custom_kernel_size)

    # Replace NaNs with zeros to allow loss computation
    a_true = tf.where(tf.math.is_nan(a_true), tf.zeros_like(a_true), a_true)
    a_pred = tf.where(tf.math.is_nan(a_pred), tf.zeros_like(a_pred), a_pred)

    # Compute squared differences for each stress tensor component
    loss_xx = tf.reduce_mean(tf.square(a_true[:, :, :, 0, 0] - a_pred[:, :, :, 0, 0]))
    loss_xy = tf.reduce_mean(tf.square(a_true[:, :, :, 0, 1] - a_pred[:, :, :, 0, 1]))
    loss_yx = tf.reduce_mean(tf.square(a_true[:, :, :, 1, 0] - a_pred[:, :, :, 1, 0]))
    loss_yy = tf.reduce_mean(tf.square(a_true[:, :, :, 1, 1] - a_pred[:, :, :, 1, 1]))

    # Weighted combination of stress tensor losses (can tune coefficients)
    a_loss = 3.24 * loss_xx + 4 * (loss_xy + loss_yx) + 5.22 * loss_yy

    # Total loss: MSE + (optionally) stress tensor loss
    total_loss = mse_u + mse_v + 0 * a_loss  # Currently stress loss multiplied by 0

    return total_loss

# ------------------------------------------------------------------
# Custom callback to track and store loss components during training
# ------------------------------------------------------------------
class LossHistory(Callback):
    """
    Custom callback to monitor training:
    - MSE for U and V
    - Physical anisotropic stress tensor loss
    - Total loss
    """
    def __init__(self, x_train, y_train):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train

    def on_train_begin(self, logs=None):
        self.mse_u = []
        self.mse_v = []
        self.a_loss = []
        self.total_loss = []

    def on_epoch_end(self, epoch, logs=None):
        # Predict on training data
        y_pred = self.model.predict(self.x_train)
        y_true = tf.cast(self.y_train, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Split channels
        y_true_u, y_true_v = tf.split(y_true, num_or_size_splits=2, axis=-1)
        y_pred_u, y_pred_v = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        # Mask missing values
        mask_u_a = tf.cast(y_true_u != 0, tf.float32)
        mask_v_a = tf.cast(y_true_v != 0, tf.float32)

        # Compute MSE components
        mse_u = tf.reduce_mean(tf.square(y_true_u - y_pred_u) * mask_u_a).numpy()
        mse_v = tf.reduce_mean(tf.square(y_true_v - y_pred_v) * mask_v_a).numpy()

        # Compute physical stress tensor components
        u_tensor = tf.reshape(y_true_u, [-1, 128, 128])
        v_tensor = tf.reshape(y_true_v, [-1, 128, 128])
        u_pred_tensor = tf.reshape(y_pred_u * mask_u_a, [-1, 128, 128])
        v_pred_tensor = tf.reshape(y_pred_v * mask_v_a, [-1, 128, 128])

        a_true = barycentric_snapshot(u_tensor, v_tensor, (5, 5))
        a_pred = barycentric_snapshot(u_pred_tensor, v_pred_tensor, (5, 5))

        a_true = tf.where(tf.math.is_nan(a_true), tf.zeros_like(a_true), a_true)
        a_pred = tf.where(tf.math.is_nan(a_pred), tf.zeros_like(a_pred), a_pred)

        # Stress tensor component losses
        loss_xx = tf.reduce_mean(tf.square(a_true[:, :, :, 0, 0] - a_pred[:, :, :, 0, 0])).numpy()
        loss_xy = tf.reduce_mean(tf.square(a_true[:, :, :, 0, 1] - a_pred[:, :, :, 0, 1])).numpy()
        loss_yx = tf.reduce_mean(tf.square(a_true[:, :, :, 1, 0] - a_pred[:, :, :, 1, 0])).numpy()
        loss_yy = tf.reduce_mean(tf.square(a_true[:, :, :, 1, 1] - a_pred[:, :, :, 1, 1])).numpy()

        a_loss = loss_xx + loss_xy + loss_yx + loss_yy

        # Store losses
        self.mse_u.append(mse_u)
        self.mse_v.append(mse_v)
        self.a_loss.append(a_loss)
        self.total_loss.append(logs['loss'])

    def plot_losses(self):
        """Plot all loss components over epochs"""
        epochs = range(1, len(self.total_loss) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.mse_u, 'r-', label='MSE_u')
        plt.plot(epochs, self.mse_v, 'b-', label='MSE_v')
        plt.plot(epochs, self.a_loss, 'g-', label='A_loss')
        plt.plot(epochs, self.total_loss, 'k-', label='Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Components During Training')
        plt.legend()
        plt.show()

# ------------------------------------------------------------------
# Define the autoencoder model
# ------------------------------------------------------------------
def build_autoencoder(input_shape):
    """
    Simple convolutional autoencoder with:
    - Encoder: 2 Conv2D layers with LeakyReLU + MaxPooling
    - Decoder: 2 Conv2DTranspose layers with LeakyReLU + UpSampling
    Loss function includes custom_mse_loss
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(16, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), padding='same')(encoded)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(16, (3, 3), padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2DTranspose(input_shape[2], (3, 3), padding='same')(x)
    decoded = layers.LeakyReLU(negative_slope=0.2)(decoded)

    autoencoder = models.Model(inputs=inputs, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss=custom_mse_loss)

    return autoencoder


# In[22]:


# ------------------------------------------------------------------
# Build the autoencoder model
# ------------------------------------------------------------------
input_shape = (128, 128, 2)  # Input: 128x128 grid with 2 channels (U and V velocities)
autoencoder = build_autoencoder(input_shape)

# Display a summary of the model architecture
autoencoder.summary()


# In[23]:


# Create custom loss history callback
loss_history = LossHistory(x_train, y_train)

# Train the model
history = autoencoder.fit(x_train, y_train,
                          epochs=5,
                          batch_size=10,
                          callbacks=[loss_history])

# Plot the tracked loss components
loss_history.plot_losses()


# In[24]:


# ------------------------------------------------------------------
# Predict on test data using the trained autoencoder
# ------------------------------------------------------------------
x_test_encoded = autoencoder.predict(x_test)

# Store the reconstructed velocity fields (U and V) for evaluation
reconstructed_data = x_test_encoded

# reconstructed_data.shape => (num_test_samples, 128, 128, 2)
print("Shape of reconstructed data:", reconstructed_data.shape)


# In[ ]:





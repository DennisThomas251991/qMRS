"""
Functions.py

This file contains a collection of utility functions for quantitative Magnetic Resonance Spectroscopy (qMRS) data processing.
It includes routines for:
- Overlaying spectroscopy VOI on MRI images and saving as TIFF
- Creating LCModel control files
- Generating MRS voxel masks from Siemens RDA files
- Reslicing MRI images to spectroscopy space
- Reading Siemens .rda MRS files and extracting header/data
- Saving MRS data in RAW text format
- Running LCModel from Python
- Combining LCModel output tables into a CSV
- Modifying and saving CSVs for further analysis
- Processing RAW files for repeated data blocks

These functions support workflows for MRSI/MRS data analysis, visualization, and integration with LCModel and other tools.
"""

import os
import re
import csv
import math
import sys
import glob
import subprocess
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FixedLocator
from scipy.linalg import inv
from pathlib import Path
from nilearn.image import resample_img
from nilearn.image import load_img
def save_spec_VOI_on_MRI_image(mri_file, spec_header, save_path, alpha=0.5):
    """
    Overlay the spectroscopy VOI (Volume Of Interest) on an MRI image and save as TIFF.
    Also saves:
      - an image with only the original (red) rectangle (_original_voxels)
      - an image with only the expanded (green) rectangle (_expanded_voxels)
      - an image with both rectangles (default name)
    Args:
        mri_file (str): Path to the MRI NIfTI file (already resliced).
        spec_header (str): Path to the spectroscopy header text file.
        alpha (float): Transparency for the grid overlay.
    Returns:
        dict: Dictionary with calculated voxel indices for the VOI.
    """
    # Load the MRI image (resliced)
    resliced = nib.load(mri_file)
    header_text_path = spec_header

    # Read header text and extract parameters
    with open(header_text_path, "r") as file_header:
        lines = file_header.readlines()

    dict_params = {}
    # List of header keys to extract
    keys = [
        'SliceThickness:', 'PositionVector[0]:', 'PositionVector[1]:',
        'PositionVector[2]:', 'RowVector[0]:', 'RowVector[1]:',
        'RowVector[2]:', 'ColumnVector[0]:', 'ColumnVector[1]:',
        'ColumnVector[2]:', 'VOINormalSag:', 'VOINormalCor:',
        'VOINormalTra:', 'VOIPositionSag:', 'VOIPositionCor:',
        'VOIPositionTra:', 'VOIPhaseFOV', 'VOIReadoutFOV',
        'FoVHeight:', 'FoVWidth:', 'NumberOfRows:', 'NumberOfColumns:',
        'PixelSpacingRow:', 'PixelSpacingCol:'
    ]
    # Extract values for each key from the header lines
    for word in keys:
        for line in lines:
            if word in line:
                index = line.find(word)
                dict_params[word] = line[index + len(word) + 1:].strip()
                break

    # Get affine and its inverse for coordinate transforms
    affine_resliced = resliced.affine
    inv_affine_resliced = inv(affine_resliced)

    # Extract VOI center position from header
    mid_position = [
        dict_params['VOIPositionSag:'],
        dict_params['VOIPositionCor:'],
        dict_params['VOIPositionTra:']
    ]

    # Transform VOI center from world to voxel coordinates
    slice_position = inv_affine_resliced @ [-np.float64(mid_position[0]),
                                            -np.float64(mid_position[1]),
                                             np.float64(mid_position[2]),
                                             1]

    # Extract VOI FOV (field of view) dimensions
    phase_fov = np.float64(dict_params['VOIPhaseFOV'])
    readout_fov = np.float64(dict_params['VOIReadoutFOV'])
    x0 = slice_position[0] - phase_fov / 2
    y0 = slice_position[1] - readout_fov / 2
    x_width = phase_fov
    y_width = readout_fov

    # Get pixel spacing for conversion between mm and voxel indices
    pixel_spacing_row = float(dict_params['PixelSpacingRow:'])
    pixel_spacing_col = float(dict_params['PixelSpacingCol:'])

    # Convert mm to voxel indices with proper rounding
    xx = int(math.floor(x0 / pixel_spacing_row))
    xy = int(math.ceil((x0 + x_width) / pixel_spacing_row))
    yx = int(math.floor(y0 / pixel_spacing_col)) 
    yy = int(math.ceil((y0 + y_width) / pixel_spacing_col))

    # Set up grid lines based on number of rows/columns
    num_rows = int(np.float64(dict_params['NumberOfRows:']))
    num_cols = int(np.float64(dict_params['NumberOfColumns:']))

    slice_index = int(round(slice_position[2]))
    img_slice = resliced.get_fdata()[:, :, slice_index]

    # --- 1. Save figure with only the red rectangle (original_voxels) ---
    fig1, ax1 = plt.subplots()
    ax1.imshow(img_slice, cmap='gray', origin='lower')
    rect = patches.Rectangle((x0, y0), x_width, y_width, linewidth=1, edgecolor='r', facecolor='none')
    ax1.add_patch(rect)
    ax1.xaxis.set_major_locator(FixedLocator(np.linspace(0, resliced.shape[0], num=num_rows + 1)))
    ax1.yaxis.set_major_locator(FixedLocator(np.linspace(0, resliced.shape[1], num=num_cols + 1)))
    ax1.grid(which='major', alpha=alpha, color='white')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    filename = os.path.basename(mri_file)
    if filename.endswith('.gz'):
        filename = filename[:-7]
    else:
        filename = filename[:-4]
    file_path1 = os.path.join(save_path, filename + '_original_voxels.tiff')
    plt.savefig(file_path1, dpi=350)
    plt.close(fig1)

    # --- 2. Save figure with only the green rectangle (expanded_voxels) ---
    fig2, ax2 = plt.subplots()
    ax2.imshow(img_slice, cmap='gray', origin='lower')
    green_x0 = xx * pixel_spacing_row
    green_y0 = yx * pixel_spacing_col
    green_width = (xy - xx) * pixel_spacing_row
    green_height = (yy - yx) * pixel_spacing_col
    green_rect = patches.Rectangle(
        (green_x0, green_y0), green_width, green_height,
        linewidth=2, edgecolor='g', facecolor='none'  # Thicker line
    )
    ax2.add_patch(green_rect)
    ax2.xaxis.set_major_locator(FixedLocator(np.linspace(0, resliced.shape[0], num=num_rows + 1)))
    ax2.yaxis.set_major_locator(FixedLocator(np.linspace(0, resliced.shape[1], num=num_cols + 1)))
    ax2.grid(which='major', alpha=alpha, color='white')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    file_path2 = os.path.join(save_path, filename + '_expanded_voxels.tiff')
    plt.savefig(file_path2, dpi=350)
    plt.close(fig2)

    # --- 3. Save figure with both rectangles (default name) ---
    fig3, ax3 = plt.subplots()
    ax3.imshow(img_slice, cmap='gray', origin='lower')
    rect = patches.Rectangle((x0, y0), x_width, y_width, linewidth=1, edgecolor='r', facecolor='none')
    ax3.add_patch(rect)
    green_rect = patches.Rectangle(
        (green_x0, green_y0), green_width, green_height,
        linewidth=2, edgecolor='g', facecolor='none' # Thicker line
    )
    ax3.add_patch(green_rect)
    ax3.xaxis.set_major_locator(FixedLocator(np.linspace(0, resliced.shape[0], num=num_rows + 1)))
    ax3.yaxis.set_major_locator(FixedLocator(np.linspace(0, resliced.shape[1], num=num_cols + 1)))
    ax3.grid(which='major', alpha=alpha, color='white')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    # Add legend/key to the top right
    legend_handles = [
        patches.Patch(edgecolor='r', facecolor='none', linewidth=1, label='Measured VOI'),
        patches.Patch(edgecolor='g', facecolor='none', linewidth=2, label='Processed VOI')
    ]
    ax3.legend(handles=legend_handles, loc='upper right', frameon=True)

    file_path3 = os.path.join(save_path, filename + '.tiff')
    plt.savefig(file_path3, dpi=350)
    plt.close(fig3)

    # Return the calculated voxel indices for further use
    return {'xx': xx, 'xy': xy, 'yx': yx, 'yy': yy}

def create_control_file_lcm(raw_mrs_folder, corners, type):
    """Create LCModel control file with proper paths and voxel bounds"""
    # Create LCM directory inside raw MRS folder
    lcm_dir = Path(raw_mrs_folder) / 'lcm'
    lcm_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the .basis file in the folder
    basis_file = next(Path(raw_mrs_folder).glob('*.basis'))
    if not basis_file.exists():
        raise FileNotFoundError(f"Basis file not found in {raw_mrs_folder}")

    # Calculate voxel bounds from corners dictionary
    icolst = corners['xx'] +1
    icolen = corners['xy'] 
    irowst = corners['yx'] +1
    irowen = corners['yy'] 

    # Choose water reference file based on sequence type
    if type == 'STEAM':
        filh2o = Path(raw_mrs_folder) / 'STEAM_svRAW_32x32.RAW'
    else: 
        filh2o = Path(raw_mrs_folder) / 'Water_ref_RAW.RAW'

    # Prepare LCModel control file content
    control_content = f'''$LCMODL
wconc= 1.0
title= 'MRS Analysis'
savdir= '{lcm_dir.as_posix()}/'
ppmst= 4.0
ppmend= 0.2
nunfil= 1024
ndslic= 1
ndrows= 32
ndcols= 32
ltable= 7
lps= 8
lcsv= 11
key= 210387309
islice= 1
hzpppm= 1.2326e+02
filtab= '{lcm_dir.as_posix()}/table'
filraw= '{Path(raw_mrs_folder).as_posix()}/Metabolite_RAW.RAW'
filh2o= '{filh2o.as_posix()}'
filps= '{lcm_dir.as_posix()}/ps'
filcsv= '{lcm_dir.as_posix()}/spreadsheet.csv'
filbas= '{basis_file.as_posix()}'
echot= 40.00
dows= T
deltat= 5.000e-04
attmet= 1.0
atth2o= 1.0
irowst= {irowst}
irowen= {irowen}
icolst= {icolst}
icolen= {icolen}
$END'''

    # Save control file in the LCM directory
    control_file = Path(raw_mrs_folder)/ 'control.control'
    with open(control_file, 'w') as f:
        f.write(control_content)
    
    print(f"LCModel control file created at: {control_file}")
    return control_file


def GannetMask_SiemensRDA(fname, nii_file, output_path=None):
    """
    Generate a binary mask of the MRS voxel from a Siemens RDA file and save as NIfTI.
    Args:
        fname: Path to .rda file
        nii_file: Path to anatomical T1-weighted NIfTI
        output_path: Optional output path for mask
    Returns:
        Tuple: (mask_array, affine_matrix)
    """
    # --- 1. Parse RDA File for Voxel Parameters ---
    with open(fname, 'r', encoding='latin-1') as f:
        content = f.read()
    
    # Extract relevant parameters from the RDA header using regex
    params = {}
    for p in ['VOINormalSag', 'VOINormalCor', 'VOINormalTra',
              'VOIPositionSag', 'VOIPositionCor', 'VOIPositionTra',
              'VOIThickness', 'VOIReadoutFOV', 'VOIPhaseFOV',
              'VOIRotationInPlane']:
        match = re.search(rf'{p}:\s*([-\d.]+)', content)
        params[p] = float(match.group(1)) if match else 0.0

    # --- 2. Coordinate System Setup ---
    # Convert to RAS (Right-Anterior-Superior) coordinates
    position_ras = np.array([
        -params['VOIPositionSag'],  # Flip X
        -params['VOIPositionCor'],  # Flip Y
        params['VOIPositionTra']    # Keep Z
    ])
    voxel_size = np.array([
        params['VOIReadoutFOV'],  # X
        params['VOIPhaseFOV'],    # Y
        params['VOIThickness']    # Z
    ])

    # --- 3. Rotation Matrix ---
    Norm = np.array([
        -params['VOINormalSag'],
        -params['VOINormalCor'],
        params['VOINormalTra']
    ])
    ROT = np.radians(params['VOIRotationInPlane'])

    # Determine primary orientation and calculate phase vector
    maxdir = np.argmax(np.abs(Norm))
    vox_orient = ['s', 'c', 't'][maxdir]
    Phase = np.zeros(3)
    if vox_orient == 't':
        denom = Norm[1]**2 + Norm[2]**2
        if denom > 0:
            Phase[1] = Norm[2] * np.sqrt(1/denom)
            Phase[2] = -Norm[1] * np.sqrt(1/denom)
    elif vox_orient == 'c':
        denom = Norm[0]**2 + Norm[1]**2
        if denom > 0:
            Phase[0] = Norm[1] * np.sqrt(1/denom)
            Phase[1] = -Norm[0] * np.sqrt(1/denom)
    elif vox_orient == 's':
        denom = Norm[0]**2 + Norm[1]**2
        if denom > 0:
            Phase[0] = -Norm[1] * np.sqrt(1/denom)
            Phase[1] = Norm[0] * np.sqrt(1/denom)

    # Calculate readout vector and rotation matrix
    Readout = np.cross(Norm, Phase)
    rotmat = np.column_stack([Phase, Readout, Norm])

    # Apply in-plane rotation
    rotmat = rotmat @ np.array([
        [np.cos(ROT), -np.sin(ROT), 0],
        [np.sin(ROT), np.cos(ROT), 0],
        [0, 0, 1]
    ])

    # --- 4. Create Voxel Grid ---
    img = nib.load(nii_file)
    affine = img.affine
    dims = img.header['dim'][1:4]

    # Create meshgrid of voxel indices (1-based, then convert to 0-based)
    i, j, k = np.meshgrid(
        np.arange(1, dims[0]+1),
        np.arange(1, dims[1]+1),
        np.arange(1, dims[2]+1),
        indexing='ij' 
    )
    voxel_indices = np.column_stack([
        i.ravel()-1, 
        j.ravel()-1,
        k.ravel()-1
    ])

    # Transform to world coordinates using affine
    world_coords = nib.affines.apply_affine(affine, voxel_indices)

    # Apply half-voxel shift for SPM convention
    voxdim = np.abs(img.header.get_zooms()[:3])
    halfpixshift = -voxdim/2 * np.array([1, 1, -1])
    world_coords += halfpixshift

    # --- 5. Create Rectangular Voxel Mask ---
    # Transform coordinates to local voxel space
    inv_rot = np.linalg.inv(rotmat)
    local_coords = (inv_rot @ (world_coords - position_ras).T).T

    # Test which points are inside the voxel box
    mask = np.zeros(np.prod(dims))
    inside = np.all(np.abs(local_coords) <= voxel_size/2, axis=1)
    mask[inside] = 1
    mask = mask.reshape(dims)

    # --- 6. Save Output ---
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(fname),
            os.path.basename(fname).replace('.rda', '_mask.nii.gz')
        )
    
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), output_path)

    return mask, affine



def spectro_MRI_reslice(arguments):
    """
    Reslices anatomical MRI images into the space of the spectroscopy VOI (Volume Of Interest)
    using geometric information from a spectroscopy header file. Optionally applies PSF (Point Spread Function)
    correction and saves both high- and low-resolution images.

    Args:
        arguments (dict): Dictionary with the following keys:
            - 'reslicing_files_path' (str): Path to folder containing MRI images to be resliced.
            - 'header_text_path' (str): Path to the spectroscopy header text file.
            - 'PSF_corr' (bool): Whether to apply PSF correction (True/False).
            - 'hamm_width' (float, optional): Hamming filter width for PSF correction (required if PSF_corr is True).

    Workflow:
        - For each MRI image in the specified folder:
            - Extracts geometric parameters from the spectroscopy header.
            - Computes the affine transformation to match the spectroscopy VOI.
            - Reslices the MRI image into the spectroscopy space and saves it.
            - Downsamples the resliced image to the MRSI grid (low-res).
            - If PSF_corr is True, applies a Hamming filter in k-space before downsampling.
            - Saves info about the reslicing in a text file.

    Output:
        - Resliced MRI images in 'resliced_MRSI_space' subfolder.
        - Low-resolution images (with and without PSF correction).
        - Info file with grid and header details.

    Example:
        arguments = {
            'reslicing_files_path': '/path/to/mri/files',
            'header_text_path': '/path/to/header.txt',
            'PSF_corr': True,
            'hamm_width': 0.15
        }
        spectro_MRI_reslice(arguments)
    """
    # Get the folder containing MRI images to reslice
    root = arguments['reslicing_files_path']
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    for file in files:
        MRI_image_path = os.path.join(root, file)
        header_text_path = arguments['header_text_path']

        # Read spectroscopy header and extract relevant parameters
        file_header = open(header_text_path, "r")
        dict_params = dict()
        lines = file_header.readlines()
        words = ['SliceThickness:', 'PositionVector[0]:', 'PositionVector[1]:',
                 'PositionVector[2]:', 'RowVector[0]:', 'RowVector[1]:',
                 'RowVector[2]:', 'ColumnVector[0]:', 'ColumnVector[1]:',
                 'ColumnVector[2]:', 'VOINormalSag:', 'VOINormalCor:',
                 'VOINormalTra:','VOIPositionSag:', 'VOIPositionCor:',
                 'VOIPositionTra:', 'VOIPhaseFOV', 'VOIReadoutFOV',
                 'FoVHeight:', 'FoVWidth:', 'NumberOfRows:', 'NumberOfColumns:',
                 'CSIMatrixSizeOfScan[0]', 'CSIMatrixSizeOfScan[1]']
        for word in words:
            for line in lines:
                # Extract value for each parameter
                if line.find(word) != -1:
                    line_found = line
                    index = line.find(word)
                    dict_params[word] = line_found[(index + len(word) + 1):-1]
        file_header.close()      

        # Load the MRI image
        nii_image = load_img(MRI_image_path)

        # Extract orientation and position vectors from header
        row_vec = [dict_params['RowVector[0]:'], dict_params['RowVector[1]:'], dict_params['RowVector[2]:']]
        col_vec = [dict_params['ColumnVector[0]:'], dict_params['ColumnVector[1]:'], dict_params['ColumnVector[2]:']]
        s_vec = [dict_params['VOINormalSag:'], dict_params['VOINormalCor:'], dict_params['VOINormalTra:']]
        shift =  [dict_params['PositionVector[0]:'], dict_params['PositionVector[1]:'], dict_params['PositionVector[2]:']]
        mid_position = [dict_params['VOIPositionSag:'], dict_params['VOIPositionCor:'], dict_params['VOIPositionTra:']]

        # Convert vectors to float and adjust for coordinate system
        x1 = -1*np.float64(row_vec[0])
        y1 = -1*np.float64(row_vec[1])
        z1 =  1*np.float64(row_vec[2])
        x2 = -1*np.float64(col_vec[0])
        y2 = -1*np.float64(col_vec[1])
        z2 =  1*np.float64(col_vec[2])
        x3 = -1*np.float64(s_vec[0])
        y3 = -1*np.float64(s_vec[1])
        z3 =  1*np.float64(s_vec[2])

        row_vec_new = np.asarray([x1, y1, z1])
        col_vec_new = np.asarray([x2, y2, z2])

        # Compute the shift for the affine transformation
        shift1 = -np.float64(shift[0])
        shift2 = -np.float64(shift[1])
        shift3 =  np.float64(shift[2])
        shift_array = np.asarray([shift1, shift2, shift3])
        shift_array = shift_array + 0.5*(np.float64(row_vec_new)*1.0 + np.float64(col_vec_new)*1.0 - np.cross(np.float64(row_vec_new),np.float64(col_vec_new))*(np.float64(dict_params['SliceThickness:']) - 1.0))   

        # Build the affine matrix for reslicing
        affine_spec = np.asarray([[x1, x2, x3, shift_array[0]], [y1, y2, y3, shift_array[1]], [z1, z2, z3, shift_array[2]], [0, 0, 0, 1]])

        # Reslice the MRI image into the spectroscopy space
        resliced = resample_img(nii_image, target_affine=affine_spec, target_shape=[
            round(np.float64(dict_params['FoVWidth:'])),
            round(np.float64(dict_params['FoVHeight:'])),
            round(np.float64(dict_params['SliceThickness:']))**2
        ])

        # Find the slice corresponding to the VOI center
        affine_resliced = resliced.affine
        inv_affine_resliced = inv(affine_resliced)
        slice_position = inv_affine_resliced@[-np.float64(mid_position[0]), -np.float64(mid_position[1]), np.float64(mid_position[2]), 1]

        # Extract the relevant slices around the VOI
        img_resliced = resliced.get_fdata()[:,:,round(slice_position[2])-round(np.float64(dict_params['SliceThickness:'])/2):
                                       round(slice_position[2])+round(np.float64(dict_params['SliceThickness:'])/2)]
        if (round(slice_position[2])-round(np.float64(dict_params['SliceThickness:'])/2)) < 0:
            img_resliced = resliced.get_fdata()[:,:,round(slice_position[2])-round(np.float64(dict_params['SliceThickness:'])/2)+1:
                                           round(slice_position[2])+round(np.float64(dict_params['SliceThickness:'])/2)+1]

        # Save the resliced image as a new NIfTI file
        nii_resliced = nib.Nifti1Image(img_resliced, affine_resliced)
        if not os.path.isdir(os.path.join(root, 'resliced_MRSI_space')):
            os.mkdir(os.path.join(root, 'resliced_MRSI_space'))
        filename = os.path.basename(file)
        if filename[-3:] == '.gz':
            filename = filename[:-7]
        else:
            filename = filename[:-4]
        file_path = os.path.join(root, 'resliced_MRSI_space', filename+'_resliced_spec')
        nib.save(nii_resliced, file_path)

        # Downsample to MRSI grid (low resolution)
        print("Resampling the file %s"%file)
        data = nii_resliced.get_fdata()
        data_low_res = np.zeros([round(np.float64(dict_params['NumberOfRows:'])),
                                   round(np.float64(dict_params['NumberOfColumns:']))])
        for index in np.ndindex(data_low_res.shape):
            voxel_size_x = data.shape[0]/round(np.float64(dict_params['NumberOfRows:']))
            voxel_size_y = data.shape[0]/round(np.float64(dict_params['NumberOfColumns:']))
            x_0 = round(index[0]*voxel_size_x)
            x_1 = round(index[0]*voxel_size_x + voxel_size_x)
            y_0 = round(index[1]*voxel_size_y)
            y_1 = round(index[1]*voxel_size_y + voxel_size_y)
            data_low_res[index] = np.mean(data[x_0:x_1, y_0:y_1,:])
        nii_low_res = nib.Nifti1Image(data_low_res, affine=affine_resliced)
        nib.save(nii_low_res, file_path + '_resampled_NO_PSF_corr')

        # If requested, apply PSF correction using a Hamming filter in k-space
        if arguments['PSF_corr']:
            data_PSF = np.zeros(data.shape)
            for i in range(nii_resliced.shape[-1]):
                x0 = round(slice_position[0]-np.float64(dict_params['VOIPhaseFOV'])/2)
                y0 = round(slice_position[1]-np.float64(dict_params['VOIReadoutFOV'])/2)
                x_width = round(np.float64(dict_params['VOIPhaseFOV']))
                y_width = round(np.float64(dict_params['VOIReadoutFOV']))
                data_i = data[:,:,i]
                data_i[:y0,:] = 0
                data_i[y0+y_width:,:] = 0
                data_i[:,:x0] = 0
                data_i[:,x0+x_width:] = 0
                inversefft_PSF = np.fft.ifftshift(np.fft.ifft2(data_i)) # Inverse fft of map/image
                filt = np.zeros([data_i.shape[0], data_i.shape[1]])
                filt_mod_hamm = np.zeros([int(dict_params['CSIMatrixSizeOfScan[0]']), int(dict_params['CSIMatrixSizeOfScan[1]'])])
                middle_point = [int((filt_mod_hamm.shape[0]/2 - 0.5)), int((filt_mod_hamm.shape[1]/2 - 0.5))]
                ka_image = np.zeros(filt_mod_hamm.shape)
                for x in range(filt_mod_hamm.shape[0]):
                    for y in range(filt_mod_hamm.shape[1]):
                        k1_max = (filt_mod_hamm.shape[0]/2 - 0.5)
                        k2_max = (filt_mod_hamm.shape[1]/2 - 0.5)
                        delta_k = np.sqrt(((x - middle_point[0])/k1_max)**2 + ((y - middle_point[1])/k2_max)**2)
                        hamm_width = arguments['hamm_width']
                        ka = (delta_k - (1 - hamm_width))/hamm_width
                        ka_image[x,y] = ka
                        if ka < 0:
                            filt_mod_hamm[x,y] = 1
                        elif ka > 1:
                            filt_mod_hamm[x,y] = 0.08
                        else:
                            filt_mod_hamm[x,y] = 0.54 + 0.46*np.cos(np.pi*ka)
                filt[int(data_i.shape[0]/2 - int(dict_params['CSIMatrixSizeOfScan[0]'])/2):int(data_i.shape[0]/2 + int(dict_params['CSIMatrixSizeOfScan[0]'])/2), 
                     int(data_i.shape[1]/2 - int(dict_params['CSIMatrixSizeOfScan[1]'])/2):int(data_i.shape[1]/2 + int(dict_params['CSIMatrixSizeOfScan[1]'])/2)] = filt_mod_hamm
                inversefft_PSF = inversefft_PSF * filt
                final_img = np.abs(np.fft.fft2(np.fft.fftshift(inversefft_PSF)))
                data_PSF[:,:,i] = final_img
            data_low_res_PSF = np.zeros([round(np.float64(dict_params['NumberOfRows:'])),
                                       round(np.float64(dict_params['NumberOfColumns:']))])
            for index in np.ndindex(data_low_res_PSF.shape):
                voxel_size_x = data.shape[0]/round(np.float64(dict_params['NumberOfRows:']))
                voxel_size_y = data.shape[0]/round(np.float64(dict_params['NumberOfColumns:']))
                x_0 = round(index[0]*voxel_size_x)
                x_1 = round(index[0]*voxel_size_x + voxel_size_x)
                y_0 = round(index[1]*voxel_size_y)
                y_1 = round(index[1]*voxel_size_y + voxel_size_y)
                data_low_res_PSF[index] = np.mean(data_PSF[x_0:x_1, y_0:y_1,:])
            nii_low_res_PSF = nib.Nifti1Image(data_low_res_PSF, affine=affine_resliced)
            nib.save(nii_low_res_PSF, file_path + '_resampled_PSF_corr')

    # Save info about the reslicing process
    os.chdir(os.path.join(root, 'resliced_MRSI_space'))
    with open("info.txt", "w") as file:
        file.write("NumberOfRows = %s\n"%dict_params['NumberOfRows:'])
        file.write("NumberOfColumns = %s\n"%dict_params['NumberOfColumns:'])
        if arguments['PSF_corr']:
            file.write("PSF hamming width = %i\n"%hamm_width)
        file.write("Header path = %s"%header_text_path)
        file.close()
        

def read_rda(fileName,seq_type):
    """
    Reads the header and binary data from a Siemens MRS .rda file.

    The header is read using latin-1 encoding. This function extracts:
      - TE (echo time)
      - MRFrequency (used to compute hzpppm; no extra multiplication is applied)
      - VectorSize (number of data points per FID)
      - NumberOfRows, NumberOfColumns, NumberOf3DParts (for reshaping binary data)
      - PixelSpacingRow, PixelSpacingCol, PixelSpacing3D (used for voxel volume calculation)
      - The NMID block (which may include id, fmtdat, volume, tramp).
        If the NMID block is missing (or lacks an 'id'), an attempt is made to extract a PatientID;
        defaults are used otherwise.

    Parameters:
        fileName (str): Name of the file (with or without ".rda" extension)

    Returns:
        data_complex (np.ndarray): Complex data array with dimensions:
                                   (samples, NumberOfRows, NumberOfColumns, NumberOf3DParts)
        info (dict): Dictionary of extracted parameters. Keys include:
                     'TE', 'transmit_frequency', 'samples', 'dim', 'PixelSpacing',
                     plus a nested 'NMID' dict.
    """
    if not fileName.lower().endswith('.rda'):
        fileName += '.rda'
        
    header_lines = []
    with open(fileName, 'r', encoding='latin-1') as f:
        for line in f:
            header_lines.append(line.rstrip("\n"))
            if ">>> End of header <<<" in line:
                break
    
    info = {}
    # Extract selected parameters from the header.
    for ln in header_lines:
        if "TE:" in ln:
            try:
                info['TE'] = float(ln.split(':')[-1].strip())
            except Exception:
                pass
        elif "MRFrequency:" in ln:
            try:
                # Do not apply any multiplication; assume the header already has the correct Hz value.
                info['transmit_frequency'] = float(ln.split(':')[-1].strip())
            except Exception:
                pass
        elif "VectorSize:" in ln:
            try:
                info['samples'] = int(float(ln.split(':')[-1].strip()))
            except Exception:
                pass
        elif "NumberOfRows:" in ln:
            try:
                info.setdefault('dim', [None, None, None])
                info['dim'][0] = int(float(ln.split(':')[-1].strip()))
            except Exception:
                pass
        elif "NumberOfColumns:" in ln:
            try:
                info.setdefault('dim', [None, None, None])
                info['dim'][1] = int(float(ln.split(':')[-1].strip()))
            except Exception:
                pass
        elif "NumberOf3DParts:" in ln:
            try:
                info.setdefault('dim', [None, None, None])
                info['dim'][2] = int(float(ln.split(':')[-1].strip()))
            except Exception:
                pass
        elif "PixelSpacingRow:" in ln:
            try:
                info.setdefault('PixelSpacing', [None, None, None])
                info['PixelSpacing'][0] = float(ln.split(':')[-1].strip())
            except Exception:
                pass
        elif "PixelSpacingCol:" in ln:
            try:
                info.setdefault('PixelSpacing', [None, None, None])
                info['PixelSpacing'][1] = float(ln.split(':')[-1].strip())
            except Exception:
                pass
        elif "PixelSpacing3D:" in ln:
            try:
                info.setdefault('PixelSpacing', [None, None, None])
                info['PixelSpacing'][2] = float(ln.split(':')[-1].strip())
            except Exception:
                pass

    # --- Parse the NMID block ---
    nmid = {}
    in_nmid = False
    for ln in header_lines:
        if ln.strip() == "$NMID":
            in_nmid = True
            continue
        if in_nmid:
            if "$END" in ln:
                in_nmid = False
                break
            # Capture key=value pairs using a simple regex.
            m = re.search(r"(\w+)\s*=\s*'?([^'\s]+)'?", ln)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                if key.lower() in ['volume', 'tramp']:
                    try:
                        nmid[key] = float(val)
                    except Exception:
                        nmid[key] = val
                else:
                    nmid[key] = val
                        
    # If NMID block was not found or lacks an 'id', try to get a PatientID.
    if 'id' not in nmid:
        for ln in header_lines:
            if re.search(r"Patient\s*ID:", ln, re.IGNORECASE):
                nmid['id'] = ln.split(':')[-1].strip()
                break
        if 'id' not in nmid:
            nmid['id'] = "UNKNOWN"
    if 'fmtdat' not in nmid:
        nmid['fmtdat'] = "(2E15.6)"
    if 'volume' not in nmid:
        # Compute volume from PixelSpacing; assume these are in mm and convert to cm^3 
        # (1 cm^3 = 1000 mm^3).
        if 'PixelSpacing' in info and None not in info['PixelSpacing']:
            vol_mm3 = info['PixelSpacing'][0] * info['PixelSpacing'][1] * info['PixelSpacing'][2]
            nmid['volume'] = vol_mm3 / 1000.0
        else:
            nmid['volume'] = 1.0
    if 'tramp' not in nmid:
        nmid['tramp'] = 1.0
        
    info['NMID'] = nmid

    # Verify that required dimensions and sample size are present.
    if 'dim' not in info or any(x is None for x in info['dim']):
        raise ValueError("Missing dimension parameters in header.")
    if 'samples' not in info:
        raise ValueError("Missing VectorSize parameter in header.")
    
    # --- Read the binary data (after the text header) ---
    with open(fileName, 'rb') as f:
        # Skip the text header until the termination marker.
        while True:
            line = f.readline()
            if b'>>> End of header <<<' in line:
                break
        total_vals = info['dim'][0] * info['dim'][1] * info['dim'][2] * info['samples'] * 2

        data = np.fromfile(f, dtype=np.float64, count=total_vals)
    # Reshape data with Fortran order
    data = data.reshape((2, info['samples'], info['dim'][0], info['dim'][1], info['dim'][2]), order='F')
    data_complex = data[0, ...] + 1j * data[1, ...]
    
    return data_complex, info


def save_raw_text(data, fileName, info, seq_type):
    """
    Saves the complex data to a raw text file with header blocks and two-column data.

    The output header contains:
      $SEQPAR
       echot= <TE formatted to 2 decimals>
       hzpppm= <transmit_frequency formatted in scientific notation>
       [seq= 'STEAM']    <-- This line is added only if seq_type is 'Water_ref' (case-insensitive)
      $END
      $NMID
       id='<id from NMID>', fmtdat='<fmtdat from NMID>'
       volume=<volume formatted in scientific notation>
       tramp= <tramp value (formatted to 1 decimal)>
      $END

    Following the headers, the data are written as two columns:
      The first column is the real part and the second column is the imaginary part,
      each formatted in scientific notation.

    In addition, the multi-dimensional FIDs are averaged over the spatial dimensions (axes 1–3)
    so that a single FID (one complex value per sample) is output.

    Note: The data are flattened using Fortran (column-major) order.

    Parameters:
        data (np.ndarray): Complex data array with shape (samples, dim1, dim2, dim3)
        fileName (str): Base filename for output ('.raw' will be appended)
        info (dict): Extracted header parameters
        seq_type (str): If equal to 'STEAM' (case-insensitive), the seq line is included in $SEQPAR

    Returns:
        raw_filename (str): Name of the generated raw text file.
    """
    raw_filename = fileName + ".RAW"
    with open(raw_filename, 'w', encoding='latin-1') as f:
        # Write the $SEQPAR block.
        f.write("$SEQPAR\n")
        echot = info.get('TE', 40.00)
        f.write(f"echot= {echot:.2f}\n")
        # Conditionally include the seq line.
        if seq_type.strip().upper() == 'Water_ref':
            f.write("seq= 'STEAM'\n")
        hzpppm = info.get('transmit_frequency', 1.2326e+02)
        f.write(f"hzpppm= {hzpppm:.4e}\n")
        f.write("$END\n")
        
        # Write the $NMID block.
        nmid = info.get('NMID', {})
        f.write("$NMID\n")
        f.write(f"id='{nmid.get('id', 'UNKNOWN')} ', fmtdat='{nmid.get('fmtdat', '(2E15.6)')}'\n")
        f.write(f"volume={nmid.get('volume', 1.0):.3e}\n")
        f.write(f"tramp= {nmid.get('tramp', 1.0):.1f}\n")
        f.write("$END\n")

        data_flat = data.reshape(data.shape[0], -1)  # shape: (samples, voxels)
        num_voxels = data_flat.shape[1]
        num_points = data_flat.shape[0]

        for voxel in range(num_voxels):
            for i, c in enumerate(data_flat[:, voxel]):
                is_last = (voxel == num_voxels - 1) and (i == num_points - 1)
                if not is_last:
                    f.write(f"{c.real:.6e}   {c.imag:.6e}\n")
                else:
                    f.write(f"{c.real:.6e}   {c.imag:.6e}")  # no newline for the last line

    return raw_filename

def run_lcmodel(control_file_path, lcmodel_executable_path="lcmodel"):
    """
    Run LCModel using the provided control file and LCModel executable path.
    Args:
        control_file_path (str): Path to the LCModel control file.
        lcmodel_executable_path (str): Path to the LCModel binary (default assumes it's in PATH).
    """
    try:
        subprocess.run(
            f"{lcmodel_executable_path} < {control_file_path}",
            check=True,
            shell=True
        )

        print("LCModel run completed.")
    except subprocess.CalledProcessError as e:
        print(f"LCModel failed: {e}")
def get_output_filename():
    """Finds the first .rda file (excluding water_ref) and returns its name for CSV output."""
    rda_files = glob.glob('../Raw_MRS_files/*.rda')
    for file in rda_files:
        if 'water_ref' not in file:
            return os.path.splitext(os.path.basename(file))[0] + '.csv'
    return 'output.csv'  # Fallback if no matching .rda file is found

def combine_tables(filename: str) -> str:
    filename = filename.replace('^', '_')
    cur_dir = os.getcwd()
    path, name = os.path.split(filename)
    lcm_dir = os.path.join(path, 'lcm')
    os.chdir(lcm_dir)
    csv_name = get_output_filename()
    csv_file = os.path.join(lcm_dir, csv_name)
    
    # Delete existing CSV if present
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    # Gather all .table files
    table_files = [f for f in os.listdir() if f.endswith('.table')]
    if not table_files:
        raise FileNotFoundError('.table files not found! Please run LCModel analysis first')

    print('Combining the LCModel results table...')
    title = None
    default_title = ['row', 'col', 'FWHM', 'SNR', 'Ph_shift', 'Region']  # Default titles if parsing fails
    with open(csv_file, 'w', newline='') as fid_w:
        csv_writer = csv.writer(fid_w)
        for i, cur_file in enumerate(table_files):
            print(f'Processing file {i+1}/{len(table_files)}: {cur_file}')
            if cur_file != csv_name:
                try:
                    dataStruct = io_readlcmtab_dst(cur_file)
                    table_fieldnames = list(dataStruct.keys()) + ['Region']
                    if title is None:
                        title = table_fieldnames
                        csv_writer.writerow(title)
                    table_values = list(dataStruct.values())
                except Exception as e:
                    print(f'ERROR: Could not read the LCModel result .table file {cur_file}!')
                    print('Filling its data with zeros')
                    if title is None:
                        title = default_title
                        csv_writer.writerow(title)
                    table_values = [0] * len(title)

                region = re.split(f'_{name}', cur_file)[0]
                table_values.append(region)
                csv_writer.writerow(table_values)

    print('Finished combining tables!')
    os.chdir(cur_dir)
    return csv_file

def sanitize_field_name(name):
    """Replace invalid characters with underscores to mimic MATLAB's genvarname."""
    return re.sub(r'\W+', '_', name).strip('_')

def io_readlcmtab_dst(filename: str) -> dict:
    out = {'row': 1, 'col': 1, 'FWHM': 0.0, 'SNR': 0.0, 'Ph_shift': 0.0}
    
    # 1. Parse row and column from filename 
    try:
        # Find the last occurrence of '-' in the filename
        last_minus = filename.rfind('-')
        if last_minus != -1:
            # Extract substring around the last '-'
            substr_start = max(0, last_minus - 2)
            substr_end = min(len(filename), last_minus + 3)
            row_col_substr = filename[substr_start:substr_end]
            # Find all digit sequences in the substring
            digits = re.findall(r'\d+', row_col_substr)
            if len(digits) >= 2:
                out['row'], out['col'] = map(int, digits[:2])
    except Exception:
        pass  # Default to (1,1) on error
    
    # 2. Read FWHM, SNR, and Ph_shift from header
    with open(filename, 'r') as fid:
        print(f'Reading file: {filename}')
        # Find line with FWHM
        fwhm_line = None
        for line in fid:
            if 'FWHM' in line:
                fwhm_line = line
                break
        
        # Extract FWHM and SNR 
        if fwhm_line:
            equals = [i for i, c in enumerate(fwhm_line) if c == '=']
            ppm_idx = fwhm_line.find('ppm')
            if equals and ppm_idx != -1:
                fwhm_str = fwhm_line[equals[0]+1 : ppm_idx].strip()
                out['FWHM'] = float(fwhm_str) if fwhm_str else 0.0
            if len(equals) >= 2:
                snr_str = fwhm_line[equals[1]+1:].split()[0].strip()
                out['SNR'] = float(snr_str) if snr_str else 0.0
        # Read two lines after FWHM line and extract Ph_shift from the second line
        try:
            next(fid)  # Skip first line
            ph_line = next(fid)  # Second line
            ph_idx = ph_line.find('Ph')
            deg_idx = ph_line.find('deg', ph_idx)
            if ph_idx != -1 and deg_idx != -1:
                ph_shift_str = ph_line[ph_idx+4 : deg_idx].strip()
                out['Ph_shift'] = float(ph_shift_str) if ph_shift_str else 0.0
        except StopIteration:
            pass
    
    # 3. Reopen file to search for $$CONC from the start 
    with open(filename, 'r') as fid:
        conc_found = False
        # Find $$CONC section
        for line in fid:
            if '$$CONC' in line:
                conc_found = True
                break
        if not conc_found:
            return out
        
        # Skip one lines after $$CONC
        next(fid, None)
        
        # Process data lines 
        for line in fid:
            if len(line.strip()) <= 2:
                break  # Stop on empty line
            
            # Extract values using  column indices 
            name_part = line[23:].strip()  
            value = line[0:9].strip()      
            sd = line[10:13].strip()       
            cr = line[15:22].strip()      
            
            # Sanitize field names (mimic genvarname)
            name = sanitize_field_name(name_part)
            if name:
                out[name] = float(value) if value else 0.0
                out[f'{name}_%SD'] = float(sd) if sd else 0.0
                out[f'{name}/Cr'] = float(cr) if cr else 0.0
    
    return out

def modify_and_save_csv(csv_file: str):
    """Creates a modified version of the CSV with swapped row/col, renames columns, and sorts the data."""
    df = pd.read_csv(csv_file)
    
    # Swap row and col
    df.rename(columns={'row': 'i', 'col': 'j'}, inplace=True)
    
    # Sort by i then j
    df.sort_values(by=['i', 'j'], inplace=True)
    
    # Save the modified CSV
    modified_csv_file = os.path.join(os.path.dirname(csv_file), 'table_1h.csv')
    df.to_csv(modified_csv_file, index=False)
    print(f'Modified CSV saved as {modified_csv_file}')
    
    return modified_csv_file

   
def process_raw_file(input_file, output_file, repetitions=32*32):
    with open(input_file, "rb") as f:
        content = f.read()

    # Find the first $END in the content
    first_end = content.find(b"$END")
    if first_end == -1:
        print("First '$END' marker not found!")
        return
    
    # Find the second $END after the first one
    second_end = content.find(b"$END", first_end + len(b"$END"))
    if second_end == -1:
        print("Second '$END' marker not found!")
        return
    
    # The header size is up to the second $END marker (include the second $END)
    header_size = second_end + len(b"$END")

    # Extract the header and data after the second $END
    header = content[:header_size]
    raw_data = content[header_size:]

    # Repeat and write the header and data to a new file
    with open(output_file, "wb") as out_f:
        out_f.write(header)  # Write the header first
        for _ in range(repetitions):
            out_f.write(raw_data)  # Write the raw data repeatedly, ensuring no spaces between repetitions

    print(f"Processed data written to {output_file}")



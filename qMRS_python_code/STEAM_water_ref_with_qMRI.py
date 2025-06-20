
# === Imports ===
import os
import pandas
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# === Custom utility functions ===
from Functions import (
    spectro_MRI_reslice,
    save_spec_VOI_on_MRI_image,
    run_lcmodel,
    read_rda,
    save_raw_text,
    combine_tables, modify_and_save_csv, process_raw_file, GannetMask_SiemensRDA, create_control_file_lcm
)


# === Paths Configuration ===
input_folder = r'C:\Users\mmari\Desktop\qMRS'
raw_mrs_folder = os.path.join(input_folder, 'Raw_MRS_files')
reslicing_folder = os.path.join(input_folder, 'reslicing_HR_space')
resliced_folder = os.path.join(reslicing_folder, 'resliced_MRSI_space')

# === Step 1: Parse .rda files and create header.txt ===
rda_file = next((f for f in os.listdir(raw_mrs_folder) if f.endswith('.rda') and '_water_ref_' not in f), None)
if not rda_file:
    raise FileNotFoundError('No suitable .rda file found in Raw_MRS_files.')

rda_path = os.path.join(raw_mrs_folder, rda_file)
txt_filename = os.path.join(input_folder, 'header.txt')

# Extract and save the header part of the .rda file
with open(rda_path, 'r', encoding='latin-1') as f_in, open(txt_filename, 'w', encoding='latin-1') as f_out:
    for line in f_in:
        f_out.write(line)
        if 'End of header' in line:
            break
print(f'Header saved to {txt_filename}')

# === Step 2: Process Water Reference and Metabolite Signals ===
# Water reference
water_ref_rda = next((f for f in os.listdir(raw_mrs_folder) if f.endswith('.rda') and '_water_ref_' in f), None)
water_ref_rda_path = os.path.join(raw_mrs_folder, water_ref_rda)
output_raw_water_ref = os.path.join(raw_mrs_folder, 'Water_ref_RAW')
data, info = read_rda(water_ref_rda_path, seq_type='Water_ref')
save_raw_text(data, output_raw_water_ref, info, seq_type='Water_ref')

# Metabolite signal
output_raw_metabolite = os.path.join(raw_mrs_folder, 'Metabolite_RAW')
data_met, info_met = read_rda(rda_path, seq_type='')
save_raw_text(data_met, output_raw_metabolite, info_met, seq_type='')

# === Step 3: Reslice MRSI and anatomical images ===
reslice_args = {
    'reslicing_files_path': reslicing_folder,
    'header_text_path': txt_filename,
    'PSF_corr': True,
    'hamm_width': 0.5
}
# spectro_MRI_reslice(reslice_args)
print('Reslicing completed.')

# === Step 4: Overlay VOI on MRI ===
mri_file = os.path.join(resliced_folder, 'T1w_mag_brain_resliced_spec.nii')
corners =save_spec_VOI_on_MRI_image(mri_file, txt_filename, alpha=0.5)
# Extract values using their keys
xx = corners['xx'] # Left line x=xx
xy = corners['xy'] # Right line x=xy
yx = corners['yx'] # Low line y=yx
yy = corners['yy'] # Upper line y=yy
print("Corners:", xx, xy, yx, yy)
print('Spectroscopy data saved on MRI image.')

# === Step 5: Convert to 32x32 grid for LCModel ===
os.chdir(os.path.join(input_folder, 'Figures'))
raw_file = os.path.join(raw_mrs_folder, 'STEAM_svRAW.RAW')
output_file = os.path.join(raw_mrs_folder, 'STEAM_svRAW_32x32.RAW')
process_raw_file(raw_file, output_file)
print('STEAM single voxel raw file converted to 32x32 grid.')
# Update the control file with the new voxel bounds
create_control_file_lcm(raw_mrs_folder, corners,type='STEAM')

# === Step 6: Run LCModel ===
run_lcmodel(
    control_file_path=os.path.join(raw_mrs_folder, 'control.control'),
    lcmodel_executable_path=os.path.join(raw_mrs_folder, 'lcmodel.exe')
)
print('LCModel analysis completed.')

# === Step 7: Parse LCModel output table ===

results_folder = os.path.join(raw_mrs_folder, 'lcm')
# Create the folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)
csv_file = combine_tables(results_folder)
modify_and_save_csv(csv_file)
print('Table file created and saved.')

# === Step 8: Load and normalize tissue segmentations ===

table = pandas.read_csv(os.path.join(results_folder, 'table_1h.csv'))

H2O_slazer = nib.load(os.path.join(resliced_folder, 'H2O_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
CSF = nib.load(os.path.join(resliced_folder, 'c3T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
GM = nib.load(os.path.join(resliced_folder, 'c1T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
WM = nib.load(os.path.join(resliced_folder, 'c2T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()

# Normalize tissue masks so that their sum is 1
Total_mask = WM + GM + CSF
WM = WM/Total_mask
GM = GM/Total_mask
CSF = CSF/Total_mask

# === Step 9: Create STEAM voxel mask using Gannet ===
nii_file = os.path.join(reslicing_folder, 'T1w_mag_brain.nii.gz')
output_path = os.path.join(results_folder, 'STEAM_mask.nii')
GannetMask_SiemensRDA(water_ref_rda_path, nii_file, output_path)
print('STEAM voxel mask created.')

# === Step 10: Load additional maps for B1+/T1/T2* correction ===
st_mask = nib.load(output_path).get_fdata()
H2O = nib.load(os.path.join(reslicing_folder, 'H2O.nii')).get_fdata()
B1p = nib.load(os.path.join(reslicing_folder, 'B1_MAP_from_fast_EPI_standard_2_T1w_mag.nii.gz')).get_fdata()
T1 = nib.load(os.path.join(reslicing_folder, 'T1_map_B1corr_True_Spoilcorr_True_2echoes.nii')).get_fdata()
T2_star = nib.load(os.path.join(reslicing_folder, 'T2Star_avg.nii')).get_fdata()

T2 = T2_star * 60/50

H2O_st_on = np.mean(H2O[st_mask>0])
B1p_st_on = np.mean(B1p[st_mask>0])
T1_st_on = np.mean(T1[st_mask>0])
T2_st_on = np.mean(T2[st_mask>0])
FA_st= 90
st_prescan_on = 1
st_b1pcorr_prescan_on = st_prescan_on * np.sin(np.deg2rad(FA_st))/(np.sin(np.deg2rad(B1p_st_on*FA_st)))**3


# Relaxation correction 

FA = np.deg2rad(90)
TR = 10000
TE = 20
TM = 10
T1 = T1_st_on
SF = (1-np.exp((TE/2 + TM - TR)*1/T1))*np.exp(-TM/T1)

T2 = T2_st_on
T2F = np.exp(-TE/T2)
water_corr = SF * T2F

st_b1pcorr_prescan_on = st_b1pcorr_prescan_on


# Steam calibration to CSI SLASER
voxel_size_factor = 1

FA_st= 90
FA_slazer = 90
FA_factor = np.sin(np.deg2rad(FA_st))/np.sin(np.deg2rad(FA_slazer))



voxel_size_FA_factor = voxel_size_factor*FA_factor

st_b1pcorr_calib_to_slazer_prescan_on = st_b1pcorr_prescan_on/voxel_size_FA_factor*2 # for half signal

st_b1pcorr_calib_to_slazer_prescan_on = st_b1pcorr_calib_to_slazer_prescan_on

Water_STEAM = H2O_slazer*st_b1pcorr_calib_to_slazer_prescan_on/H2O_st_on


# Correction for CSF partial voluming 
H2O_GM_WM = H2O_slazer - CSF  #  Check, addition made on 11/04/2024; Equation 3 in Gasparovic 2009 is WRONG

# Correction for the relaxation correction
H2O_GM_WM = H2O_GM_WM/(water_corr*H2O_slazer)

# Factor conversion of Molality to Molarity

Molal_to_molar = (H2O_slazer - CSF*1)/(1-CSF)



# Acquisition parameters
FA = np.deg2rad(90)
TR = 2000
TE = 40
B1 = 1


# NAA


T1 = 1350
T2 = 295

T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab # Factor for metabolite relaxation

# Calculate number of voxels in x/y directions
n_voxels_y = xy - xx
n_voxels_x = yy - yx 

# Get corresponding region from segmentation maps
h2o_gm_wm_sub = H2O_GM_WM[yx:yy,xx:xy] 
molal_to_molar_sub = Molal_to_molar[yx:yy,xx:xy]
water_steam_sub = Water_STEAM[yx:yy,xx:xy]

image_NAA_slazer_on_lcm = np.reshape(np.asarray((table['NAA_NAAG'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_NAA_slazer_on_lcm_watercorr = image_NAA_slazer_on_lcm/h2o_gm_wm_sub # Multiply the water signal with the CSF partial voluming and relaxation correction factor
image_NAA_slazer_on_lcm_watercorr_PD_mMol = image_NAA_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_NAA_per_tissue = image_NAA_slazer_on_lcm_watercorr_PD_mMol
plt.figure()
plt.imshow(STEAM_qMRI_NAA_per_tissue, cmap = 'jet', origin = 'lower', vmin=7, vmax=14)
plt.title('tNAA [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_tNAA_per_tissue.tiff', dpi = 350)


image_NAA_slazer_on_lcm = np.reshape(np.asarray((table['NAA_NAAG'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_NAA_slazer_on_lcm_watercorr = image_NAA_slazer_on_lcm/h2o_gm_wm_sub
image_NAA_slazer_on_lcm_watercorr_PD_mMol = image_NAA_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_NAA_per_water = image_NAA_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(STEAM_qMRI_NAA_per_water, cmap = 'jet', origin = 'lower', vmin=10, vmax=18)
plt.title('tNAA [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_tNAA_per_water.tiff', dpi = 350)


#Cho

T1 = 1080
T2 = 187

T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Cho_slazer_on_lcm = np.reshape(np.asarray((table['GPC_Cho'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cho_slazer_on_lcm_watercorr = image_Cho_slazer_on_lcm/h2o_gm_wm_sub 
image_Cho_slazer_on_lcm_watercorr_PD_mMol = image_Cho_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Cho_per_tissue = image_Cho_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Cho_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=0, vmax=3)
plt.title('tCho [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Cho_per_tissue.tiff', dpi = 350)


image_Cho_slazer_on_lcm = np.reshape(np.asarray((table['GPC_Cho'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cho_slazer_on_lcm_watercorr = image_Cho_slazer_on_lcm/h2o_gm_wm_sub 
image_Cho_slazer_on_lcm_watercorr_PD_mMol = image_Cho_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Cho_per_water = image_Cho_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Cho_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=0, vmax=5)
plt.title('tCho [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Cho_per_water.tiff', dpi = 350)


# Cr

T1 = 1240
T2 = 156


T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Cr_slazer_on_lcm = np.reshape(np.asarray((table['Cr'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cr_slazer_on_lcm_watercorr = image_Cr_slazer_on_lcm/h2o_gm_wm_sub 
image_Cr_slazer_on_lcm_watercorr_PD_mMol = image_Cr_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Cr_per_tissue = image_Cr_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Cr_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=3, vmax=14)
plt.title('tCr [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Cr_per_tissue.tiff', dpi = 350)



image_Cr_slazer_on_lcm = np.reshape(np.asarray((table['Cr'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cr_slazer_on_lcm_watercorr = image_Cr_slazer_on_lcm/h2o_gm_wm_sub 
image_Cr_slazer_on_lcm_watercorr_PD_mMol = image_Cr_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Cr_per_water = image_Cr_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Cr_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=3, vmax=18)
plt.title('tCr [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Cr_per_water.tiff', dpi = 350)



# Glu


T1 = 960
T2 = 180

T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Glu_slazer_on_lcm = np.reshape(np.asarray((table['Glu'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Glu_slazer_on_lcm_watercorr = image_Glu_slazer_on_lcm/h2o_gm_wm_sub 
image_Glu_slazer_on_lcm_watercorr_PD_mMol = image_Glu_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Glu_per_tissue = image_Glu_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Glu_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=3, vmax=14)
plt.title('tGlu [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Glu_per_tissue.tiff', dpi = 350)



image_Glu_slazer_on_lcm = np.reshape(np.asarray((table['Glu'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Glu_slazer_on_lcm_watercorr = image_Glu_slazer_on_lcm/h2o_gm_wm_sub 
image_Glu_slazer_on_lcm_watercorr_PD_mMol = image_Glu_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Glu_per_water = image_Glu_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Glu_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=3, vmax=18)
plt.title('tGlu [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Glu_per_water.tiff', dpi = 350)



# Gln

T1 = 960
T2 = 180
T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Gln_slazer_on_lcm = np.reshape(np.asarray((table['Gln'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Gln_slazer_on_lcm_watercorr = image_Gln_slazer_on_lcm/h2o_gm_wm_sub 
image_Gln_slazer_on_lcm_watercorr_PD_mMol = image_Gln_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Gln_per_tissue = image_Gln_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Gln_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=0, vmax=8)
plt.title('tGln [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Gln_per_tissue.tiff', dpi = 350)



image_Gln_slazer_on_lcm = np.reshape(np.asarray((table['Gln'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Gln_slazer_on_lcm_watercorr = image_Gln_slazer_on_lcm/h2o_gm_wm_sub 
image_Gln_slazer_on_lcm_watercorr_PD_mMol = image_Gln_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Gln_per_water = image_Gln_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Gln_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=0, vmax=10)
plt.title('tGln [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Gln_per_water.tiff', dpi = 350)



# Lac


T1 = 2000
T2 = 240

T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Lac_slazer_on_lcm = np.reshape(np.asarray((table['Lac'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Lac_slazer_on_lcm_watercorr = image_Lac_slazer_on_lcm/h2o_gm_wm_sub 
image_Lac_slazer_on_lcm_watercorr_PD_mMol = image_Lac_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Lac_per_tissue = image_Lac_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Lac_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=0, vmax=30)
plt.title('tLac [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Lac_per_tissue.tiff', dpi = 350)



image_Lac_slazer_on_lcm = np.reshape(np.asarray((table['Lac'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Lac_slazer_on_lcm_watercorr = image_Lac_slazer_on_lcm/h2o_gm_wm_sub 
image_Lac_slazer_on_lcm_watercorr_PD_mMol = image_Lac_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Lac_per_water = image_Lac_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Lac_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=3, vmax=18)
plt.title('tLac [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Lac_per_water.tiff', dpi = 350)


# mins


T1 = 1010
T2 = 196

T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_mIns_slazer_on_lcm = np.reshape(np.asarray((table['mIns'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_mIns_slazer_on_lcm_watercorr = image_mIns_slazer_on_lcm/h2o_gm_wm_sub 
image_mIns_slazer_on_lcm_watercorr_PD_mMol = image_mIns_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_mIns_per_tissue = image_mIns_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_mIns_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=3, vmax=8)
plt.title('tmIns [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_mIns_per_tissue.tiff', dpi = 350)



image_mIns_slazer_on_lcm = np.reshape(np.asarray((table['mIns'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_mIns_slazer_on_lcm_watercorr = image_mIns_slazer_on_lcm/h2o_gm_wm_sub 
image_mIns_slazer_on_lcm_watercorr_PD_mMol = image_mIns_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_mIns_per_water = image_mIns_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_mIns_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=3, vmax=10)
plt.title('tmIns [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_mIns_per_water.tiff', dpi = 350)


# Ala


T1 = 1350
T2 = 295

T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Ala_slazer_on_lcm = np.reshape(np.asarray((table['Ala'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Ala_slazer_on_lcm_watercorr = image_Ala_slazer_on_lcm/h2o_gm_wm_sub 
image_Ala_slazer_on_lcm_watercorr_PD_mMol = image_Ala_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Ala_per_tissue = image_Ala_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Ala_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=5, vmax=70)
plt.title('tAla [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Ala_per_tissue.tiff', dpi = 350)



image_Ala_slazer_on_lcm = np.reshape(np.asarray((table['Ala'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Ala_slazer_on_lcm_watercorr = image_Ala_slazer_on_lcm/h2o_gm_wm_sub 
image_Ala_slazer_on_lcm_watercorr_PD_mMol = image_Ala_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Ala_per_water = image_Ala_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Ala_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=5, vmax=80)
plt.title('tAla [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Ala_per_water.tiff', dpi = 350)


# Gly


T1 = 1350
T2 = 295

T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Gly_slazer_on_lcm = np.reshape(np.asarray((table['Gly'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Gly_slazer_on_lcm_watercorr = image_Gly_slazer_on_lcm/h2o_gm_wm_sub 
image_Gly_slazer_on_lcm_watercorr_PD_mMol = image_Gly_slazer_on_lcm_watercorr * molal_to_molar_sub * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Gly_per_tissue = image_Gly_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Gly_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=0, vmax=0.6)
plt.title('tGly [mmol/L of tissue]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Gly_per_tissue.tiff', dpi = 350)



image_Gly_slazer_on_lcm = np.reshape(np.asarray((table['Gly'][:1600])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Gly_slazer_on_lcm_watercorr = image_Gly_slazer_on_lcm/h2o_gm_wm_sub 
image_Gly_slazer_on_lcm_watercorr_PD_mMol = image_Gly_slazer_on_lcm_watercorr * 55600/ (metab_corr*water_steam_sub)

STEAM_qMRI_Gly_per_water = image_Gly_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(image_Gly_slazer_on_lcm_watercorr_PD_mMol, cmap = 'jet', origin = 'lower', vmin=0, vmax=0.8)
plt.title('tGly [mmol/L of water]', pad=15)
plt.colorbar()
plt.savefig('qMRS_Gly_per_water.tiff', dpi = 350)




NAA_T1= 1.35
NAA_T2= 0.295
Glu_T1= 0.96
Glu_T2= 0.18
Gln_T1= 0.96
Gln_T2= 0.18
Cho_T1= 1.08
Cho_T2= 0.187
Cr_T1= 1.24
Cr_T2= 0.156
Lac_T1= 2
Lac_T2= 0.24 #Akshay Madan 2015
mins_T1 = 1.01
mins_T2= 0.196
ala_T1 = 1.35
ala_T2= 0.295 #Same with NAA
gly_T1 = 1.35
gly_T2= 0.295 #Same with NAA



NAA_tissue = STEAM_qMRI_NAA_per_tissue.ravel(order='C')
NAA_water = STEAM_qMRI_NAA_per_water.ravel(order='C')

Cho_tissue = STEAM_qMRI_Cho_per_tissue.ravel(order='C')
Cho_water = STEAM_qMRI_Cho_per_water.ravel(order='C')

Cr_tissue = STEAM_qMRI_Cr_per_tissue.ravel(order='C')
Cr_water = STEAM_qMRI_Cr_per_water.ravel(order='C')

Glu_tissue = STEAM_qMRI_Glu_per_tissue.ravel(order='C')
Glu_water = STEAM_qMRI_Glu_per_water.ravel(order='C')

Gln_tissue = STEAM_qMRI_Gln_per_tissue.ravel(order='C')
Gln_water = STEAM_qMRI_Gln_per_water.ravel(order='C')

Lac_tissue = STEAM_qMRI_Lac_per_tissue.ravel(order='C')
Lac_water = STEAM_qMRI_Lac_per_water.ravel(order='C')

mIns_tissue = STEAM_qMRI_mIns_per_tissue.ravel(order='C')
mIns_water = STEAM_qMRI_mIns_per_water.ravel(order='C')

Ala_tissue = STEAM_qMRI_Ala_per_tissue.ravel(order='C')
Ala_water = STEAM_qMRI_Ala_per_water.ravel(order='C')

Gly_tissue = STEAM_qMRI_Gly_per_tissue.ravel(order='C')
Gly_water = STEAM_qMRI_Gly_per_water.ravel(order='C')



i_index = np.repeat(np.linspace(1, 40, num=40, dtype=int), repeats=40)
j_index = np.tile(np.linspace(1, 40, num=40, dtype=int), reps=40)


qMRS_table = pandas.DataFrame()
table_dict = dict()


table_dict['i'] = {'i':i_index}
table_dict['j'] = {'j':j_index}
table_dict['NAA_NAAG [tissue]'] = {'NAA_NAAG [tissue]':NAA_tissue}
table_dict['Cr [tissue]'] = {'Cr [tissue]':Cr_tissue}
table_dict['GPC_Cho [tissue]'] = {'GPC_Cho [tissue]':Cho_tissue}
table_dict['Glu [tissue]'] = {'Glu [tissue]':Glu_tissue}
table_dict['Gln [tissue]'] = {'Gln [tissue]i':Gln_tissue}
table_dict['Lac [tissue]'] = {'Lac [tissue]':Lac_tissue}
table_dict['mIns [tissue]'] = {'mIns [tissue]':mIns_tissue}
table_dict['Ala [tissue]'] = {'Ala [tissue]':Ala_tissue}
table_dict['Gly [tissue]'] = {'Gly [tissue]':Gly_tissue}

table_dict['NAA_NAAG [water]'] = {'NAA_NAAG [water]':NAA_water}
table_dict['Cr [water]'] = {'Cr [water]':Cr_water}
table_dict['GPC_Cho [water]'] = {'GPC_Cho [water]':Cho_water}
table_dict['Glu [water]'] = {'Glu [water]':Glu_water}
table_dict['Gln [water]'] = {'Gln [water]i':Gln_water}
table_dict['Lac [water]'] = {'Lac [water]':Lac_water}
table_dict['mIns [water]'] = {'mIns [water]':mIns_water}
table_dict['Ala [water]'] = {'Ala [water]':Ala_water}
table_dict['Gly [water]'] = {'Gly [water]':Gly_water}


qMRS_table_tissue = pandas.concat([qMRS_table, pandas.DataFrame(table_dict['i']), pandas.DataFrame(table_dict['j']),
                           pandas.DataFrame(table_dict['NAA_NAAG [tissue]']), pandas.DataFrame(table_dict['Cr [tissue]']), 
                           pandas.DataFrame(table_dict['GPC_Cho [tissue]']), pandas.DataFrame(table_dict['Glu [tissue]']), 
                           pandas.DataFrame(table_dict['Gln [tissue]']), pandas.DataFrame(table_dict['Lac [tissue]']),
                           pandas.DataFrame(table_dict['mIns [tissue]']), pandas.DataFrame(table_dict['Ala [tissue]']),
                           pandas.DataFrame(table_dict['Gly [tissue]'])], axis=1)


qMRS_table_water = pandas.concat([pandas.DataFrame(table_dict['NAA_NAAG [water]']), pandas.DataFrame(table_dict['Cr [water]']), 
                            pandas.DataFrame(table_dict['GPC_Cho [water]']), pandas.DataFrame(table_dict['Glu [water]']), 
                            pandas.DataFrame(table_dict['Gln [water]']), pandas.DataFrame(table_dict['Lac [water]']),
                            pandas.DataFrame(table_dict['mIns [water]']), pandas.DataFrame(table_dict['Ala [water]']),
                            pandas.DataFrame(table_dict['Gly [water]'])], axis=1)


qMRS_table = pandas.concat([qMRS_table_tissue, qMRS_table_water], axis=1)

qMRS_table.to_excel('qMRS_results.xlsx')
qMRS_table.to_csv('qMRS_results.csv')

import pandas
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from Functions import (
    spectro_MRI_reslice,
    save_spec_VOI_on_MRI_image,
    run_lcmodel,
    read_rda,
    save_raw_text,
    combine_tables, modify_and_save_csv, process_raw_file, create_control_file_lcm
)


# === Paths Configuration ===
input_folder = r'C:\Users\mmari\Desktop\GLIOGLUT26'
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
# Analysis
reslice_args = {
    'reslicing_files_path': reslicing_folder,
    'header_text_path': txt_filename,
    'PSF_corr': True,
    'hamm_width': 0.5
}
# spectro_MRI_reslice(reslice_args)
print('Reslicing completed.')
# Volume of interest
mri_file = os.path.join(resliced_folder, 'T1w_mag_brain_resliced_spec.nii')
corners =save_spec_VOI_on_MRI_image(mri_file, txt_filename, alpha=0.5)
# Extract values using their keys
xx = corners['xx'] # Left line x=xx
xy = corners['xy']# Right line x=xy
yx = corners['yx']  # Low line y=yx
yy = corners['yy'] # Upper line y=yy
print("Corners:", xx, xy, yx, yy)
# Update the control file with the new voxel bounds
# Water reference
water_ref_rda = next((f for f in os.listdir(raw_mrs_folder) if f.endswith('.rda') and '_water_ref_' in f), None)
water_ref_rda_path = os.path.join(raw_mrs_folder, water_ref_rda)
output_raw_water_ref = os.path.join(raw_mrs_folder, 'Water_ref_RAW')
data, info = read_rda(water_ref_rda_path, seq_type='')
save_raw_text(data, output_raw_water_ref, info, seq_type='')

# Metabolite signal
output_raw_metabolite = os.path.join(raw_mrs_folder, 'Metabolite_RAW')
data_met, info_met = read_rda(rda_path, seq_type='')
save_raw_text(data_met, output_raw_metabolite, info_met, seq_type='')
create_control_file_lcm(raw_mrs_folder, corners,type='')

# === Step 6: Run LCModel ===
run_lcmodel(
    control_file_path=os.path.join(raw_mrs_folder, 'control.control'),
    lcmodel_executable_path=os.path.join(r'C:\Users\mmari\Desktop\qMRS\Raw_MRS_files', 'lcmodel.exe')
)


# === Step 7: Parse LCModel output table ===
os.chdir(os.path.join(input_folder, 'Figures'))
results_folder = os.path.join(raw_mrs_folder, 'lcm')
# Create the folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)
csv_file = combine_tables(results_folder)
modify_and_save_csv(csv_file)
print('Table file created and saved.')

# Spectroscopic file

table = pandas.read_csv(os.path.join(results_folder, 'table_1h.csv'))



CSF = nib.load(os.path.join(resliced_folder, 'c3T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
GM = nib.load(os.path.join(resliced_folder, 'c1T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
WM = nib.load(os.path.join(resliced_folder, 'c2T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()

# Correction for the fact that sum of the masks is greater than 1
Total_mask = WM + GM + CSF
WM = WM/Total_mask
GM = GM/Total_mask
CSF = CSF/Total_mask

# Relaxation and PD values initialise
WM_T1 = 878
WM_T2 = 58.68
WM_PD = 0.70

GM_T1 = 1425
GM_T2 = 69.45
GM_PD = 0.82

CSF_T1 = 4300
CSF_T2 = 2000
CSF_PD = 1.0

# Acquisition parameters
FA = np.deg2rad(90)
TR = 2000
TE = 40
B1 = 1

# T1 relaxation effects
SF_WM = (1-np.exp(-TR/WM_T1))/(1-np.exp(-TR/WM_T1)*np.cos(B1*FA))
SF_GM = (1-np.exp(-TR/GM_T1))/(1-np.exp(-TR/GM_T1)*np.cos(B1*FA))
SF_CSF = (1-np.exp(-TR/CSF_T1))/(1-np.exp(-TR/CSF_T1)*np.cos(B1*FA))

# T2 relaxation effects
T2F_WM = np.exp(-TE/WM_T2)
T2F_GM = np.exp(-TE/GM_T2)
T2F_CSF = np.exp(-TE/CSF_T2)

# T1 + T2 relaxation effects
water_corr_WM = SF_WM * T2F_WM
water_corr_GM = SF_GM * T2F_GM
water_corr_CSF = SF_CSF * T2F_CSF

# Fraction of the Proton density from the different tissue classes
WM_PD_frac = WM*WM_PD/(WM*WM_PD + GM*GM_PD + CSF*CSF_PD)
GM_PD_frac = GM*GM_PD/(WM*WM_PD + GM*GM_PD + CSF*CSF_PD)
CSF_PD_frac = CSF*CSF_PD/(WM*WM_PD + GM*GM_PD + CSF*CSF_PD)

# Correction for CSF partial voluming
H2O_GM_WM = (1-CSF_PD_frac)

# Correction for the relaxation correction
H2O_GM_WM = H2O_GM_WM/(WM_PD_frac*water_corr_WM + 
GM_PD_frac*water_corr_GM + CSF_PD_frac*water_corr_CSF)

# Factor conversion of Molality to Molarity

WM_added = WM + CSF * WM/(WM + GM) # Option 1, yields same results as option 2
GM_added = GM + CSF * GM/(WM + GM) # Option 1, yields same results as option 2

WM_added = WM/(WM + GM) # Option 2, yields same results as option 1
GM_added = GM/(WM + GM) # Option 2, yields same results as option 1

PD_WM_GM_added = WM_added*WM_PD + GM_added*GM_PD


# As an alternative, molar concentrations can be directly calculted from LCModel output (Gasprovic 2017, equation 15)
# Correction for CSF partial voluming
H2O_GM_WM_molar = (1-CSF)

# Correction for the relaxation correction
H2O_GM_WM_molar = H2O_GM_WM_molar/(WM*WM_PD*water_corr_WM + GM*GM_PD*water_corr_GM + CSF*CSF_PD*water_corr_CSF)

"NAA Conc. with water scaling using CSI slazer and LCModel"

# NAA

T1 = 1410
T2 = 271

TR = 2000
TE = 40
T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab # Factor for metabolite relaxation

n_voxels_y = xy - xx  
n_voxels_x = yy - yx  

h2o_gm_wm_sub = H2O_GM_WM[yx:yy,xx:xy] 
PD_WM_GM_added_sub = PD_WM_GM_added[yx:yy,xx:xy]


# Figures 
image_NAA_slazer_on_lcm = np.reshape(np.asarray((table['NAA_NAAG'][:1024])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_NAA_slazer_on_lcm_watercorr = image_NAA_slazer_on_lcm/h2o_gm_wm_sub # Multiply the water signal with the CSF partial voluming and relaxation correction factor
image_NAA_slazer_on_lcm_watercorr_PD_mMol = image_NAA_slazer_on_lcm_watercorr * PD_WM_GM_added_sub * 55510/ metab_corr


Ref_No_qMRI_NAA_per_tissue = image_NAA_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(Ref_No_qMRI_NAA_per_tissue, cmap = 'jet', origin = 'lower', vmin=7, vmax=14)
plt.title('Ref without qMRI (tNAA [mmol/L of tissue])', pad=15)
plt.colorbar()
plt.savefig('Ref_NOqMRI_NAA_per_tissue.tiff', dpi = 350)


image_NAA_slazer_on_lcm = np.reshape(np.asarray((table['NAA_NAAG'][:1024])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_NAA_slazer_on_lcm_watercorr = image_NAA_slazer_on_lcm/h2o_gm_wm_sub
image_NAA_slazer_on_lcm_watercorr_PD_mMol = image_NAA_slazer_on_lcm_watercorr * 55510/ metab_corr


Ref_No_qMRI_NAA_per_water = image_NAA_slazer_on_lcm_watercorr_PD_mMol

plt.figure()

plt.imshow(Ref_No_qMRI_NAA_per_water, cmap = 'jet', origin = 'lower', vmin=10, vmax=18)
plt.title('Ref without qMRI (tNAA [mmol/L of water])', pad=15)
plt.colorbar()
plt.savefig('Ref_NOqMRI_NAA_per_water.tiff', dpi = 350)


#Cho

T1 = 1190
T2 = 197

TR = 2000
TE = 40
T2F_metab = np.exp(-TE/T2)
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Cho_slazer_on_lcm = np.reshape(np.asarray((table['GPC_Cho'][:1024])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cho_slazer_on_lcm_watercorr = image_Cho_slazer_on_lcm/h2o_gm_wm_sub
image_Cho_slazer_on_lcm_watercorr_PD_mMol = image_Cho_slazer_on_lcm_watercorr * PD_WM_GM_added_sub * 55510/ metab_corr

Ref_No_qMRI_Cho_per_tissue = image_Cho_slazer_on_lcm_watercorr_PD_mMol

plt.figure()

plt.imshow(Ref_No_qMRI_Cho_per_tissue, cmap = 'jet', origin = 'lower', vmin=0, vmax=3)
plt.title('Ref without qMRI (Cho [mmol/L of tissue])', pad=15)
plt.colorbar()
plt.savefig('Ref_NOqMRI_Cho_per_tissue.tiff', dpi = 350)


image_Cho_slazer_on_lcm = np.reshape(np.asarray((table['GPC_Cho'][:1024])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cho_slazer_on_lcm_watercorr = image_Cho_slazer_on_lcm/h2o_gm_wm_sub
image_Cho_slazer_on_lcm_watercorr_PD_mMol = image_Cho_slazer_on_lcm_watercorr * 55510/ metab_corr

Ref_No_qMRI_Cho_per_water = image_Cho_slazer_on_lcm_watercorr_PD_mMol

plt.figure()

plt.imshow(Ref_No_qMRI_Cho_per_water, cmap = 'jet', origin = 'lower', vmin=0, vmax=5)
plt.title('Ref without qMRI (Cho [mmol/L of water])', pad=15)
plt.colorbar()
plt.savefig('Ref_NOqMRI_Cho_per_water.tiff', dpi = 350)



# Cr
TE = 40
T2 = 154
T2F_metab = np.exp(-TE/T2)

T1 = 1350
SF = (1-np.exp(-TR/T1))/(1-np.exp(-TR/T1)*np.cos(B1*FA))
metab_corr = SF * T2F_metab


image_Cr_slazer_on_lcm = np.reshape(np.asarray((table['Cr'][:1024])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cr_slazer_on_lcm_watercorr = image_Cr_slazer_on_lcm/h2o_gm_wm_sub
image_Cr_slazer_on_lcm_watercorr_PD_mMol = image_Cr_slazer_on_lcm_watercorr * PD_WM_GM_added_sub * 55510/ metab_corr

Ref_No_qMRI_Cr_per_tissue = image_Cr_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(Ref_No_qMRI_Cr_per_tissue, cmap = 'jet', origin = 'lower', vmin=3, vmax=14)
plt.title('Ref without qMRI (Cr [mmol/L of tissue])', pad=15)
plt.colorbar()
plt.savefig('Ref_NOqMRI_Cr_per_tissue.tiff', dpi = 350)


image_Cr_slazer_on_lcm = np.reshape(np.asarray((table['Cr'][:1024])), newshape=(n_voxels_x, n_voxels_y), order='C')
image_Cr_slazer_on_lcm_watercorr = image_Cr_slazer_on_lcm/h2o_gm_wm_sub
image_Cr_slazer_on_lcm_watercorr_PD_mMol = image_Cr_slazer_on_lcm_watercorr * 55510/ metab_corr


Ref_No_qMRI_Cr_per_water = image_Cr_slazer_on_lcm_watercorr_PD_mMol

plt.figure()
plt.imshow(Ref_No_qMRI_Cr_per_water, cmap = 'jet', origin = 'lower', vmin=3, vmax=18)
plt.title('Ref without qMRI (Cr [mmol/L of water])', pad=15)
plt.colorbar()
plt.savefig('Ref_NOqMRI_Cr_per_water.tiff', dpi = 350)
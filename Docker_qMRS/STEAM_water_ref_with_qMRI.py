"""
STEAM_water_ref_with_qMRI.py

This module provides a class for processing STEAM Single Voxel Water Reference MRS data with quantitative MRI (qMRI) corrections.
The main class, STEAMProcessor, is designed to be used within a pipeline object that provides paths and common data.
"""
import logging
import os
import numpy as np
import nibabel as nib
from Functions import GannetMask_SiemensRDA
import pandas as pd
import matplotlib.pyplot as plt


class STEAMProcessor:
    """
    Class for processing STEAM MRS data with qMRI correction.
    Performs mask creation, qMRI correction, metabolite quantification, and result visualization.
    """
    def __init__(self, pipeline):
        """
        Initialize the processor with a pipeline object containing paths and shared data.
        Args:
            pipeline: An object with attributes for file paths and common data.
        """
        self.pipeline = pipeline
        self.results = {}

    def process(self):
        """
        Main processing function for STEAM voxel quantification with qMRI correction.
        Steps:
        - Create STEAM voxel mask using Gannet.
        - Load tissue segmentations and qMRI maps.
        - Normalize tissue masks.
        - Compute water and B1 corrections within the STEAM voxel.
        - Apply relaxation correction.
        - Calibrate STEAM water reference to SLASER.
        - Correct for CSF partial voluming and relaxation.
        - Convert molality to molarity.
        - Quantify metabolites and generate result maps and plots.
        - Save quantitative results to Excel and CSV.
        """
        logging.info("Executing STEAM_water_ref_with_qMRI processing")
        # Create STEAM voxel mask using Gannet ===
        T1_brain = os.path.join(self.pipeline.paths['reslicing'], 'T1w_mag_brain.nii.gz')
        Steam_mask_path = os.path.join(self.pipeline.input_folder, 'STEAM_mask.nii')
        GannetMask_SiemensRDA(os.path.join(self.pipeline.paths['raw_mrs'], self.pipeline.water_ref_rda), T1_brain, Steam_mask_path)
        print('STEAM voxel mask created.')

        # Load tissue segmentations and qMRI maps
        H2O_slazer = nib.load(os.path.join(self.pipeline.paths['resliced'], 'H2O_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
        CSF = nib.load(os.path.join(self.pipeline.paths['resliced'], 'c3T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
        GM = nib.load(os.path.join(self.pipeline.paths['resliced'], 'c1T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
        WM = nib.load(os.path.join(self.pipeline.paths['resliced'], 'c2T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()

        # Load qMRI maps for the STEAM voxel
        H2O = nib.load(os.path.join(self.pipeline.paths['reslicing'], 'H2O.nii')).get_fdata()
        B1p = nib.load(os.path.join(self.pipeline.paths['reslicing'], 'B1_MAP.nii.gz')).get_fdata()
        T1 = nib.load(os.path.join(self.pipeline.paths['reslicing'], 'T1_map.nii')).get_fdata()
        T2_star = nib.load(os.path.join(self.pipeline.paths['reslicing'], 'T2Star_avg.nii')).get_fdata()
        
        # Load STEAM voxel mask
        steam_mask = nib.load(Steam_mask_path).get_fdata()

        # Normalize tissue masks so that their sum is 1 (partial volume correction)
        Total_mask = WM + GM + CSF
        WM = WM/Total_mask
        GM = GM/Total_mask
        CSF = CSF/Total_mask
        T2 = T2_star * 59/47  # Adjust T2* to T2

        # Compute mean values within the STEAM voxel
        H2O_st_on = np.mean(H2O[steam_mask>0])
        B1p_st_on = np.mean(B1p[steam_mask>0])
        T1_st_on = np.mean(T1[steam_mask>0])
        T2_st_on = np.mean(T2[steam_mask>0])
        FA_st= 90
        st_prescan_on = 1
        # B1+ correction for STEAM voxel
        st_b1pcorr_prescan_on = st_prescan_on * np.sin(np.deg2rad(FA_st))/(np.sin(np.deg2rad(B1p_st_on*FA_st)))**3

        # Relaxation correction 
        FA = np.deg2rad(90)
        TR = 10000
        TE = 20
        TM = 10
        T1 = T1_st_on
        # Signal fraction for relaxation
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
        # Calibration factor for STEAM to SLASER
        st_b1pcorr_calib_to_slazer_prescan_on = st_b1pcorr_prescan_on/voxel_size_FA_factor*2 # for half signal
        st_b1pcorr_calib_to_slazer_prescan_on = st_b1pcorr_calib_to_slazer_prescan_on
        Water_STEAM = H2O_slazer*st_b1pcorr_calib_to_slazer_prescan_on/H2O_st_on

        # Correction for CSF partial voluming 
        # (Equation 3 in Gasparovic 2009 is noted as incorrect)
        H2O_GM_WM = H2O_slazer - CSF  #  Check, addition made on 11/04/2024

        # Correction for the relaxation correction
        H2O_GM_WM = H2O_GM_WM/(water_corr*H2O_slazer)

        # Factor conversion of Molality to Molarity
        Molal_to_molar = (H2O_slazer - CSF*1)/(1-CSF)

        # Acquisition parameters for metabolite quantification
        FA = np.deg2rad(90)
        TR = 2000
        TE = 40
        B1 = 1

        # Define metabolites and their parameters (T1, T2, display ranges)
        metabolites = {
            'NAA_NAAG': {'T1': 1410, 'T2': 271, 'tissue_range': (7, 14), 'water_range': (10, 18)},
            'GPC_Cho': {'T1': 1190, 'T2': 197, 'tissue_range': (0, 3), 'water_range': (0, 5)},
            'Cr': {'T1': 1350, 'T2': 154, 'tissue_range': (3, 14), 'water_range': (3, 18)},
            'Glu': {'T1': 960, 'T2': 180, 'tissue_range': (3, 14), 'water_range': (3, 18)},
            'Gln': {'T1': 960, 'T2': 180, 'tissue_range': (0, 8), 'water_range': (0, 10)},
            'Lac': {'T1': 2000, 'T2': 240, 'tissue_range': (0, 30), 'water_range': (3, 18)},
            'mIns': {'T1': 1010, 'T2': 196, 'tissue_range': (3, 8), 'water_range': (3, 10)},
            'Ala': {'T1': 1350, 'T2': 295, 'tissue_range': (5, 70), 'water_range': (5, 80)},
            'Gly': {'T1': 1350, 'T2': 295, 'tissue_range': (0, 0.6), 'water_range': (0, 0.8)}
        }

        # Calculate number of voxels in x/y directions for the quantification grid
        n_voxels_y = self.pipeline.common_data['xy'] - self.pipeline.common_data['xx']  
        n_voxels_x = self.pipeline.common_data['yy'] - self.pipeline.common_data['yx']  

        # Get corresponding region from segmentation maps
        h2o_gm_wm_sub = H2O_GM_WM[self.pipeline.common_data['yx']:self.pipeline.common_data['yy'], 
                                self.pipeline.common_data['xx']:self.pipeline.common_data['xy']] 
        molal_to_molar_sub = Molal_to_molar[self.pipeline.common_data['yx']:self.pipeline.common_data['yy'], 
                                        self.pipeline.common_data['xx']:self.pipeline.common_data['xy']]
        water_steam_sub = Water_STEAM[self.pipeline.common_data['yx']:self.pipeline.common_data['yy'], 
                                    self.pipeline.common_data['xx']:self.pipeline.common_data['xy']]

        results = {}
        
        for metab, params in metabolites.items():
            # Calculate metabolite correction factor (relaxation and B1)
            T2F_metab = np.exp(-TE/params['T2'])
            SF = (1-np.exp(-TR/params['T1']))/(1-np.exp(-TR/params['T1'])*np.cos(B1*FA))
            metab_corr = SF * T2F_metab

            # Process tissue concentration
            image_metab = np.reshape(
                np.asarray((self.pipeline.common_data['results'][metab][:1600])), 
                newshape=(n_voxels_x, n_voxels_y), 
                order='C'
            )
            image_metab_watercorr = image_metab/h2o_gm_wm_sub
            image_metab_watercorr_PD_mMol = image_metab_watercorr * molal_to_molar_sub * 55510/ (metab_corr*water_steam_sub)
            
            # Process water concentration
            image_metab_watercorr_PD_mMol_water = image_metab_watercorr * 55510/ (metab_corr*water_steam_sub)
            # Store results
            results[metab] = {
                'tissue': image_metab_watercorr_PD_mMol,
                'water': image_metab_watercorr_PD_mMol_water
            }
            
            # Generate plots for each metabolite
            self._generate_metab_plots(metab, image_metab_watercorr_PD_mMol, image_metab_watercorr_PD_mMol_water, params)
        
        # Save quantitative results to Excel and CSV
        self._save_quantitative_results(results)

    def _generate_metab_plots(self, metab_name, tissue_data, water_data, params):
        """
        Helper function to generate and save plots for a single metabolite.
        Args:
            metab_name (str): Name of the metabolite.
            tissue_data (ndarray): Quantitative map per tissue.
            water_data (ndarray): Quantitative map per water.
            params (dict): Display range and other parameters.
        """
        plt.figure()
        plt.imshow(tissue_data, cmap='jet', origin='lower', vmin=params['tissue_range'][0], vmax=params['tissue_range'][1])
        plt.title(f't{metab_name} [mmol/L of tissue]', pad=15)
        plt.colorbar()
        plt.savefig(os.path.join(self.pipeline.paths['Results'], f'qMRS_{metab_name}_per_tissue.tiff'), dpi=350)
        plt.close()
        
        plt.figure()
        plt.imshow(water_data, cmap='jet', origin='lower', vmin=params['water_range'][0], vmax=params['water_range'][1])
        plt.title(f't{metab_name} [mmol/L of water]', pad=15)
        plt.colorbar()
        plt.savefig(os.path.join(self.pipeline.paths['Results'], f'qMRS_{metab_name}_per_water.tiff'), dpi=350)
        plt.close()

    def _save_quantitative_results(self, results):
        """
        Save all quantitative results to Excel and CSV files.
        Args:
            results (dict): Dictionary of metabolite results.
        """
        # Prepare data for DataFrame
        data_dict = {}
        
        # Get actual data dimensions from results
        sample_metab = next(iter(results.values()))
        n_voxels_x, n_voxels_y = sample_metab['tissue'].shape
        
        # Generate accurate indices for each voxel
        i_index = np.repeat(np.arange(1, n_voxels_x+1), n_voxels_y)
        j_index = np.tile(np.arange(1, n_voxels_y+1), n_voxels_x)
        
        data_dict['i'] = i_index
        data_dict['j'] = j_index
        
        # Add metabolite concentrations to the DataFrame
        for metab, values in results.items():
            data_dict[f'{metab} [tissue]'] = values['tissue'].ravel(order='C')
            data_dict[f'{metab} [water]'] = values['water'].ravel(order='C')
        
        # Create and save DataFrame
        qMRS_table = pd.DataFrame(data_dict)
        qMRS_table.to_excel(os.path.join(self.pipeline.paths['Results'], 'qMRS_results.xlsx'))
        qMRS_table.to_csv(os.path.join(self.pipeline.paths['Results'], 'qMRS_results.csv'))

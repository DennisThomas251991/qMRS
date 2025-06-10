"""
MRSI_water_ref_no_qMRI.py

This module provides a class for processing MRSI (Magnetic Resonance Spectroscopic Imaging) data
using literature water T1 and T2 values instead of quantitative MRI (qMRI) corrections. 
The main class, MRSIProcessor, is designed to be used within a pipeline object that provides paths and common data.
"""
import logging
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


class MRSIProcessor:
    """
    Class for processing MRSI data using water reference scaling without qMRI corrections.
    Handles tissue segmentation, relaxation correction, metabolite quantification, and result visualization.
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
        Main processing function for MRSI quantification using literature-based corrections.
        Steps:
        - Load tissue segmentations.
        - Normalize tissue masks so their sum is 1 (partial volume correction).
        - Apply literature values for T1, T2, and proton density (PD) for each tissue.
        - Compute water correction factors for each tissue.
        - Calculate partial volume fractions and correct for CSF.
        - Calculate quantitative metabolite maps (per tissue and per water).
        - Generate and save result plots.
        - Save quantitative results to Excel and CSV.
        """
        logging.info("Executing MRSI_water_ref_no_qMRI processing")
        
        # Load tissue segmentations
        CSF = nib.load(os.path.join(self.pipeline.paths['resliced'], 'c3T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
        GM = nib.load(os.path.join(self.pipeline.paths['resliced'], 'c1T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()
        WM = nib.load(os.path.join(self.pipeline.paths['resliced'], 'c2T1w_mag_resliced_spec_resampled_PSF_corr.nii')).get_fdata()

        # Correction for the fact that sum of the masks is greater than 1
        Total_mask = WM + GM + CSF
        WM = WM/Total_mask
        GM = GM/Total_mask
        CSF = CSF/Total_mask

        # Relaxation and PD values from literature (Gasparovic et al.)
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

        # T1 relaxation effects for each tissue
        SF_WM = (1-np.exp(-TR/WM_T1))/(1-np.exp(-TR/WM_T1)*np.cos(B1*FA))
        SF_GM = (1-np.exp(-TR/GM_T1))/(1-np.exp(-TR/GM_T1)*np.cos(B1*FA))
        SF_CSF = (1-np.exp(-TR/CSF_T1))/(1-np.exp(-TR/CSF_T1)*np.cos(B1*FA))

        # T2 relaxation effects for each tissue
        T2F_WM = np.exp(-TE/WM_T2)
        T2F_GM = np.exp(-TE/GM_T2)
        T2F_CSF = np.exp(-TE/CSF_T2)

        # Combined T1 + T2 relaxation effects
        water_corr_WM = SF_WM * T2F_WM
        water_corr_GM = SF_GM * T2F_GM
        water_corr_CSF = SF_CSF * T2F_CSF

        # Fraction of the proton density from the different tissue classes
        WM_PD_frac = WM*WM_PD/(WM*WM_PD + GM*GM_PD + CSF*CSF_PD)
        GM_PD_frac = GM*GM_PD/(WM*WM_PD + GM*GM_PD + CSF*CSF_PD)
        CSF_PD_frac = CSF*CSF_PD/(WM*WM_PD + GM*GM_PD + CSF*CSF_PD)

        # Correction for CSF partial voluming (fraction of GM+WM)
        H2O_GM_WM = (1-CSF_PD_frac)

        # Correction for the relaxation correction (Gasparovic 2006)
        H2O_GM_WM = H2O_GM_WM/(WM_PD_frac*water_corr_WM + GM_PD_frac*water_corr_GM + CSF_PD_frac*water_corr_CSF)

        # Factor conversion of Molality to Molarity (two options, both yield same result)
        WM_added = WM + CSF * WM/(WM + GM) # Option 1
        GM_added = GM + CSF * GM/(WM + GM) # Option 1

        WM_added = WM/(WM + GM) # Option 2
        GM_added = GM/(WM + GM) # Option 2

        PD_WM_GM_added = WM_added*WM_PD + GM_added*GM_PD

        # Alternative: molar concentrations directly from LCModel output (Gasparovic 2017, eq. 15)
        H2O_GM_WM_molar = (1-CSF)
        H2O_GM_WM_molar = H2O_GM_WM_molar/(WM*WM_PD*water_corr_WM + GM*GM_PD*water_corr_GM + CSF*CSF_PD*water_corr_CSF)

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
        PD_WM_GM_added_sub = PD_WM_GM_added[self.pipeline.common_data['yx']:self.pipeline.common_data['yy'], 
                                self.pipeline.common_data['xx']:self.pipeline.common_data['xy']] 

        results = {}
        
        for metab, params in metabolites.items():
            # Calculate metabolite correction (relaxation and B1)
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
            image_metab_watercorr_PD_mMol = image_metab_watercorr * PD_WM_GM_added_sub * 55510/ metab_corr
            # Process water concentration
            image_metab_watercorr_PD_mMol_water = image_metab_watercorr * 55510/ metab_corr
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
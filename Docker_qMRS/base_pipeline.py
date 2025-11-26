import os
import logging
import pandas as pd 
from glob import glob  
from Functions import (
    spectro_MRI_reslice,      
    save_spec_VOI_on_MRI_image, 
    run_lcmodel,             
    read_rda,               
    save_raw_text,            
    combine_tables,           
    modify_and_save_csv,      
    process_raw_file,         
    create_control_file_lcm   
)

class MRSPipelineBase:
    """Base class for MRS processing pipeline with common functionality
    
    Attributes:
        input_folder (str): Root directory containing input data
        method (str): Processing methodology selection
        common_data (dict): Shared data storage between processing steps
        paths (dict): Configured file system paths for data organization
    """
    
    def __init__(self, input_folder, method):
        """Initialize pipeline with input parameters and path configuration
        
        Args:
            input_folder (str): Path to root directory containing input data
            method (str): Processing method (MRSI_water_ref_no_qMRI/STEAM_water_ref_with_qMRI)
        """
        self.input_folder = input_folder
        self.method = method
        self.common_data = {}  # Shared data storage
        
        # Configure standardized directory paths
        self.paths = {
            'raw_mrs': os.path.join(input_folder, 'Raw_MRS_files'),        # Raw spectroscopy data
            'reslicing': os.path.join(input_folder, 'reslicing_HR_space'),  # Anatomical reference data
            'resliced': os.path.join(input_folder, 'reslicing_HR_space', 'resliced_MRSI_space'),  # Resliced data
            'Results': os.path.join(input_folder, 'Results')                # Output Folder
        }

    def validate_inputs(self):
        """Validate input directory structure and required files based on processing method
        
        Raises:
            FileNotFoundError: If required files for selected method are missing
        """
        # Define file requirements for each processing method
        required = {
            'MRSI_water_ref_no_qMRI': {
                'raw_mrs': [
                    '*.rda',    # Raw spectroscopy files
                    '*.basis'   # Basis set files
                ],
                'reslicing': [
                    'c1T1w_mag.nii*',  # Gray matter segmentation
                    'c2T1w_mag.nii*',  # White matter segmentation
                    'c3T1w_mag.nii*',  # CSF segmentation
                    'T1w_mag_brain.nii*'  # Skull-stripped T1 image
                ]
            },
            'STEAM_water_ref_with_qMRI': {
                'raw_mrs': ['*.rda', '*.basis'],
                'reslicing': [
                    'c1T1w_mag.nii*',      # Gray matter segmentation
                    'c2T1w_mag.nii*',      # White matter segmentation
                    'c3T1w_mag.nii*',      # CSF segmentation
                    'T1_map*.nii',        # Quantitative T1 map
                    'H2O.nii',             # H2O map
                    'T2Star_avg.nii',      # T2* map
                    'B1_MAP*.nii.gz',     # B1 map
                    'T1w_mag_brain.nii.gz' # Skull-stripped T1 image
                ]
            }
        }

        missing = []
        # Validate files in each required subdirectory
        for subfolder, patterns in required[self.method].items():
            base_dir = self.paths[subfolder]
            for pattern in patterns:
                if not glob(os.path.join(base_dir, pattern)):
                    missing.append(os.path.join(base_dir, pattern))

        if missing:
            raise FileNotFoundError(
                f"Missing required files for {self.method}:\n" + "\n".join(missing))
        logging.info("Input validation completed successfully")

    def execute_common_steps(self):
        """Execute processing steps common to both methodologies
        
        Returns:
            bool: True if all steps completed successfully, False otherwise
        """
        try:
            # Core processing workflow
            self._process_header()            # Extract and save header information
            self._convert_raw_files()         # Convert raw data formats
            self._reslice_images()            # Reslice the maps
            self._process_voi()               # Volume of Interest definition
            self._run_lcmodel_processing()    # LC Model analysis
            
            return True
        except Exception as e:
            logging.error(f"Common processing failure: {str(e)}")
            return False

    def _process_header(self):
        """Extract and save header information from metabolite RDA file
        
        Identifies and processes:
        - Metabolite data file (water_ref not in filename)
        - Water reference file (water_ref in filename)
        """
        # Locate relevant RDA files
        self.metabolite_rda = next(
            f for f in os.listdir(self.paths['raw_mrs'])
            if f.endswith('.rda') and 'water_ref' not in f
        )
        self.water_ref_rda = next(
            f for f in os.listdir(self.paths['raw_mrs'])
            if 'water_ref' in f and f.endswith('.rda')
        )

        # Extract header section from metabolite file
        header_path = os.path.join(self.input_folder, 'header.txt')
        with open(os.path.join(self.paths['raw_mrs'], self.metabolite_rda), 
                 'r', encoding='latin-1') as src_file, \
             open(header_path, 'w', encoding='latin-1') as dest_file:
            for line in src_file:
                dest_file.write(line)
                if 'End of header' in line:
                    break  # Stop at end of header section
        print(f'Metadata header saved to: {header_path}')

    def _convert_raw_files(self):
        """Convert RDA files to LCModel-compatible .RAW format
        
        Handles special case for STEAM water reference requiring grid expansion
        """
        # Process water reference
        water_data, water_info = read_rda(
            os.path.join(self.paths['raw_mrs'], self.water_ref_rda), 
            seq_type=''
        )
        if self.method == 'STEAM_water_ref_with_qMRI':
            self._process_steam_water(water_data, water_info)
        else:
            save_raw_text(
                water_data, 
                os.path.join(self.paths['raw_mrs'], 'Water_ref_RAW'),
                water_info, 
                seq_type=''
            )

        # Process metabolite data
        metab_data, metab_info = read_rda(
            os.path.join(self.paths['raw_mrs'], self.metabolite_rda),
            seq_type=''
        )
        save_raw_text(
            metab_data,
            os.path.join(self.paths['raw_mrs'], 'Metabolite_RAW'),
            metab_info,
            seq_type=''
        )

    def _process_steam_water(self, water_data, water_info):
        """Special processing for STEAM water reference data
        
        Expands single-voxel STEAM data to match MRSI grid dimensions (32*32)
        """
        save_raw_text(
            water_data,
            os.path.join(self.paths['raw_mrs'], 'STEAM_svRAW'),
            water_info, 
            seq_type='STEAM'
        )
        # Expand single voxel to 32x32 grid
        raw_file = os.path.join(self.paths['raw_mrs'], 'STEAM_svRAW.RAW')
        processed_raw = os.path.join(self.paths['raw_mrs'], 'STEAM_svRAW_32x32.RAW')
        process_raw_file(raw_file, processed_raw)

    def _reslice_images(self):
        """Reslice anatomical images to MRSI acquisition space
        """
        spectro_MRI_reslice({
            'reslicing_files_path': self.paths['reslicing'],  # Source directory
            'header_text_path': os.path.join(self.input_folder, 'header.txt'),  
            'PSF_corr': True,    
            'hamm_width': 0.5    
        })

    def _process_voi(self):
        """Define and save Volume of Interest (VOI) on anatomical reference
        
        Stores VOI corner coordinates in common_data for subsequent processing
        """
        mri_path = os.path.join(self.paths['resliced'], 'T1w_mag_brain_resliced_spec.nii')
        os.makedirs(self.paths['Results'], exist_ok=True)  # Ensure directory 'Results' where all output files will be saved exists
        save_path = self.paths['Results']
        self.voi_corners = save_spec_VOI_on_MRI_image(
            mri_path, 
            os.path.join(self.input_folder, 'header.txt'),save_path=save_path
        )
        self.common_data.update(self.voi_corners)

    def _run_lcmodel_processing(self):
        """Execute LCModel workflow for metabolite quantification
        
        Generates control file, runs analysis, and processes output tables
        """
        # Create method-specific control file
        create_control_file_lcm(
            self.paths['raw_mrs'], 
            self.voi_corners,
            'STEAM' if self.method == 'STEAM_water_ref_with_qMRI' else ''
        )
        
        # LCModel execution
        run_lcmodel(
            control_file_path=os.path.join(self.paths['raw_mrs'], 'control.control'),
            lcmodel_executable_path='/opt/lcmodel'
        )

        # Process LCM results and store them in .CSV format
        self.results_dir = os.path.join(self.paths['raw_mrs'], 'lcm')
        os.makedirs(self.results_dir, exist_ok=True)
        
        results_csv = combine_tables(self.results_dir)
        modify_and_save_csv(results_csv)
        self.common_data['results'] = pd.read_csv(
            os.path.join(self.results_dir, 'table_1h.csv'))
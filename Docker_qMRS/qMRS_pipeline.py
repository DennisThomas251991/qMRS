"""
MRS Quantitative Analysis Pipeline - Main Entry Point
=====================================================

This script serves as the command-line interface for processing Magnetic Resonance Spectroscopy (MRS) data.
It supports two different processing methodologies and handles the complete workflow from input validation
to final results generation.
"""

import argparse
import logging
from base_pipeline import MRSPipelineBase  # Common pipeline functionality
from STEAM_water_ref_with_qMRI import STEAMProcessor  # STEAM method processor
from MRSI_water_ref_no_qMRI import MRSIProcessor  # MRSI method processor

def main():
    """Main function handling command-line interface and pipeline execution"""
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Process MRS data using different quantification methods',
        epilog='Example: python qMRS_pipeline.py -i /data/input -m STEAM_water_ref_with_qMRI'
    )
    
    # Define required arguments
    parser.add_argument('-i', '--input', 
                      required=True,
                      help='Path to input directory containing MRS data and required files')
    parser.add_argument('-m', '--method',
                      required=True,
                      choices=['MRSI_water_ref_no_qMRI', 'STEAM_water_ref_with_qMRI'],
                      help='Processing methodology to use:\n'
                           '- MRSI_water_ref_no_qMRI: Without qMRI correction using literature water T1 and T2 values\n'
                           '- STEAM_water_ref_with_qMRI: With qMRI correction')

    # Parse command line arguments
    args = parser.parse_args()

    try:
        # Initialize the base pipeline with input parameters
        pipeline = MRSPipelineBase(args.input, args.method)
        
        # Validate input directory structure and required files
        pipeline.validate_inputs()
        
        # Execute common processing steps for both methods
        if not pipeline.execute_common_steps():
            raise RuntimeError("Common processing steps failed")

        # Select and initialize the appropriate processing method
        if args.method == 'STEAM_water_ref_with_qMRI':
            processor = STEAMProcessor(pipeline)  # STEAM method with qMRI correction
        else:
            processor = MRSIProcessor(pipeline)   # MRSI method without qMRI correction

        # Execute method-specific processing steps
        processor.process()

        # Final success message
        logging.info(f"Successfully completed {args.method} processing")

    except Exception as e:
        # Handle any exceptions during processing
        logging.error(f"Pipeline execution failed: {str(e)}")
        exit(1)  # Exit with error code

if __name__ == "__main__":
    """Entry point when executed as a standalone script"""
    main()
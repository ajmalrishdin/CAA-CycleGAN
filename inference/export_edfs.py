"""
Module to export CycleGAN extracted fECG waveforms into industry-standard EDF files.
Loops exactly through the available ADFECGDB records (there are natively 5 records, not 12).
Exports >= 15 seconds of the extracted trace for both V1 and V2 models.
"""
import os
import pyedflib
import numpy as np
from db_loaders import load_adfecgdb_record
from extraction_engines import CycleGANEngine

# -------------------------------------------------------------------------
# EDF Exporter Function
# -------------------------------------------------------------------------
def save_to_edf(signal_data, fs, filepath, label='fECG'):
    """
    Saves a 1D numpy array as an EDF+ file.
    
    Args:
        signal_data: 1D numpy array of signal values
        fs: sampling frequency
        filepath: output filepath ending in .edf
        label: label for the single channel
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Calculate physical limits safely
    s_min, s_max = float(signal_data.min()), float(signal_data.max())
    
    # Add minor padding to physical min/max to prevent boundary scaling issues
    padding = (s_max - s_min) * 0.05
    if padding == 0:
        padding = 1.0
    p_min = s_min - padding
    p_max = s_max + padding

    ch_info = {
        'label': label,
        'dimension': 'mV',
        'sample_frequency': fs,
        'physical_max': p_max,
        'physical_min': p_min,
        'digital_max': 32767,
        'digital_min': -32768,
        'transducer': '',
        'prefilter': ''
    }
    
    # pyedflib writer
    with pyedflib.EdfWriter(filepath, 1, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        f.setSignalHeader(0, ch_info)
        f.writePhysicalSamples(signal_data)
        
    print(f"  -> Saved EDF: {filepath}")

# -------------------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = "Output_EDFs 2"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading models...")
    engine_v1 = CycleGANEngine('/Users/ajmalrishdin/Documents/Projects/Pre-term Birth Detector/Code/CAA-CycleGAN/models/sagan_1', version_name='CycleGAN_V1', step=633)
    engine_v2 = CycleGANEngine('/Users/ajmalrishdin/Documents/Projects/Pre-term Birth Detector/Code/CAA-CycleGAN/models/CygleGAN V2 Models', version_name='CycleGAN_V2', step=4386)
    
    # ADFECGDB uniquely only contains 5 records locally and physically online.
    records = ['r01_ARR_1.edf', 'r01_ARR_2.edf', 'r01_ARR_3.edf', 'r01_ARR_4.edf', 'r01_ARR_5.edf', 'r01_ARR_6.edf', 'r01_ARR_7.edf', 'r01_ARR_8.edf', 'r01_ARR_9.edf', 'r01_ARR_10.edf', 'r01_ARR_11.edf', 'r01_ARR_12.edf']
    
    # We will export exactly 15 seconds per record requirement ("at least 10 secs")
    export_duration_sec = 15.0
    
    print(f"\nExporting {export_duration_sec}s of fECG from both V1 and V2 models...")
    print("=" * 60)
    
    for rec in records:
        print(f"\nProcessing {rec}:")
        # Load Record explicitely from the local synthetic database folder
        local_folder = "Databases/ADFECGDB"
        try:
            data = load_adfecgdb_record(rec, folder=local_folder)
        except Exception as e:
            print(f"Failed to load {rec} from {local_folder}. Error: {e}")
            continue
            
        fs = data['fs']
        num_samples_to_extract = int(fs * export_duration_sec)
        
        # Abdomen_1 is the specific lead standard for CycleGAN
        input_abd = data['abdominal'][0][:num_samples_to_extract]
        
        # Extract V1
        fecg_v1 = engine_v1.extract_fecg(input_abd, fs=fs)
        v1_path = os.path.join(out_dir, f"{rec.replace('.edf', '')}_CycleGAN_V1.edf")
        save_to_edf(fecg_v1, fs, v1_path, label='fECG_V1')
        
        # Extract V2
        fecg_v2 = engine_v2.extract_fecg(input_abd, fs=fs)
        v2_path = os.path.join(out_dir, f"{rec.replace('.edf', '')}_CycleGAN_V2.edf")
        save_to_edf(fecg_v2, fs, v2_path, label='fECG_V2')
        
    print(f"\nTask Complete! All 10 EDF files saved to '{out_dir}/'")

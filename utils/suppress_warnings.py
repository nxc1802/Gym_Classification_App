#!/usr/bin/env python3
"""
Module to suppress various warnings and logging
Import this FIRST to ensure warnings are suppressed before other imports
"""

import os
import sys
import warnings
import logging

def suppress_all_warnings():
    """Suppress all common warnings from ML libraries"""
    
    # Environment variables (must be set before imports)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow C++ warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization warnings
    os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')
    
    # Specific warning suppressions
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Suppress TensorFlow warnings (after import)
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        # Disable TensorFlow function retracing warnings
        tf.config.run_functions_eagerly(False)
        
    except ImportError:
        pass
    
    # Suppress MediaPipe and related warnings
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
    logging.getLogger('absl').setLevel(logging.ERROR)
    logging.getLogger('absl.logging').setLevel(logging.ERROR)
    
    # Suppress PyTorch warnings
    try:
        import torch
        torch.set_num_threads(1)  # Reduce threading messages
    except ImportError:
        pass
    
    # Suppress other common ML library warnings
    for logger_name in [
        'h5py', 'PIL', 'matplotlib', 'sklearn', 
        'scipy', 'numpy', 'pandas', 'cv2'
    ]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

# Call immediately when module is imported
suppress_all_warnings() 
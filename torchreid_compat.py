"""
TorchReID Compatibility Module

This module provides compatibility for the old torchreid.utils import path
that some libraries like deep-sort-realtime still expect.
"""

import sys
import torchreid

# Create a compatibility module for torchreid.utils
class TorchReIDUtilsCompat:
    """Compatibility wrapper for torchreid.utils"""
    
    def __init__(self):
        # Import the actual FeatureExtractor from the correct location
        try:
            from torchreid.reid.utils import FeatureExtractor
            self.FeatureExtractor = FeatureExtractor
        except ImportError:
            try:
                from torchreid.utils.feature_extractor import FeatureExtractor
                self.FeatureExtractor = FeatureExtractor
            except ImportError:
                from torchreid.utils import FeatureExtractor
                self.FeatureExtractor = FeatureExtractor

# Create the compatibility module
torchreid_utils_compat = TorchReIDUtilsCompat()

# Add it to sys.modules so other libraries can import it
sys.modules['torchreid.utils'] = torchreid_utils_compat

# Also make the FeatureExtractor available directly
FeatureExtractor = torchreid_utils_compat.FeatureExtractor

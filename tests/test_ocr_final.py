import os
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
import sys
sys.path.insert(0, 'services')
sys.path.insert(0, 'services/hvac-ai')
from text_extractor_service import TextExtractor
print('Testing TextExtractor initialization...')
t = TextExtractor()
print('✅ TextExtractor initialized successfully')

# Test extraction
import numpy as np
test_image = np.ones((50, 50, 3), dtype=np.uint8) * 255  # White image
result = t.extract_single_text(test_image)
print(f'Test extraction result: {result}')
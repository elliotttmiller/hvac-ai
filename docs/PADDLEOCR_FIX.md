# PaddleOCR Fix - Image Format Issue Resolution

## Problem
The TextExtractorDeployment was crashing with the following errors:
1. `TypeError: PaddleOCR.predict() got an unexpected keyword argument 'cls'`
2. `IndexError: tuple index out of range` when accessing `img.shape[2]` (grayscale image)

The root cause was that preprocessed images were being converted to grayscale (2D arrays), but PaddleOCR expects 3-channel color images (3D arrays). This caused crashes deep in the PaddleX pipeline during image normalization.

## Solution

### 1. Image Format Validation in TextExtractor (`text_extractor_service.py`)
Added a new static method `_ensure_rgb_image()` that:
- Detects grayscale images (2D arrays with shape H×W)
- Converts them to 3-channel BGR format using `cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)`
- Passes 3-channel images through unchanged
- Is called before OCR processing in `extract_text()`

### 2. Preprocessing Pipeline Update (`utils/geometry.py`)
Modified `preprocess_for_ocr()` to:
- **Disable grayscale conversion by default** (`apply_threshold=False` now)
- Preserve color format when preprocessing images
- Convert back to BGR format if intermediate processing steps produce grayscale
- Added detailed documentation explaining why color format is maintained

### 3. OCR Pipeline Impact
The updated pipeline now ensures:
- Images extracted from OBB regions maintain color format when sent to OCR
- Fallback conversion to 3-channel format in `TextExtractor.extract_text()` as a safety net
- Compatibility with modern PaddleOCR versions that expect color images

## Files Modified
1. `services/hvac-ai/text_extractor_service.py` - Added `_ensure_rgb_image()` method
2. `services/hvac-ai/utils/geometry.py` - Updated `preprocess_for_ocr()` behavior

## Testing
To verify the fix:
1. Restart the AI-ENGINE with `python scripts/start_unified.py`
2. Submit an HVAC document analysis request
3. Verify that text extraction completes without actor crashes
4. Check logs for successful text extraction messages

## Future Improvements
- Consider adding image quality enhancement (CLAHE, denoising) without changing format
- Add telemetry to track image format conversions
- Implement more robust fallback handling for malformed images

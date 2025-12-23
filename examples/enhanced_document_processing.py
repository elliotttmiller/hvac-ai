"""
Example: Enhanced Document Processing Pipeline

Demonstrates the complete enhanced document processing workflow
integrating research from AI document processing papers.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_services.core.document.enhanced_processor import create_enhanced_processor
from python_services.core.document.hybrid_processor import create_hybrid_processor


def example_1_basic_processing():
    """Example 1: Basic enhanced document processing"""
    print("\n" + "="*60)
    print("Example 1: Basic Enhanced Document Processing")
    print("="*60)
    
    # Create processor with caching enabled
    processor = create_enhanced_processor(use_cache=True)
    
    # Load sample blueprint (you would use a real blueprint here)
    # For demo, create a sample image
    image = create_sample_blueprint()
    
    # Process the blueprint
    print("\nProcessing blueprint...")
    results = processor.process(image)
    
    # Display results
    print(f"\n✓ Processing complete!")
    print(f"  Quality Score: {results['quality_info']['quality_score']:.2f}")
    print(f"  Estimated DPI: {results['quality_info']['estimated_dpi']}")
    print(f"  Enhanced: {results['metadata']['enhanced']}")
    print(f"  Regions Found: {results['metadata']['region_count']}")
    print(f"  Text Blocks: {results['metadata']['text_region_count']}")
    
    # Show region details
    print("\nDetected Regions:")
    for i, region in enumerate(results['regions'], 1):
        print(f"  {i}. {region['type']} - Confidence: {region['confidence']:.2f}")
    
    return results


def example_2_hybrid_ocr_vlm():
    """Example 2: Hybrid OCR + VLM processing"""
    print("\n" + "="*60)
    print("Example 2: Hybrid OCR + VLM Processing")
    print("="*60)
    
    # Create hybrid processor
    processor = create_hybrid_processor(
        ocr_engine="easyocr",
        vlm_model="qwen2-vl",
        confidence_threshold=0.6
    )
    
    # Load sample blueprint
    image = create_sample_blueprint()
    
    # Process with hybrid approach
    print("\nProcessing with hybrid OCR + VLM...")
    results = processor.process(image)
    
    # Display results
    print(f"\n✓ Processing complete!")
    print(f"  OCR Results: {results['metadata']['ocr_count']}")
    print(f"  VLM Entities: {results['metadata']['vlm_entities']}")
    print(f"  Validated: {results['metadata']['validated_count']}")
    print(f"  VLM Confidence: {results['metadata']['vlm_confidence']:.2f}")
    
    # Show validated text
    print("\nValidated Text (top 5):")
    for i, result in enumerate(results['results'][:5], 1):
        print(f"  {i}. '{result.text}'")
        print(f"     Confidence: {result.confidence:.2f}")
        print(f"     Validated: {result.validated}")
        print(f"     Sources: {', '.join(result.sources)}")
    
    return results


def example_3_quality_assessment():
    """Example 3: Image quality assessment and enhancement"""
    print("\n" + "="*60)
    print("Example 3: Quality Assessment and Enhancement")
    print("="*60)
    
    # Create test images with different qualities
    images = {
        'high_quality': create_sample_blueprint(quality='high'),
        'low_quality': create_sample_blueprint(quality='low'),
        'blurry': create_sample_blueprint(blur=True),
        'low_contrast': create_sample_blueprint(contrast='low')
    }
    
    processor = create_enhanced_processor(use_cache=False)
    
    print("\nAssessing different image qualities...")
    for name, image in images.items():
        results = processor.process(image)
        quality = results['quality_info']
        
        print(f"\n{name.replace('_', ' ').title()}:")
        print(f"  Quality Score: {quality['quality_score']:.2f}")
        print(f"  Blur Score: {quality['blur_score']:.1f}")
        print(f"  Contrast: {quality['contrast']:.1f}")
        print(f"  Is Blurry: {quality['is_blurry']}")
        print(f"  Is Low Contrast: {quality['is_low_contrast']}")
        print(f"  Needs Enhancement: {quality['needs_enhancement']}")
        print(f"  Enhanced: {results['metadata']['enhanced']}")


def example_4_region_specific_processing():
    """Example 4: Region-specific processing"""
    print("\n" + "="*60)
    print("Example 4: Region-Specific Processing")
    print("="*60)
    
    # Create processors
    doc_processor = create_enhanced_processor()
    hybrid_processor = create_hybrid_processor()
    
    # Load blueprint
    image = create_sample_blueprint()
    
    # First, segment the document
    print("\nSegmenting document...")
    doc_results = doc_processor.process(image)
    
    print(f"Found {len(doc_results['regions'])} regions")
    
    # Process each region with hybrid approach
    print("\nProcessing regions with hybrid OCR + VLM...")
    region_results = hybrid_processor.process_with_regions(
        image,
        doc_results['regions']
    )
    
    # Display results by region
    print(f"\n✓ Processed {region_results['total_regions']} regions")
    
    for i, region in enumerate(region_results['regions'], 1):
        print(f"\nRegion {i} - {region['region_type']}:")
        print(f"  Text elements: {len(region['results'])}")
        print(f"  OCR count: {region['metadata']['ocr_count']}")
        print(f"  VLM entities: {region['metadata']['vlm_entities']}")


def example_5_semantic_caching():
    """Example 5: Semantic caching performance"""
    print("\n" + "="*60)
    print("Example 5: Semantic Caching Performance")
    print("="*60)
    
    import time
    
    # Create processor with caching
    processor = create_enhanced_processor(use_cache=True)
    
    # Create test image
    image = create_sample_blueprint()
    
    # First processing (no cache)
    print("\nFirst processing (no cache)...")
    start = time.time()
    results1 = processor.process(image)
    time1 = time.time() - start
    print(f"  Time: {time1:.3f}s")
    
    # Second processing (cache hit)
    print("\nSecond processing (cache hit)...")
    start = time.time()
    results2 = processor.process(image)
    time2 = time.time() - start
    print(f"  Time: {time2:.3f}s")
    
    # Calculate speedup
    speedup = (time1 - time2) / time1 * 100
    print(f"\n✓ Cache speedup: {speedup:.1f}%")
    
    # Verify results are identical
    assert results1['metadata']['region_count'] == results2['metadata']['region_count']
    print("✓ Results verified identical")


def example_6_complete_pipeline():
    """Example 6: Complete analysis pipeline"""
    print("\n" + "="*60)
    print("Example 6: Complete Analysis Pipeline")
    print("="*60)
    
    # Initialize all processors
    doc_processor = create_enhanced_processor()
    text_processor = create_hybrid_processor()
    
    # Load blueprint
    image = create_sample_blueprint()
    
    # Step 1: Document analysis
    print("\nStep 1: Document Analysis")
    doc_results = doc_processor.process(image)
    print(f"  ✓ Quality: {doc_results['quality_info']['quality_score']:.2f}")
    print(f"  ✓ Regions: {doc_results['metadata']['region_count']}")
    
    # Step 2: Text extraction
    print("\nStep 2: Text Extraction (Hybrid OCR + VLM)")
    text_results = text_processor.process(image)
    print(f"  ✓ Text elements: {len(text_results['results'])}")
    print(f"  ✓ Validated: {text_results['metadata']['validated_count']}")
    
    # Step 3: Combine results
    print("\nStep 3: Combining Results")
    complete_analysis = {
        'document': {
            'quality_score': doc_results['quality_info']['quality_score'],
            'regions': doc_results['regions'],
            'enhanced': doc_results['metadata']['enhanced']
        },
        'text': {
            'elements': [
                {
                    'text': r.text,
                    'confidence': r.confidence,
                    'validated': r.validated,
                    'bbox': r.bbox
                }
                for r in text_results['results']
            ],
            'context': text_results['context']
        },
        'metadata': {
            'quality': doc_results['quality_info']['quality_score'],
            'text_elements': len(text_results['results']),
            'regions': doc_results['metadata']['region_count']
        }
    }
    
    # Save results
    output_path = Path(__file__).parent / "output" / "complete_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Convert to JSON-serializable format
        json_output = {
            'document': complete_analysis['document'],
            'text': complete_analysis['text'],
            'metadata': complete_analysis['metadata']
        }
        json.dump(json_output, f, indent=2)
    
    print(f"\n✓ Complete analysis saved to: {output_path}")
    
    return complete_analysis


def create_sample_blueprint(quality='medium', blur=False, contrast='normal'):
    """
    Create a sample blueprint image for demonstration
    
    Args:
        quality: 'high', 'medium', or 'low'
        blur: Apply blur
        contrast: 'high', 'normal', or 'low'
    
    Returns:
        Sample blueprint image
    """
    # Create a blank white image
    if quality == 'high':
        size = (2400, 1800)
    elif quality == 'low':
        size = (800, 600)
    else:
        size = (1600, 1200)
    
    image = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Draw some blueprint-like elements
    # Title block (bottom-right)
    cv2.rectangle(image, 
                  (size[0]-400, size[1]-200), 
                  (size[0]-50, size[1]-50),
                  (0, 0, 0), 2)
    cv2.putText(image, "TITLE BLOCK", 
                (size[0]-350, size[1]-150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Main drawing area
    cv2.rectangle(image,
                  (100, 100),
                  (size[0]-500, size[1]-300),
                  (0, 0, 0), 2)
    cv2.putText(image, "MAIN DRAWING AREA",
                (200, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Schedule (right side)
    cv2.rectangle(image,
                  (size[0]-450, 100),
                  (size[0]-50, 400),
                  (0, 0, 0), 2)
    cv2.putText(image, "SCHEDULE",
                (size[0]-400, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Notes
    cv2.putText(image, "NOTE: Sample blueprint",
                (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Apply quality adjustments
    if blur:
        image = cv2.GaussianBlur(image, (15, 15), 0)
    
    if contrast == 'low':
        # Reduce contrast
        image = cv2.addWeighted(image, 0.5, np.ones_like(image) * 128, 0.5, 0)
    elif contrast == 'high':
        # Increase contrast
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=-50)
    
    return image


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Enhanced Document Processing Examples")
    print("Based on AI Document Processing Research")
    print("="*60)
    
    try:
        # Run examples
        example_1_basic_processing()
        example_2_hybrid_ocr_vlm()
        example_3_quality_assessment()
        example_4_region_specific_processing()
        example_5_semantic_caching()
        example_6_complete_pipeline()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

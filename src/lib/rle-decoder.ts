/**
 * RLE (Run-Length Encoding) Decoder for SAM Masks
 * 
 * Decodes masks from the backend SAM inference engine that uses
 * standard COCO RLE format (pycocotools).
 */

/**
 * RLE mask format from backend
 */
interface RLEMask {
    size: [number, number];  // [height, width]
    counts: string;          // RLE counts string
}

/**
 * Decode COCO RLE mask to binary mask array
 * 
 * The backend sends masks in standard COCO RLE format as a JSON object:
 * { "size": [height, width], "counts": "..." }
 * where counts is the COCO RLE format (run-length encoded in Fortran order)
 * 
 * @param rle - RLE mask object from API
 * @returns Binary mask as 2D array where 1 = mask pixel, 0 = background
 */
export function decodeRLEMask(rle: RLEMask): number[][] {
    try {
        // Extract size and counts from the RLE object
        const [height, width] = rle.size;
        const countsStr = rle.counts;
        
        if (!height || !width || !countsStr) {
            throw new Error('Invalid RLE format');
        }
        
        // Decode the RLE counts
        const counts = decodeRLECounts(countsStr);
        
        // Convert RLE to binary mask (Fortran order - column-major)
        const mask = rleToBinaryMask(counts, height, width);
        
        return mask;
    } catch (error) {
        console.error('Failed to decode RLE mask:', error);
        // Return empty mask on error
        return [];
    }
}

/**
 * Decode COCO RLE counts string to array of run lengths
 * 
 * COCO uses a compressed format where counts can be stored as
 * ASCII characters or multi-byte sequences.
 * 
 * @param countsStr - RLE counts string
 * @returns Array of run lengths
 */
function decodeRLECounts(countsStr: string): number[] {
    const counts: number[] = [];
    let i = 0;
    
    while (i < countsStr.length) {
        let count = 0;
        let k = 0;
        let more = true;
        
        // Decode variable-length integer (COCO format)
        while (more) {
            const c = countsStr.charCodeAt(i++);
            const val = c & 0x1f; // Get lower 5 bits
            count |= val << (5 * k);
            more = (c & 0x20) !== 0; // Check if more bytes follow
            k++;
            
            if (i > countsStr.length) {
                break;
            }
        }
        
        counts.push(count);
    }
    
    return counts;
}

/**
 * Convert RLE counts to binary mask in Fortran (column-major) order
 * 
 * @param counts - Array of run lengths
 * @param height - Mask height
 * @param width - Mask width
 * @returns Binary mask as 2D array
 */
function rleToBinaryMask(counts: number[], height: number, width: number): number[][] {
    // Initialize mask with zeros
    const mask: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    
    let pixelIndex = 0;
    let currentValue = 0; // Start with background (0)
    
    // Process runs in Fortran order (column by column)
    for (const count of counts) {
        for (let i = 0; i < count; i++) {
            if (pixelIndex >= height * width) {
                break;
            }
            
            // Convert linear index to (row, col) in Fortran order
            const col = Math.floor(pixelIndex / height);
            const row = pixelIndex % height;
            
            if (row < height && col < width) {
                mask[row][col] = currentValue;
            }
            
            pixelIndex++;
        }
        
        // Alternate between background (0) and foreground (1)
        currentValue = 1 - currentValue;
    }
    
    return mask;
}

/**
 * Draw binary mask on canvas with specified color and opacity
 * 
 * @param ctx - Canvas 2D context
 * @param mask - Binary mask (2D array)
 * @param color - RGB color as [r, g, b] (0-255)
 * @param opacity - Opacity (0-1)
 */
export function drawMaskOnCanvas(
    ctx: CanvasRenderingContext2D,
    mask: number[][],
    color: [number, number, number] = [0, 255, 0],
    opacity: number = 0.4
): void {
    if (!mask.length || !mask[0]) {
        return;
    }
    
    const height = mask.length;
    const width = mask[0].length;
    
    // Create ImageData for the mask
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;
    
    // Fill ImageData with mask pixels
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            
            if (mask[y][x] === 1) {
                data[i] = color[0];     // R
                data[i + 1] = color[1]; // G
                data[i + 2] = color[2]; // B
                data[i + 3] = Math.floor(opacity * 255); // A
            } else {
                data[i + 3] = 0; // Transparent for background
            }
        }
    }
    
    // Draw the mask
    ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw mask with boundary outline
 * 
 * @param ctx - Canvas 2D context
 * @param mask - Binary mask (2D array)
 * @param fillColor - Fill color as [r, g, b]
 * @param fillOpacity - Fill opacity (0-1)
 * @param strokeColor - Stroke color as CSS color string
 * @param strokeWidth - Stroke width in pixels
 */
export function drawMaskWithBoundary(
    ctx: CanvasRenderingContext2D,
    mask: number[][],
    fillColor: [number, number, number] = [0, 255, 0],
    fillOpacity: number = 0.3,
    strokeColor: string = 'rgba(0, 255, 0, 1)',
    strokeWidth: number = 2
): void {
    if (!mask.length || !mask[0]) {
        return;
    }
    
    // Draw filled mask
    drawMaskOnCanvas(ctx, mask, fillColor, fillOpacity);
    
    // Draw boundary
    const height = mask.length;
    const width = mask[0].length;
    
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = strokeWidth;
    ctx.beginPath();
    
    // Find and draw boundaries (edge detection)
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (mask[y][x] === 1) {
                // Check if this is a boundary pixel
                const isTop = y === 0 || mask[y - 1][x] === 0;
                const isBottom = y === height - 1 || mask[y + 1][x] === 0;
                const isLeft = x === 0 || mask[y][x - 1] === 0;
                const isRight = x === width - 1 || mask[y][x + 1] === 0;
                
                if (isTop || isBottom || isLeft || isRight) {
                    ctx.fillStyle = strokeColor;
                    ctx.fillRect(x, y, 1, 1);
                }
            }
        }
    }
    
    ctx.stroke();
}

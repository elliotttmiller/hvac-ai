import cv from '@techstark/opencv-js';
import Tesseract from 'tesseract.js';

export interface BlueprintFeatures {
  // Geometric Analysis
  lines: {
    walls: Line[];
    doors: Line[];
    windows: Line[];
    dimensions: Line[];
    centerlines: Line[];
  };

  // Textual Information
  text: {
    roomLabels: TextRegion[];
    dimensions: TextRegion[];
    notes: TextRegion[];
    titleBlock: TextRegion[];
  };

  // Architectural Elements
  elements: {
    rooms: Room[];
    doors: Door[];
    windows: Window[];
    stairs: Stair[];
    fixtures: Fixture[];
  };

  // Scale and Measurements
  scale: {
    detected: boolean;
    ratio: number; // pixels per unit (mm, ft, etc.)
    unit: 'mm' | 'cm' | 'm' | 'in' | 'ft';
    confidence: number;
  };

  // Drawing Standards
  standard: {
    type: 'architectural' | 'structural' | 'mep' | 'site' | 'unknown';
    projection: 'plan' | 'elevation' | 'section' | 'detail' | 'isometric';
    level: 'floor_plan' | 'ceiling_plan' | 'roof_plan' | 'basement';
    confidence: number;
  };
}

interface Line {
  start: Point;
  end: Point;
  thickness: number;
  style: 'solid' | 'dashed' | 'dotted' | 'centerline';
  confidence: number;
}

interface Point {
  x: number;
  y: number;
}

interface TextRegion {
  text: string;
  bbox: BoundingBox;
  confidence: number;
  fontSize: number;
  rotation: number;
}

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Room {
  label: string;
  area: number;
  perimeter: Point[];
  centroid: Point;
  purpose: 'bedroom' | 'bathroom' | 'kitchen' | 'living' | 'office' | 'storage' | 'utility' | 'unknown';
}

interface Door {
  type: 'single' | 'double' | 'sliding' | 'folding' | 'revolving';
  width: number;
  swing: 'left' | 'right' | 'both' | 'sliding';
  position: Point;
  wall: Line;
}

interface Window {
  type: 'single' | 'double' | 'casement' | 'awning' | 'sliding' | 'bay';
  width: number;
  height: number;
  position: Point;
  wall: Line;
}

interface Stair {
  type: 'straight' | 'l_shaped' | 'u_shaped' | 'spiral' | 'curved';
  steps: number;
  direction: 'up' | 'down';
  bbox: BoundingBox;
}

interface Fixture {
  type: 'toilet' | 'sink' | 'bathtub' | 'shower' | 'kitchen_sink' | 'dishwasher' | 'refrigerator' | 'stove';
  position: Point;
  rotation: number;
}

export class BlueprintAnalyzer {
  private static instance: BlueprintAnalyzer;

  public static getInstance(): BlueprintAnalyzer {
    if (!BlueprintAnalyzer.instance) {
      BlueprintAnalyzer.instance = new BlueprintAnalyzer();
    }
    return BlueprintAnalyzer.instance;
  }

  /**
   * Enhanced blueprint analysis with computer vision
   */
  public async analyzeBlueprint(
    file: File,
    options: {
      enableOCR?: boolean;
      enhanceImage?: boolean;
      detectScale?: boolean;
      classifyElements?: boolean;
    } = {}
  ): Promise<BlueprintFeatures> {

    console.log('🔍 Starting enhanced blueprint analysis...');

    try {
      // Convert file to image data
      const imageData = await this.loadImage(file);

      // Preprocess image for better recognition
      const preprocessed = options.enhanceImage ?
        await this.preprocessBlueprint(imageData) : imageData;

      // Parallel analysis tasks
      const [
        lineAnalysis,
        textAnalysis,
        scaleAnalysis,
        standardAnalysis
      ] = await Promise.all([
        this.analyzeLines(preprocessed),
        options.enableOCR ? this.analyzeText(preprocessed) : this.getEmptyTextAnalysis(),
        options.detectScale ? this.detectScale(preprocessed) : this.getDefaultScale(),
        this.classifyDrawingStandard(preprocessed)
      ]);

      // Extract architectural elements from lines and text
      const elements = options.classifyElements ?
        await this.extractArchitecturalElements(lineAnalysis, textAnalysis, scaleAnalysis) :
        this.getEmptyElements();

      const features: BlueprintFeatures = {
        lines: lineAnalysis,
        text: textAnalysis,
        elements,
        scale: scaleAnalysis,
        standard: standardAnalysis
      };

      console.log('✅ Blueprint analysis complete:', {
        linesDetected: Object.values(lineAnalysis).flat().length,
        textRegions: Object.values(textAnalysis).flat().length,
        elementsFound: Object.values(elements).flat().length,
        scaleDetected: scaleAnalysis.detected,
        drawingType: standardAnalysis.type
      });

      return features;

    } catch (error) {
      console.error('❌ Blueprint analysis failed:', error);
      return this.getFallbackAnalysis(file);
    }
  }

  /**
   * Preprocess blueprint image for better recognition
   */
  private async preprocessBlueprint(imageData: ImageData): Promise<ImageData> {
    // Convert to OpenCV Mat
    const src = cv.matFromImageData(imageData);
    const processed = new cv.Mat();

    try {
      // Convert to grayscale
      cv.cvtColor(src, processed, cv.COLOR_RGBA2GRAY);

      // Enhance contrast and reduce noise
      const enhanced = new cv.Mat();
      cv.equalizeHist(processed, enhanced);

      // Apply morphological operations to clean up lines
      const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
      const morphed = new cv.Mat();
      cv.morphologyEx(enhanced, morphed, cv.MORPH_CLOSE, kernel);

      // Convert back to ImageData
      const result = new ImageData(
        new Uint8ClampedArray(morphed.data),
        morphed.cols,
        morphed.rows
      );

      // Cleanup
      src.delete();
      processed.delete();
      enhanced.delete();
      morphed.delete();
      kernel.delete();

      return result;

    } catch (error) {
      console.error('Image preprocessing failed:', error);
      src.delete();
      processed.delete();
      return imageData;
    }
  }

  /**
   * Analyze lines in the blueprint using Hough transform and edge detection
   */
  private async analyzeLines(imageData: ImageData): Promise<BlueprintFeatures['lines']> {
    const src = cv.matFromImageData(imageData);
    const lines = {
      walls: [] as Line[],
      doors: [] as Line[],
      windows: [] as Line[],
      dimensions: [] as Line[],
      centerlines: [] as Line[]
    };

    try {
      // Edge detection
      const edges = new cv.Mat();
      cv.Canny(src, edges, 50, 150);

      // Hough line detection
      const detectedLines = new cv.Mat();
      cv.HoughLinesP(edges, detectedLines, 1, Math.PI / 180, 50, 30, 10);

      // Classify lines based on characteristics
      for (let i = 0; i < detectedLines.rows; i++) {
        const line = this.extractLineFromMat(detectedLines, i);
        const classification = this.classifyLine(line, imageData);

        lines[classification].push(line);
      }

      // Cleanup
      src.delete();
      edges.delete();
      detectedLines.delete();

    } catch (error) {
      console.error('Line analysis failed:', error);
      src.delete();
    }

    return lines;
  }

  /**
   * Extract and analyze text using Tesseract OCR
   */
  private async analyzeText(imageData: ImageData): Promise<BlueprintFeatures['text']> {
    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      canvas.width = imageData.width;
      canvas.height = imageData.height;
      ctx.putImageData(imageData, 0, 0);

      // OCR with basic configuration
      const { data } = await Tesseract.recognize(canvas, 'eng', {
        logger: m => console.log('OCR:', m)
      });

      // Classify text regions
      const text = {
        roomLabels: [] as TextRegion[],
        dimensions: [] as TextRegion[],
        notes: [] as TextRegion[],
        titleBlock: [] as TextRegion[]
      };

      // Process OCR results safely with type assertion
      const ocrData = data as any;
      if (ocrData.words && Array.isArray(ocrData.words)) {
        ocrData.words.forEach((word: any) => {
          if (word.confidence && word.confidence > 60) {
            const region: TextRegion = {
              text: word.text || '',
              bbox: word.bbox || { x: 0, y: 0, width: 0, height: 0 },
              confidence: word.confidence || 0,
              fontSize: word.bbox ? (word.bbox.y1 - word.bbox.y0) : 12,
              rotation: 0
            };

            const category = this.classifyTextRegion(region);
            text[category].push(region);
          }
        });
      } else {
        // Fallback: process just the main text
        if (data.text && data.text.length > 0) {
          const lines = data.text.split('\n').filter(line => line.trim().length > 0);
          lines.forEach((line, index) => {
            const region: TextRegion = {
              text: line.trim(),
              bbox: { x: 0, y: index * 20, width: 100, height: 20 },
              confidence: 70,
              fontSize: 12,
              rotation: 0
            };

            const category = this.classifyTextRegion(region);
            text[category].push(region);
          });
        }
      }

      return text;

    } catch (error) {
      console.error('Text analysis failed:', error);
      return this.getEmptyTextAnalysis();
    }
  }

  /**
   * Detect scale information from the blueprint
   */
  private async detectScale(imageData: ImageData): Promise<BlueprintFeatures['scale']> {
    // Look for scale bars, dimension annotations, and standard scale ratios
    // This is a simplified implementation - would need more sophisticated pattern matching

    return {
      detected: false,
      ratio: 50, // Default: 50 pixels per meter
      unit: 'm',
      confidence: 0.3
    };
  }

  /**
   * Classify the type of architectural drawing
   */
  private async classifyDrawingStandard(imageData: ImageData): Promise<BlueprintFeatures['standard']> {
    // Analyze drawing characteristics to determine type
    // Look for typical architectural symbols, title blocks, etc.

    const aspectRatio = imageData.width / imageData.height;

    // Basic classification based on aspect ratio and content
    let type: BlueprintFeatures['standard']['type'] = 'architectural';
    let projection: BlueprintFeatures['standard']['projection'] = 'plan';

    if (aspectRatio > 1.5) {
      projection = 'elevation';
    }

    return {
      type,
      projection,
      level: 'floor_plan',
      confidence: 0.7
    };
  }

  /**
   * Extract architectural elements from analyzed lines and text
   */
  private async extractArchitecturalElements(
    lines: BlueprintFeatures['lines'],
    text: BlueprintFeatures['text'],
    scale: BlueprintFeatures['scale']
  ): Promise<BlueprintFeatures['elements']> {

    const elements = {
      rooms: [] as Room[],
      doors: [] as Door[],
      windows: [] as Window[],
      stairs: [] as Stair[],
      fixtures: [] as Fixture[]
    };

    // Detect rooms from enclosed areas
    elements.rooms = this.detectRooms(lines.walls, text.roomLabels);

    // Detect doors from wall openings and door symbols
    elements.doors = this.detectDoors(lines.walls, lines.doors);

    // Detect windows from wall openings and window symbols
    elements.windows = this.detectWindows(lines.walls, lines.windows);

    // Detect stairs from stair symbols and patterns
    elements.stairs = this.detectStairs(lines.walls);

    // Detect fixtures from symbols
    elements.fixtures = this.detectFixtures(lines.walls);

    return elements;
  }

  // Helper methods for element detection
  private detectRooms(walls: Line[], labels: TextRegion[]): Room[] {
    // Simplified room detection - would need more sophisticated polygon analysis
    return labels.map(label => ({
      label: label.text,
      area: 20, // Calculate from enclosed area
      perimeter: [], // Calculate from walls
      centroid: { x: label.bbox.x, y: label.bbox.y },
      purpose: this.classifyRoomPurpose(label.text)
    }));
  }

  private detectDoors(walls: Line[], doorLines: Line[]): Door[] {
    return doorLines.map(line => ({
      type: 'single' as const,
      width: this.calculateDistance(line.start, line.end),
      swing: 'right' as const,
      position: line.start,
      wall: walls[0] || line
    }));
  }

  private detectWindows(walls: Line[], windowLines: Line[]): Window[] {
    return windowLines.map(line => ({
      type: 'single' as const,
      width: this.calculateDistance(line.start, line.end),
      height: 1.2, // Standard window height
      position: line.start,
      wall: walls[0] || line
    }));
  }

  private detectStairs(walls: Line[]): Stair[] {
    // Pattern recognition for stair symbols
    return [];
  }

  private detectFixtures(walls: Line[]): Fixture[] {
    // Symbol recognition for bathroom and kitchen fixtures
    return [];
  }

  // Utility methods
  private async loadImage(file: File): Promise<ImageData> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d')!;
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        resolve(ctx.getImageData(0, 0, img.width, img.height));
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  private extractLineFromMat(mat: any, index: number): Line {
    // Extract line coordinates from OpenCV Mat
    const data = mat.data32S;
    const i = index * 4;
    return {
      start: { x: data[i], y: data[i + 1] },
      end: { x: data[i + 2], y: data[i + 3] },
      thickness: 1,
      style: 'solid',
      confidence: 0.8
    };
  }

  private classifyLine(line: Line, imageData: ImageData): keyof BlueprintFeatures['lines'] {
    const length = this.calculateDistance(line.start, line.end);

    // Classify based on length, position, and surrounding context
    if (length > 100) return 'walls';
    if (length > 50) return 'doors';
    if (length > 20) return 'windows';
    return 'dimensions';
  }

  private classifyTextRegion(region: TextRegion): keyof BlueprintFeatures['text'] {
    const text = region.text.toLowerCase();

    if (/\d+['"]/.test(text) || /\d+\s*(mm|cm|m|ft|in)/.test(text)) {
      return 'dimensions';
    }
    if (/^(bed|bath|kitchen|living|office|storage)/.test(text)) {
      return 'roomLabels';
    }
    if (region.bbox.y < 100) {
      return 'titleBlock';
    }
    return 'notes';
  }

  private classifyRoomPurpose(label: string): Room['purpose'] {
    const text = label.toLowerCase();
    if (text.includes('bed')) return 'bedroom';
    if (text.includes('bath')) return 'bathroom';
    if (text.includes('kitchen')) return 'kitchen';
    if (text.includes('living')) return 'living';
    if (text.includes('office')) return 'office';
    return 'unknown';
  }

  private calculateDistance(p1: Point, p2: Point): number {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
  }

  // Fallback methods
  private getFallbackAnalysis(file: File): BlueprintFeatures {
    return {
      lines: this.getEmptyLines(),
      text: this.getEmptyTextAnalysis(),
      elements: this.getEmptyElements(),
      scale: this.getDefaultScale(),
      standard: {
        type: 'architectural',
        projection: 'plan',
        level: 'floor_plan',
        confidence: 0.1
      }
    };
  }

  private getEmptyLines(): BlueprintFeatures['lines'] {
    return {
      walls: [],
      doors: [],
      windows: [],
      dimensions: [],
      centerlines: []
    };
  }

  private getEmptyTextAnalysis(): BlueprintFeatures['text'] {
    return {
      roomLabels: [],
      dimensions: [],
      notes: [],
      titleBlock: []
    };
  }

  private getEmptyElements(): BlueprintFeatures['elements'] {
    return {
      rooms: [],
      doors: [],
      windows: [],
      stairs: [],
      fixtures: []
    };
  }

  private getDefaultScale(): BlueprintFeatures['scale'] {
    return {
      detected: false,
      ratio: 50,
      unit: 'm',
      confidence: 0
    };
  }
}

export const blueprintAnalyzer = BlueprintAnalyzer.getInstance();

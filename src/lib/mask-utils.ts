/**
 * Mask drawing utilities (non-RLE specific)
 *
 * Contains helpers used by the frontend to draw binary masks onto canvases.
 */
export function drawMaskOnCanvas(
  ctx: CanvasRenderingContext2D,
  mask: number[][],
  color: [number, number, number] = [0, 255, 0],
  opacity: number = 0.4
): void {
  if (!mask.length || !mask[0]) return;
  const height = mask.length;
  const width = mask[0].length;
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      if (mask[y][x] === 1) {
        data[i] = color[0];
        data[i + 1] = color[1];
        data[i + 2] = color[2];
        data[i + 3] = Math.floor(opacity * 255);
      } else {
        data[i + 3] = 0;
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

export function drawMaskWithBoundary(
  ctx: CanvasRenderingContext2D,
  mask: number[][],
  fillColor: [number, number, number] = [0, 255, 0],
  fillOpacity: number = 0.3,
  strokeColor: string = 'rgba(0, 255, 0, 1)',
  strokeWidth: number = 2
): void {
  if (!mask.length || !mask[0]) return;
  drawMaskOnCanvas(ctx, mask, fillColor, fillOpacity);
  const height = mask.length;
  const width = mask[0].length;
  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = strokeWidth;
  ctx.beginPath();
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y][x] === 1) {
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

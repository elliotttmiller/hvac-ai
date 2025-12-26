const labelColorCache = new Map<string, string>();

function djb2Hash(str: string) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) + str.charCodeAt(i);
    hash = hash | 0;
  }
  return Math.abs(hash);
}

export function getColorForLabel(label: string) {
  const key = String(label || '');
  const cached = labelColorCache.get(key);
  if (cached) return cached;

  const h = djb2Hash(key + '::unique');
  const hue = h % 360;
  const sat = 68;
  const light = 52;
  const color = `hsl(${hue}, ${sat}%, ${light}%)`;
  labelColorCache.set(key, color);
  return color;
}

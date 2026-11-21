"use strict";

const Stencil = (() => {
  const clamp = (value, end) => Math.max(0, Math.min(end, value));
  const luminance = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const matrix = (width, height) =>
    new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);

  // Apply a 5×5 Laplacian to the inverted image using a separable box sum.
  function edgeStrength(pixels, width, height, stride, channel, workspace) {
    const strength = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      const row = y * width;
      for (let x = 0; x < width; x++) {
        let sum = 0;
        for (let offset = -2; offset <= 2; offset++) {
          const index = row + clamp(x + offset, width - 1);
          sum += pixels[index * stride + channel];
        }
        workspace[row + x] = sum;
      }
    }
    for (let y = 0; y < height; y++) {
      const top = clamp(y - 2, height - 1) * width;
      const above = clamp(y - 1, height - 1) * width;
      const row = y * width;
      const below = clamp(y + 1, height - 1) * width;
      const bottom = clamp(y + 2, height - 1) * width;
      for (let x = 0; x < width; x++) {
        const index = row + x;
        const sum = workspace[top + x] + workspace[above + x] + workspace[index]
          + workspace[below + x] + workspace[bottom + x];
        strength[index] = clamp(sum - 25 * pixels[index * stride + channel], 255);
      }
    }
    return strength;
  }

  function traceEdges(image, tone, softness) {
    const { data, width, height } = image;
    const workspace = new Uint16Array(width * height);
    const black = new Uint8Array(workspace.length);
    if (softness > 0) {
      const gray = matrix(width, height);
      jsfeat.imgproc.grayscale(data, width, height, gray);
      jsfeat.imgproc.gaussian_blur(gray, gray, 0, softness);
      const strength = edgeStrength(gray.data, width, height, 1, 0, workspace);
      for (let i = 0; i < black.length; i++) black[i] = strength[i] >= tone;
    } else {
      const [red, green, blue] = [0, 1, 2].map(channel =>
        edgeStrength(data, width, height, 4, channel, workspace));
      for (let i = 0; i < black.length; i++) {
        black[i] = luminance(red[i], green[i], blue[i]) >= tone;
      }
    }
    return black;
  }

  // Compare each pixel with the mean of its clipped local window.
  function traceAdaptive(image, tone, softness, outline) {
    const { data, width, height } = image;
    const gray = matrix(width, height);
    jsfeat.imgproc.grayscale(data, width, height, gray);
    const radius = Math.max(1, Math.ceil(Math.min(0.05 * Math.min(width, height), 49) * softness / 4));
    const stride = width + 1;
    const integral = new Float64Array(stride * (height + 1));
    jsfeat.imgproc.compute_integral_image(gray, integral, null, null);
    const ratio = (255 - tone) / 255;
    const black = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      const top = Math.max(0, y - radius);
      const bottom = Math.min(height, y + radius);
      for (let x = 0; x < width; x++) {
        const left = Math.max(0, x - radius);
        const right = Math.min(width, x + radius);
        const sum = integral[bottom * stride + right] - integral[top * stride + right]
          - integral[bottom * stride + left] + integral[top * stride + left];
        const area = (right - left) * (bottom - top);
        const index = y * width + x;
        black[index] = gray.data[index] <= ratio * sum / area;
      }
    }
    if (!outline) return black;
    if (width < 3 || height < 3) return new Uint8Array(black.length);
    for (let i = 0; i < black.length; i++) gray.data[i] = black[i] ? 0 : 255;
    jsfeat.imgproc.canny(gray, gray, 50, 100);
    return gray.data.subarray(0, black.length);
  }

  function render(image, { mode, tone, softness, outline }, target) {
    const black = mode === "edges"
      ? traceEdges(image, tone, softness)
      : traceAdaptive(image, tone, softness, outline);
    for (let i = 0; i < black.length; i++) {
      const index = i * 4;
      const value = black[i] ? 0 : 255;
      target.data[index] = target.data[index + 1] = target.data[index + 2] = value;
      target.data[index + 3] = 255;
    }
  }

  return { render };
})();

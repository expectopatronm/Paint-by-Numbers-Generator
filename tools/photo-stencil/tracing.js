"use strict";

const Stencil = (() => {
  const clamp = (value, end) => Math.max(0, Math.min(end, value));
  const luminance = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
  const matrix = (width, height) =>
    new jsfeat.matrix_t(width, height, jsfeat.U8_t | jsfeat.C1_t);
  const grainCache = new WeakMap();

  // Approximate a bilateral filter with horizontal and vertical passes. Both
  // passes use the original colors to keep strong edges from bleeding.
  function bilateralPass(pixels, guide, width, height, horizontal, spatial, range) {
    const output = new Uint8ClampedArray(pixels.length);
    const step = horizontal ? 1 : width;
    const axisLength = horizontal ? width : height;
    const radius = spatial.length - 1;
    for (let index = 0; index < width * height; index++) {
      const position = horizontal ? index % width : Math.floor(index / width);
      const center = index * 4;
      let red = pixels[center];
      let green = pixels[center + 1];
      let blue = pixels[center + 2];
      let total = 1;
      const first = Math.max(-radius, -position);
      const last = Math.min(radius, axisLength - 1 - position);
      for (let offset = first; offset <= last; offset++) {
        if (offset === 0) continue;
        const neighbor = center + offset * step * 4;
        const difference = Math.max(
          Math.abs(guide[center] - guide[neighbor]),
          Math.abs(guide[center + 1] - guide[neighbor + 1]),
          Math.abs(guide[center + 2] - guide[neighbor + 2]),
        );
        const weight = spatial[Math.abs(offset)] * range[difference];
        red += pixels[neighbor] * weight;
        green += pixels[neighbor + 1] * weight;
        blue += pixels[neighbor + 2] * weight;
        total += weight;
      }
      output[center] = red / total;
      output[center + 1] = green / total;
      output[center + 2] = blue / total;
      output[center + 3] = pixels[center + 3];
    }
    return output;
  }

  function reduceGrain(image, strength) {
    const cached = grainCache.get(image);
    if (cached?.strength === strength) return cached.image;
    const { data, width, height } = image;
    const scale = Math.max(0.75, Math.min(1.6, Math.max(width, height) / 1500));
    const spatialSigma = (0.8 + 0.32 * strength) * scale;
    const colorSigma = 7 + 6 * strength;
    const spatial = Float32Array.from({ length: Math.ceil(2 * spatialSigma) + 1 },
      (_, offset) => Math.exp(-0.5 * (offset / spatialSigma) ** 2));
    const range = Float32Array.from({ length: 256 },
      (_, difference) => Math.exp(-0.5 * (difference / colorSigma) ** 2));
    const horizontal = width > 1 ? bilateralPass(data, data, width, height, true, spatial, range) : data;
    const smoothed = height > 1 ? bilateralPass(horizontal, data, width, height, false, spatial, range) : horizontal;
    const filtered = { data: smoothed, width, height };
    grainCache.set(image, { strength, image: filtered });
    return filtered;
  }

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
      for (let i = 0; i < black.length; i++) {
        const value = strength[i];
        black[i] = luminance(value, value, value) >= tone;
      }
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
    // Match the reference's rounding of local sums near the threshold.
    const integral = new Float32Array(stride * (height + 1));
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
        black[index] = gray.data[index] <= ratio * (sum / area);
      }
    }
    if (!outline) return black;
    if (width < 3 || height < 3) return new Uint8Array(black.length);
    for (let i = 0; i < black.length; i++) gray.data[i] = black[i] ? 0 : 255;
    jsfeat.imgproc.canny(gray, gray, 50, 100);
    return gray.data.subarray(0, black.length);
  }

  // Discard small groups of touching ink pixels, including diagonal neighbors.
  function removeSpeckles(black, width, height, minimumSize) {
    const visited = 2;
    const component = new Int32Array(black.length);
    for (let start = 0; start < black.length; start++) {
      if (!black[start] || black[start] === visited) continue;
      let size = 1;
      component[0] = start;
      black[start] = visited;
      for (let next = 0; next < size; next++) {
        const pixel = component[next];
        const x = pixel % width;
        const y = Math.floor(pixel / width);
        const left = Math.max(0, x - 1);
        const right = Math.min(width - 1, x + 1);
        for (let row = Math.max(0, y - 1); row <= Math.min(height - 1, y + 1); row++) {
          for (let column = left; column <= right; column++) {
            const neighbor = row * width + column;
            if (!black[neighbor] || black[neighbor] === visited) continue;
            black[neighbor] = visited;
            component[size++] = neighbor;
          }
        }
      }
      if (size < minimumSize) {
        for (let i = 0; i < size; i++) black[component[i]] = 0;
      }
    }
  }

  function render(image, { mode, tone, softness, grain, noise, outline }, target) {
    const source = grain > 0 ? reduceGrain(image, grain) : image;
    const black = mode === "edges"
      ? traceEdges(source, tone, softness)
      : traceAdaptive(source, tone, softness, outline);
    if (noise > 0) {
      const sizeAt1000px = 3 * 2 ** (noise - 1);
      const scale = Math.max(image.width, image.height) / 1000;
      const minimumSize = Math.max(2, Math.round(sizeAt1000px * scale));
      removeSpeckles(black, image.width, image.height, minimumSize);
    }
    for (let i = 0; i < black.length; i++) {
      const index = i * 4;
      const value = black[i] ? 0 : 255;
      target.data[index] = target.data[index + 1] = target.data[index + 2] = value;
      target.data[index + 3] = 255;
    }
  }

  return { render };
})();

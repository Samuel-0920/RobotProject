# detection/features/color_features.py
import cv2
import numpy as np
from typing import Tuple, Dict

class ColorFeatureExtractor:
    """Color feature extractor"""

    def __init__(self, bins: int = 64):
        """
        Initializes the color feature extractor.

        Args:
            bins: The number of bins for the histogram.
        """
        self.bins = bins

        # Color semantic mapping
        self.color_mapping = {
            'red': [(150, 255), (0, 100), (0, 100)],          # Red
            'yellow': [(200, 255), (200, 255), (0, 100)],     # Yellow
            'orange': [(200, 255), (100, 200), (0, 100)],     # Orange
            'green': [(0, 100), (150, 255), (0, 100)],        # Green
            'white': [(200, 255), (200, 255), (200, 255)],    # White
            'black': [(0, 80), (0, 80), (0, 80)],             # Black
            'purple': [(100, 255), (0, 100), (150, 255)],     # Purple
            'brown': [(80, 150), (50, 120), (20, 80)],        # Brown
        }

    def compute_color_histogram(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Computes the color histogram for the masked region.

        Args:
            image: The RGB image (H, W, 3).
            mask: The binary mask (H, W).

        Returns:
            hist: The normalized color histogram (bins*3,).
        """
        if np.sum(mask) == 0:
            return np.zeros((self.bins * 3,))

        # Ensure image has the correct data type and memory layout
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Ensure image has a contiguous memory layout
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)

        # Ensure mask has the correct data type
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # Create a mask for calcHist (must be uint8 with values 0 or 255)
        mask_uint8 = np.zeros_like(mask, dtype=np.uint8)
        mask_uint8[mask > 0] = 255

        # Ensure mask is also contiguous
        if not mask_uint8.flags['C_CONTIGUOUS']:
            mask_uint8 = np.ascontiguousarray(mask_uint8)

        # Extract each channel separately, ensuring they are contiguous
        r_channel = np.ascontiguousarray(image[:, :, 0])
        g_channel = np.ascontiguousarray(image[:, :, 1])
        b_channel = np.ascontiguousarray(image[:, :, 2])

        # Compute the histogram for each channel
        hist_r = cv2.calcHist([r_channel], [0], mask_uint8, [self.bins], [0, 256])
        hist_g = cv2.calcHist([g_channel], [0], mask_uint8, [self.bins], [0, 256])
        hist_b = cv2.calcHist([b_channel], [0], mask_uint8, [self.bins], [0, 256])

        # Concatenate the histograms
        hist = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])

        # Normalize
        hist = hist / (np.sum(hist) + 1e-10)

        return hist

    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Extracts the dominant color and its semantic name.

        Args:
            image: The RGB image (H, W, 3).
            mask: The binary mask (H, W).

        Returns:
            dominant_rgb: The dominant RGB value (3,).
            color_name: The semantic name of the color.
        """
        if np.sum(mask) == 0:
            return np.array([0, 0, 0]), "unknown"

        # Extract pixels from the masked region
        masked_pixels = image[mask > 0]

        if len(masked_pixels) == 0:
            return np.array([0, 0, 0]), "unknown"

        # Calculate the average color
        mean_color = np.mean(masked_pixels, axis=0)

        # Map to a semantic color name
        color_name = self._map_to_color_name(mean_color)

        return mean_color.astype(np.uint8), color_name

    def _map_to_color_name(self, rgb: np.ndarray) -> str:
        """
        Maps an RGB value to a semantic color name.

        Args:
            rgb: The RGB value (3,).

        Returns:
            color_name: The semantic name of the color.
        """
        r, g, b = rgb

        # Check the range for each color
        for color_name, (r_range, g_range, b_range) in self.color_mapping.items():
            if (r_range[0] <= r <= r_range[1] and
                g_range[0] <= g <= g_range[1] and
                b_range[0] <= b <= b_range[1]):
                return color_name

        # If no match is found, return the closest color
        return self._find_closest_color(rgb)

    def _find_closest_color(self, rgb: np.ndarray) -> str:
        """
        Finds the closest color.

        Args:
            rgb: The RGB value (3,).

        Returns:
            color_name: The name of the closest color.
        """
        r, g, b = rgb
        min_distance = float('inf')
        closest_color = "mixed"

        for color_name, (r_range, g_range, b_range) in self.color_mapping.items():
            # Calculate the distance to the center of the range
            r_center = (r_range[0] + r_range[1]) / 2
            g_center = (g_range[0] + g_range[1]) / 2
            b_center = (b_range[0] + b_range[1]) / 2

            distance = np.sqrt((r - r_center)**2 + (g - g_center)**2 + (b - b_center)**2)

            if distance < min_distance:
                min_distance = distance
                closest_color = color_name

        return closest_color

    def compute_color_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Computes the similarity between two color histograms.

        Args:
            hist1: The first histogram.
            hist2: The second histogram.

        Returns:
            similarity: The similarity value (between 0 and 1).
        """
        return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)

    def get_color_statistics(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Gets color statistics.

        Args:
            image: The RGB image (H, W, 3).
            mask: The binary mask (H, W).

        Returns:
            stats: A dictionary of color statistics.
        """
        if np.sum(mask) == 0:
            return {
                'mean_color': [0, 0, 0],
                'std_color': [0, 0, 0],
                'mean_r': 0.0,
                'mean_g': 0.0,
                'mean_b': 0.0,
                'std_r': 0.0,
                'std_g': 0.0,
                'std_b': 0.0,
                'dominant_color': [0, 0, 0],
                'color_name': "unknown",
                'histogram': np.zeros((self.bins * 3,)).tolist()
            }

        # Extract pixels from the masked region
        masked_pixels = image[mask > 0]

        # Compute statistics
        mean_color = np.mean(masked_pixels, axis=0)
        std_color = np.std(masked_pixels, axis=0)

        # Extract dominant color
        dominant_color, color_name = self.extract_dominant_color(image, mask)

        # Compute histogram
        histogram = self.compute_color_histogram(image, mask)

        return {
            'mean_color': mean_color.tolist(),
            'std_color': std_color.tolist(),
            # Add separate RGB channel statistics
            'mean_r': float(mean_color[0]),
            'mean_g': float(mean_color[1]),
            'mean_b': float(mean_color[2]),
            'std_r': float(std_color[0]),
            'std_g': float(std_color[1]),
            'std_b': float(std_color[2]),
            'dominant_color': dominant_color.tolist(),
            'color_name': color_name,
            'histogram': histogram.tolist() if isinstance(histogram, np.ndarray) else histogram
        }
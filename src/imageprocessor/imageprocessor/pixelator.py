import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension
import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
import matplotlib.pyplot as plt
import time

def skeletonizer():
    def load_and_preprocess_image(image_path):
        """
        Load and preprocess the binary image
        """
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Ensure it's binary (white lines/circles on black background)
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        return binary_img


    def thin_lines_skimage(binary_img):
        """
        Thinning using scikit-image skeletonize
        """
        # Convert to boolean array (True for white pixels)
        binary_bool = np.where(binary_img > 0, True, False)

        # Apply skeletonization
        skeleton = skeletonize(binary_bool)

        # Convert back to 0-255 format
        return (skeleton * 255).astype(np.uint8)


    def create_dotted_pattern(thinned_img, grid_spacing=5):
        """
        Create dotted pattern by selecting points at regular grid intervals
        from the thinned image

        Args:
            thinned_img: Binary thinned image
            grid_spacing: Distance between dots (higher = more sparse)

        Returns:
            Dotted version of the thinned image
        """
        dotted_img = np.zeros_like(thinned_img)

        # Find all white pixels in the thinned image
        white_pixels = np.where(thinned_img == 255)

        # Create a grid mask
        for i in range(len(white_pixels[0])):
            y, x = white_pixels[0][i], white_pixels[1][i]

            # Keep pixel if it aligns with grid spacing
            if y % grid_spacing == 0 and x % grid_spacing == 0:
                dotted_img[y, x] = 255

        return dotted_img


    def process_image(image_path, grid_spacing=5):
        """
        Main function to process the binary image and create dotted pattern

        Args:
            image_path: Path to input binary image
            grid_spacing: Spacing between dots for regular grid (ignored if adaptive_dots=True)
            adaptive_dots: If True, uses adaptive spacing for better connectivity

        Returns:
            original image, thinned image, dotted image
        """
        # Load and preprocess image
        binary_img = load_and_preprocess_image(image_path)

        # Apply thinning using skimage method
        thinned_img = thin_lines_skimage(binary_img)

        dotted_img = create_dotted_pattern(thinned_img, grid_spacing)

        return binary_img, thinned_img, dotted_img


    def visualize_results(original, thinned, dotted, save_path=None):
        """
        Visualize original, thinned, and dotted images side by side
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original Binary Image')
        axes[0].axis('off')

        axes[1].imshow(thinned, cmap='gray')
        axes[1].set_title('Thinned Image')
        axes[1].axis('off')

        axes[2].imshow(dotted, cmap='gray')
        axes[2].set_title('Dotted Pattern')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    grid_spacings = [3]

    for spacing in grid_spacings:
        # print(f"Processing with grid spacing: {spacing}")
        initial_time = time.time()
        original, thinned, dotted = process_image('/media/aayush/New Volume/Localization/src/imageprocessor/imageprocessor/damaged_image.jpeg',
                                                  grid_spacing=spacing)
        # Save results
        # cv2.imwrite(f'thinned_grid_{spacing}.png', thinned)
        # cv2.imwrite(f'dotted_grid_{spacing}.png', dotted)
        white_arr = np.where(dotted == 255)
        # print(white_arr)
        x_arr= white_arr[1] - 600
        y_arr= white_arr[0] - 400
        flattened_arr = np.stack((y_arr, x_arr), axis=1).flatten().astype(np.int16).tolist()
        # print(flattened_arr)
        # Visualize results
        # visualize_results(original, thinned, dotted)
    return flattened_arr

class Pixelator(Node):
    def __init__(self):
        super().__init__('pixelator')
        self.publisher1 = self.create_publisher(Int16MultiArray, 'pixelated_image', 10)
        self.publisher2 = self.create_publisher(Int16MultiArray, 'bounds', 10)
        self.timer = self.create_timer(0.01, self.publish_array)

    def publish_array(self):
        start_time = time.time()
        flattened_arr = skeletonizer()
        msg1 = Int16MultiArray()
        msg1.data = flattened_arr
        dim0 = MultiArrayDimension()
        dim0.label = 'points'
        dim0.size = len(flattened_arr)//2
        dim0.stride = len(flattened_arr)
        dim1 = MultiArrayDimension()
        dim1.label = 'coords'
        dim1.size = 2
        dim1.stride = 2
        msg1.layout.dim = [dim0, dim1]
        msg1.layout.data_offset = 0
        self.publisher1.publish(msg1)

        msg2 = Int16MultiArray()
        # y_min, y_max, x_min, x_max, theta_min, theta_max
        # y = [0,800], x = [0,1200], theta = [-4,4]
        msg2.data = [0, 400, 0, 600, -4, 4]
        dim0 = MultiArrayDimension()
        dim0.label = 'axes'
        dim0.size = 3
        dim0.stride = 6
        dim1 = MultiArrayDimension()
        dim1.label = 'bounds'
        dim1.size = 2
        dim1.stride = 2
        msg2.layout.dim = [dim0, dim1]
        msg2.layout.data_offset = 0
        self.publisher2.publish(msg2)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

def main(args=None):
    rclpy.init(args=args)
    pixelator = Pixelator()
    rclpy.spin(pixelator)
    pixelator.destroy_node()
    rclpy.shutdown()


import cv2
import numpy as np
from skimage.morphology import skeletonize, thin

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

    grid_spacings = [3]

    for spacing in grid_spacings:
        # print(f"Processing with grid spacing: {spacing}")
        original, thinned, dotted = process_image('/home/rattan/data/era/localisation_github/Localization/src/imageprocessor/imageprocessor/damaged_image.jpeg',
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

def visualise_results(angle, dx, dy):
    dx = dx - 600
    dy = 400 - dy
    img = cv2.imread('/home/rattan/data/era/localisation_github/Localization/src/imageprocessor/imageprocessor/img_1.png')

    cam = cv2.imread('/home/rattan/data/era/localisation_github/Localization/src/imageprocessor/imageprocessor/damaged_image.jpeg')
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    
    R = cv2.getRotationMatrix2D((cam.shape[1] // 2, cam.shape[0] // 2), 90 + np.rad2deg(angle), 1)
    cam = cv2.warpAffine(cam, R, (img.shape[1], img.shape[0]))

    T = np.float32([[1, 0, dx], [0, 1, dy]])
    cam = cv2.warpAffine(cam, T, (img.shape[1], img.shape[0]))

    # Normalize activation map to 0-255 and convert to uint8 if needed
    if cam.dtype != np.uint8:
        cam_norm = cv2.normalize(cam, None, 0, 255, cv2.NORM_MINMAX)
        cam_uint8 = cam_norm.astype(np.uint8)
    else:
        cam_uint8 = cam

    # Apply 'jet' colormap to activation map
    cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    # Blend images (alpha=0.5 for overlay)
    alpha = 0.5
    overlay = cv2.addWeighted(img, 1 - alpha, cam_color, alpha, 0)


    print("dx = " + str(dx) + " dy = " + str(dy) + " theta = " + str(angle))
    # Display the result
    cv2.imshow('Superimposed', overlay)
    cv2.waitKey(0)
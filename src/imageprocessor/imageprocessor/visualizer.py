import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension
from std_msgs.msg import Float32
import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_results(angle, dx, dy):
    img = cv2.imread('/media/aayush/New Volume/Localization/src/imageprocessor/imageprocessor/img_1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.dilate(gray, np.ones((3,3), np.float32), iterations=1)
    gray = cv2.erode(gray, np.ones((3,3), np.float32), iterations=1)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    row = gray[np.where(gray > 0)]

    thresh = np.zeros_like(gray)

    thresh[gray > 200] = 255

    cam = cv2.imread('/media/aayush/New Volume/Localization/src/imageprocessor/imageprocessor/damaged_image.jpeg')
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)

    def rotate_white_part(img, angle, dx=0, dy=0):
        # Find mask of white regions
        mask = (img == 255).astype(np.uint8)
        # Get bounding box of the entire white area
        coords = cv2.findNonZero(mask)
        if coords is None:
            return np.zeros_like(img)  # No white pixels

        x, y, w, h = cv2.boundingRect(coords)

        # Extract region of interest
        roi = img[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        # Rotation matrix centered at ROI center
        M = cv2.getRotationMatrix2D((w // 2, h // 2), np.degrees(angle), 1)
        rotated = cv2.warpAffine(roi, M, (w, h))
        rotated_mask = cv2.warpAffine(roi_mask, M, (w, h))

        # New top-left after translation
        new_x = x + dx
        new_y = y + dy

        # Compute in-bounds region
        x_start = max(new_x, 0)
        y_start = max(new_y, 0)
        x_end = min(new_x + w, img.shape[1])
        y_end = min(new_y + h, img.shape[0])

        x_offset = x_start - new_x
        y_offset = y_start - new_y
        width = x_end - x_start
        height = y_end - y_start

        # If anything is still in bounds
        result = np.zeros_like(img)
        if width > 0 and height > 0:
            cropped_rotated = rotated[y_offset:y_offset+height, x_offset:x_offset+width]
            cropped_mask = rotated_mask[y_offset:y_offset+height, x_offset:x_offset+width]

            region = result[y_start:y_start+height, x_start:x_start+width]
            region = np.where(cropped_mask == 1, cropped_rotated, region)
            result[y_start:y_start+height, x_start:x_start+width] = region

        return result

    best_cam = rotate_white_part(cam, angle, dx, dy)

    plt.figure()

    # Display the grayscale image
    plt.imshow(gray, cmap='gray')

    # Overlay with another image (e.g., activation map), adjust alpha as needed
    plt.imshow(best_cam, cmap='jet', alpha=0.5)

    plt.axis('off')  # Optional: hide axes
    plt.savefig('superimposed.png', dpi=300, bbox_inches='tight')
    plt.show()

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')
        self.spatial_sub = self.create_subscription(Int16MultiArray, 'localisedSpatialCoordinates', self.get_spatial_coords, 10)
        self.rotational_sub = self.create_subscription(Float32, 'localisedRotationalCoordinates', self.get_rotational_coords, 10)

    def get_spatial_coords(self, msg):
            spatial = msg.data
            self.y = spatial[0] - 400
            self.x = spatial[1] - 600
    def get_rotational_coords(self, msg):
            self.theta = msg.data
            visualize_results(self.theta, self.x, self.y)

def main(args=None):
    rclpy.init(args=args)
    visualizer = Visualizer()
    rclpy.spin(visualizer)
    visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension
from std_msgs.msg import Float32
import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_results(angle, dx, dy):
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

class Visualizer(Node):
    def __init__(self):
        super().__init__('visualizer')
        self.x=0
        self.y=0
        self.theta=0
        self.spatial_sub = self.create_subscription(Int16MultiArray, 'localisedSpatialCoordinates', self.get_spatial_coords, 10)
        self.rotational_sub = self.create_subscription(Float32, 'localisedRotationalCoordinates', self.get_rotational_coords, 10)

    def get_spatial_coords(self, msg):
            spatial = msg.data
            self.y = 400 - spatial[1]
            self.x = spatial[0] - 600
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
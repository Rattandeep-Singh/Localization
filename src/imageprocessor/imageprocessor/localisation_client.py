import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, MultiArrayDimension
import time
from localizer.srv import Localisation
from imageprocessor import util

class LocalisationClient(Node):

    def __init__(self):
        super().__init__('localisation_client')
        self.cli = self.create_client(Localisation, 'localise')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Localisation.Request()

    def send_request(self):
        start_time = time.time()

        flattened_arr = util.skeletonizer()

        points = Int16MultiArray()
        points.data = flattened_arr
        dim0 = MultiArrayDimension()
        dim0.label = 'points'
        dim0.size = len(flattened_arr)//2
        dim0.stride = len(flattened_arr)
        dim1 = MultiArrayDimension()
        dim1.label = 'coords'
        dim1.size = 2
        dim1.stride = 2
        points.layout.dim = [dim0, dim1]
        points.layout.data_offset = 0
        self.req.points = points

        bounds = Int16MultiArray()
        # y_min, y_max, x_min, x_max, theta_min, theta_max
        # y = [0,800], x = [0,1200], theta = [-4,4]
        bounds.data = [0, 600, 400, 800, -4, 4]
        dim0 = MultiArrayDimension()
        dim0.label = 'axes'
        dim0.size = 3
        dim0.stride = 6
        dim1 = MultiArrayDimension()
        dim1.label = 'bounds'
        dim1.size = 2
        dim1.stride = 2
        bounds.layout.dim = [dim0, dim1]
        bounds.layout.data_offset = 0
        self.req.bounds = bounds
        
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        return self.cli.call_async(self.req)


def main():
    rclpy.init()

    client = LocalisationClient()
    while True:
        future = client.send_request()
        rclpy.spin_until_future_complete(client, future)
        response = future.result()
        client.get_logger().info(
            'Response received: X = %d Y = %d Theta = %f' %
            (response.position.data[0], response.position.data[1], response.rotation.data))
        util.visualise_results(response.rotation.data, response.position.data[0], response.position.data[1])

    client.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


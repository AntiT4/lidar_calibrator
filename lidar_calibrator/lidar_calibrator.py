import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct
from typing import Tuple
from .lidar_refitter import LidarCalibrator


def spherical_to_cartesian_numpy(r_arr: np.ndarray, lat_arr: np.ndarray, lon_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray,np.ndarray]:
    """
    Spherical (r, latitude, longitude) 좌표를 Cartesian (x, y, z) 좌표로 변환.

    Args:
        r_arr: 반경 (거리)
        lat_arr: 위도 (φ, inclination angle, -π/2 ~ π/2)
        lon_arr: 경도 (θ, azimuth angle, -π ~ π)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (x, y, z) 배열
            - x (np.ndarray): X 좌표 배열
            - y (np.ndarray): Y 좌표 배열
            - z (np.ndarray): Z 좌표 배열
    """
    x_arr = r_arr * np.cos(lat_arr) * np.cos(lon_arr)
    y_arr = r_arr * np.cos(lat_arr) * np.sin(lon_arr)
    z_arr = r_arr * np.sin(lat_arr)
    return x_arr, y_arr, z_arr


def cartesian_to_spherical_numpy(x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cartesian (x, y, z) 좌표를 Spherical (r, latitude, longitude) 좌표로 변환.

    Args:
        x_arr (np.ndarray): X 좌표 배열
        y_arr (np.ndarray): Y 좌표 배열
        z_arr (np.ndarray): Z 좌표 배열

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (r, lat, lon) 배열
            - r: 반경 (거리)
            - lat: 위도 (φ, inclination angle, -π/2 ~ π/2)
            - lon: 경도 (θ, azimuth angle, -π ~ π)
    """
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2 + z_arr ** 2)
    r_arr[r_arr == 0] = 1e-8
    lat_arr = np.arcsin(z_arr / r_arr)  # -π/2 ~ π/2 범위
    lon_arr = np.arctan2(y_arr, x_arr)  # -π ~ π 범위
    return r_arr, lat_arr, lon_arr


def create_pointcloud2(x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray) -> PointCloud2:
    """
    보정된 x, y, z 데이터를 사용하여 PointCloud2 메시지를 생성.

    Args:
        x_arr (np.ndarray): (N,) 크기의 X 좌표 배열
        y_arr (np.ndarray): (N,) 크기의 Y 좌표 배열
        z_arr (np.ndarray): (N,) 크기의 Z 좌표 배열

    Returns:
        PointCloud2: 변환된 포인트 클라우드 메시지
    """
    assert x_arr.shape == y_arr.shape == z_arr.shape, "x, y, z 배열 크기가 일치해야 합니다."
    num_points = x_arr.shape[0]

    # PointCloud2 메시지 설정
    cloud_msg = PointCloud2()
    cloud_msg.header = Header()
    cloud_msg.header.frame_id = "lidar_frame"  # 프레임 ID 설정
    cloud_msg.height = 1  # 1이면 비구조화된 포인트 클라우드
    cloud_msg.width = num_points  # 포인트 개수
    cloud_msg.is_dense = True  # NaN 값 없음
    cloud_msg.is_bigendian = False

    # 필드 정의 (x, y, z)
    cloud_msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud_msg.point_step = 12  # 한 점당 12바이트 (FLOAT32 × 3)
    cloud_msg.row_step = cloud_msg.point_step * num_points

    # 포인트 데이터를 바이너리 형태로 변환
    packed_data = []
    for i in range(num_points):
        packed_data.append(struct.pack('fff', x_arr[i], y_arr[i], z_arr[i]))

    cloud_msg.data = b''.join(packed_data)  # 데이터를 바이너리로 변환

    return cloud_msg


class PointCloudPubSub(Node):
    def __init__(self):
        super().__init__('pointcloud_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/synthetic/point_cloud',
            self.pointcloud_callback,
            10
        )
        self.publisher = self.create_publisher(
            PointCloud2,
            '/converted/point_cloud',
            QoSPresetProfiles.get_from_short_key('sensor_data')
        )
        # 방지: Unused variable warning
        self.subscription
        self.publisher

        self.plate_angle: float = 0.0
        self.calibrator: LidarCalibrator = LidarCalibrator()
        self.lidar_points: np.ndarray = None
        self.lidar_angles: np.ndarray = None

    def pointcloud_callback(self, msg: PointCloud2) -> None:
        """PointCloud2 데이터를 받아서 NumPy 배열로 변환 후 출력"""
        self.lidar_points = self.convert_pointcloud2_to_numpy(msg)
        self.get_logger().info(f"Received PointCloud2 with {len(self.lidar_points)} points")

        self.publish_converted()

    def publish_converted(self) -> PointCloud2:
        if self.lidar_points is None:
            return None
        r_arr, lat_arr, lon_arr = cartesian_to_spherical_numpy(
            self.lidar_points[:, 0], self.lidar_points[:, 1], self.lidar_points[:, 2]
        )
        self.lidar_angles = self.calculate_angles()
        conv_r_arr = r_arr + self.calibrator.calibrate(self.lidar_angles)
        x_arr, y_arr, z_arr = spherical_to_cartesian_numpy(conv_r_arr, lat_arr, lon_arr)
        self.publisher.publish(create_pointcloud2(x_arr, y_arr, z_arr))

    def calculate_angles(self) -> np.ndarray:
        normal_vec = np.ndarray([np.cos(self.plate_angle), np.sin(self.plate_angle), 0.0])
        normal_ray = self.lidar_points / np.linalg.norm(normal_vec)

        dot_product = np.dot(normal_ray, normal_vec)

        incidence_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return incidence_angle.reshape(-1, 1)

    def convert_pointcloud2_to_numpy(self, cloud_msg: PointCloud2) -> np.ndarray:
        """PointCloud2 메시지를 NumPy 배열로 변환"""
        point_list = []

        # 필드 개수 및 데이터 step 크기
        point_step = cloud_msg.point_step
        row_step = cloud_msg.row_step
        data = cloud_msg.data

        # PointCloud2 데이터 파싱
        for i in range(0, len(data), point_step):
            x, y, z = struct.unpack_from('fff', data, offset=i)
            point_list.append([x, y, z])

        return np.array(point_list, dtype=np.float32)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPubSub()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

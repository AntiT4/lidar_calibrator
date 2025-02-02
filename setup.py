from setuptools import find_packages, setup

package_name = 'lidar_calibrator'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='53856879+AntiT4@users.noreply.github.com',
    description='SubPub that convert synthetic lidar data to real lidar data.',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "lidar_calibrator=lidar_calibrator.lidar_calibrator:main",
        ],
    },
)

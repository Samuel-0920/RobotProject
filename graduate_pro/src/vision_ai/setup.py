from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vision_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 添加launch文件
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # 添加配置文件
        (os.path.join('share', package_name, 'config'), 
         glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    extras_require={
        'test': ['pytest'],
    },
    zip_safe=True,
    maintainer='qi',
    maintainer_email='qi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'scan_planner_node = vision_ai.scan_planner_node:main',
            'scan_executor_node = vision_ai.scan_executor_node:main',
            'smart_stitcher_node = vision_ai.smart_stitcher_node:main',
            'gui_config_node = vision_ai.gui_config_node:main',
            'detection_node = vision_ai.detection_node:main',
            'visualization_trigger = vision_ai.visualization_trigger:main',
            # ⭐ 新增追踪系统节点 ⭐
            'tracking_node = vision_ai.tracking_system.tracking_node:main',
        ],
    },
)


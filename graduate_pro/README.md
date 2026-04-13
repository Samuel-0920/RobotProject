# Vision AI 项目总览

本项目是一个基于 ROS 2 Humble 的视觉智能系统，围绕扫描（Scan）、检测（Detect）、追踪（Track）和抓取（Grasp）四个核心模块构建，并通过脚本与 tmux 流水线实现一键启动与协同运行。主要代码位于：

- src/vision_ai/vision_ai
- 脚本：restart.sh（一键构建与启动）、dev.sh（开发/调试编排）

推荐优先阅读 restart.sh 与 dev.sh 了解整体流程编排。

## 快速开始

- 一键构建并启动（默认开启显示）：
  - ./restart.sh
- 无显示/离线渲染模式：
  - ./restart.sh --no-display
- 强制开启显示：
  - ./restart.sh --display
- 开发/调试编排：
  - ./dev.sh both        # 规划 + GUI + 检测
  - ./dev.sh full_system # 相机 + 机械臂 + 3D 检测 + 全流程

restart.sh 会：
- 激活 Conda 环境并 source ROS 2
- 配置显示/无显示运行所需的环境变量
- 在工作区构建指定包（vision_ai_interfaces、vision_ai、xarm_controller、camera_node）
- 成功后调用 ./dev.sh full_system 进入全流程

## 系统结构与模块

- Scan（扫描）
  - 节点：scan_planner_node、scan_executor_node、smart_stitcher_node
  - 作用：规划扫描路径、执行扫描动作、图像/点云拼接（stitcher）
- Detect（检测）
  - 节点：detection_node
  - 作用：目标检测与分割；集成 YOLO、SAM2 等能力与后处理
- Track（追踪）
  - 节点：tracking_node
  - 作用：目标状态估计与跟踪，可视化与在线学习
- Grasp（抓取）
  - 组件：simple_grasp_validator.py（自动静态抓取系统）
  - 作用：在检测完成后基于结果触发/验证抓取策略

硬件与驱动：
- Camera：ros2 run camera_node realsense_publisher
- XArm（机械臂）：ros2 run xarm_controller xarm_controller

## 核心算法原理

以下从 Scan → Stitch → Detect → Track → Grasp 的数据流顺序解释各核心算法如何协同工作，配合具体代码位置便于审阅。

### 1) Scan：自适应扫描规划与执行
- 规划核心
  - 几何分析求最小包围盒与最佳朝向，依据相机 1280×720 像素与内参计算视场角，确定拍摄高度与重叠率。
  - 自适应策略优先最少拍摄点数，同时保证覆盖与重叠（overlap_ratio 动态计算）。
  - 关键实现：最小包围盒与 yaw 估计、FOV/高度/点数计算、网格/邻接信息。
  - 参考代码：
    - [scan_planner_node.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/scan_planner_node.py)
- 执行核心
  - 机械臂按 waypoint 运动→采集彩色/深度→记录位姿与规划元信息→调用 Stitch 服务。
  - 参考代码：
    - [scan_executor_node.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/scan_executor_node.py)

理论要点：
- FOV 与高度：通过 fx、fy 推导水平/垂直视场角；给定区域长短边，计算满足覆盖的最小高度，并限制在安全范围。
- 重叠最优化：根据区域面积、自适应重叠率与相机分辨率，求最少采样点数与网格布置，降低拼接误差与漏检概率。

### 2) Stitch：基于规划信息的直接网格拼接
- 思路
  - 直接使用规划阶段产生的 grid 位置信息与相机相对位姿构建画布与像素映射，不依赖传统特征匹配。
  - 同时保留每个像素来源 waypoint 的映射与深度文件索引，便于后续 3D 定位/逆查。
- 核心实现
  - 提取规划信息 relationships/overlap_map/grid_layout。
  - direct_grid_stitching：按行列与重叠率放置图块，构建最终分辨率。
  - fusion_mapping/waypoint_contributions：记录像素级来源与深度关联；输出 pkl+json 摘要。
  - 参考代码：
    - [smart_stitcher_node.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/smart_stitcher_node.py)

理论要点：
- 将规划阶段的空间约束前移到图像层，拼接更稳定、可解释；像素到 waypoint 的可追溯性支持 3D 反演与不确定性估计。

### 3) Detect：YOLO+SAM2 的 2D→3D 增强检测管线
- 流程
  - YOLO 目标候选→SAM2 精细分割→特征与 3D 后处理→构建参考库和可视化。
  - 参考代码：
    - 管线入口与调度：[enhanced_detection_pipeline.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/detection/enhanced_detection_pipeline.py)
    - 3D 后处理（去重合并）：[detection_post_processor.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/detection/utils/detection_post_processor.py)
    - 坐标/动态补偿与 3D 分析：[coordinate_calculator.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/detection/utils/coordinate_calculator.py)
    - 几何/形状特征提取：[shape_features.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/detection/features/shape_features.py)

- 关键算法
  - 相机到世界坐标转换：手眼标定四元数与平移构成 T_cam→tcp，再用机械臂 TCP 欧拉角构建 T_tcp→world；并加入 yaw 角动态补偿插值，修正随姿态变化的系统误差。
  - 3D 去重/合并：基于世界坐标质心距离、平均深度差、估计高度差与 mask IoU 综合相似度，判定重复并合并边框/掩膜与置信度；减少同物体多次检测。
  - 特征库构建：提取颜色直方图/命名、2D 形状、3D 包围盒/FPFH/PCA/密度与空间特征，生成标准化对象条目与可视化图。

理论要点：
- “2D→3D”的约束增强：分割得到高质量 mask，结合深度与位姿完成 3D 重心/高度估计；在三维空间判重更稳定。
- 动态补偿：以 yaw 为自变量的插值补偿，抑制不同朝向带来的系统偏差，提高 3D 定位一致性。

### 4) Track：混合相似度匹配 + 在线学习 + 3D 卡尔曼滤波
- 流程
  - 加载参考特征库→筛选同类候选→混合相似度匹配（几何/外观/形状/空间加权）→在线学习自适应权重→3D 卡尔曼滤波估计轨迹与稳定性。
  - 参考代码：
    - 主体与相似度/学习逻辑：[enhanced_tracker.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/tracking_system/enhanced_tracker.py)
    - 3D 卡尔曼与稳定性管理：[kalman_tracker.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/tracking_system/filters/kalman_tracker.py)
    - 相似度细节（多特征融合）：[similarity_calculator.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/detection/features/similarity_calculator.py)

- 关键算法
  - 混合相似度：几何>形状>外观>空间的加权组合，并在光照/距离等上下文下由在线学习器动态调整权重；支持历史成功样本混合，提升稳健性。
  - 3D 卡尔曼：状态[x,y,z,vx,vy,vz]，仅位置观测；依据速度与协方差评估稳定性，作为抓取触发的先决条件之一。

理论要点：
- 以“物理一致性”为主导的匹配准则（3D 几何与空间）显著降低误匹配风险；在线学习使系统对环境变化更自适应。

### 5) Grasp：追踪触发的单对象抓取序列
- 流程
  - 追踪节点在目标稳定且相似度达标时发布抓取触发消息，包含目标 ID 与参考数据目录。
  - 抓取系统加载 grasp_info.json 获取世界坐标、物体宽度与高度等，发布 /xarm/target_pose 与夹爪控制。
  - 参考代码：
    - [simple_grasp_validator.py](file:///Users/chenqi/graduate_pro/src/vision_ai/vision_ai/simple_grasp_validator.py)

理论要点：
- 夹爪宽度来自分割 mask 的最小外接矩形与深度等比例换算；配合安全余量裁剪到可执行范围，保证抓取可行性与稳定性。

## 数据流与话题

- 相机
  - 输入：/camera/color/image_raw、/camera/depth/image_raw
- 机械臂
  - 输入：/xarm/current_pose；控制：/xarm/target_pose、/xarm/gripper_target；服务：/xarm/gripper_open、/xarm/gripper_close、/xarm/go_home
- Scan
  - 服务：plan_scan、execute_scan；调用 Stitch 服务：process_stitching
- Stitch
  - 输入：扫描采集图像与位姿元信息；输出：拼接图、fusion_mapping 与摘要
- Detect
  - 输入：拼接参考图与对应深度映射；输出：目标集与参考特征库、可视化图、detection_complete 信号
- Track
  - 输入：检测候选、深度与位姿；输出：目标状态/稳定性、grasp_trigger
- Grasp
  - 输入：grasp_trigger 与 grasp_info.json；输出：抓取执行与完成状态

## 进程/会话编排（tmux）

dev.sh 对应窗口布局（会话名 vision_ai）：
- 窗口0：硬件节点
  - 左：相机 realsense_publisher
  - 右：机械臂控制 xarm_controller
- 窗口1：AI 处理
  - 左：scan_planner_node
  - 右：scan_executor_node
- 窗口2：拼接与检测
  - 左：smart_stitcher_node
  - 右：detection_node
- 窗口3：GUI 与日志
  - 左：gui_config_node
  - 右：自动实验记录 auto_logger.sh（延迟启动）
- 窗口4：追踪与抓取控制台
  - tracking_node 及抓取流程控制与状态输出

默认 attach 到窗口2（拼接与检测）便于观察。

## 目录导览（重点）

src/vision_ai/vision_ai/
- scan 系列
  - scan_planner_node.py
  - scan_executor_node.py
  - smart_stitcher_node.py
- detect 系列
  - detection_node.py
  - detection/（检测管线与子模块）
    - detectors/（如 YOLO）
    - segmentors/（如 SAM2）
    - features/、utils/、pipelines（enhanced_detection_pipeline 等）
- track 系列
  - tracking_system/
    - tracking_node.py（主跟踪节点）
    - filters/、ui/、utils/、adaptive_learning/
- grasp 系列
  - simple_grasp_validator.py（自动静态抓取流程）
- 其它辅助
  - gui_config_node.py、detection_monitor.py 等
  - scripts/（如下载模型的脚本）
  - testcode/（功能验证脚本与测试）

## 环境与构建

- ROS 版本：ROS 2 Humble
- 构建工具：colcon（由 restart.sh 调用）
- Conda 环境：rel_env（由 restart.sh 激活）
- 依赖提示：
  - restart.sh 会设置 NumPy 头文件路径与必要的库路径
  - 显示模式环境变量：
    - VISION_AI_DISPLAY=on/off
    - 显示：QT_QPA_PLATFORM=xcb，MPLBACKEND=TkAgg
    - 无显示：QT_QPA_PLATFORM=offscreen，MPLBACKEND=Agg

建议通过 restart.sh 统一完成构建与启动，避免缺失环境变量导致的运行问题。

## 常见使用场景

- 仅调试规划/检测链路：
  - ./dev.sh both
- 全流程演示（含相机与机械臂）：
  - ./restart.sh 或 ./dev.sh full_system
- 无显示服务器/远程运行：
  - ./restart.sh --no-display

## 备注

- 构建包列表与节点名称以脚本为准；新增节点后可按现有模式在 dev.sh 中增加窗口与启动命令。
- testcode/ 目录提供若干验证脚本，便于快速功能核对与回归测试。

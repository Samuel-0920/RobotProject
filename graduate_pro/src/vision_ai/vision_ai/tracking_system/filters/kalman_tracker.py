# tracking/filters/kalman_tracker.py
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

class KalmanFilter3D:
    """3D卡尔曼滤波器 - 用于物体位置追踪"""
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 1.0):
        """
        初始化3D卡尔曼滤波器
        
        Args:
            process_noise: 过程噪声 (系统不确定性)
            measurement_noise: 测量噪声 (观测不确定性)
        """
        # 状态向量: [x, y, z, vx, vy, vz] (位置 + 速度)
        self.state = np.zeros(6)
        
        # 状态协方差矩阵
        self.P = np.eye(6) * 1000  # 初始不确定性较大
        
        # 状态转移矩阵 (匀速运动模型)
        self.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx*dt
            [0, 1, 0, 0, 1, 0],  # y = y + vy*dt  
            [0, 0, 1, 0, 0, 1],  # z = z + vz*dt
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1]   # vz = vz
        ])
        
        # 观测矩阵 (只能观测位置)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # 过程噪声协方差
        self.Q = np.eye(6) * process_noise
        
        # 测量噪声协方差
        self.R = np.eye(3) * measurement_noise
        
        # 滤波器状态
        self.initialized = False
        self.last_update_time = None
        self.prediction_count = 0
        self.update_count = 0
        
        print(f"[KALMAN] 3D卡尔曼滤波器初始化: 过程噪声={process_noise}, 测量噪声={measurement_noise}")

    def initialize(self, initial_position: List[float], timestamp: Optional[str] = None):
        """
        初始化滤波器
        
        Args:
            initial_position: [x, y, z] 初始位置 (mm)
            timestamp: 时间戳
        """
        try:
            self.state[:3] = initial_position  # 设置初始位置
            self.state[3:] = 0  # 初始速度为0
            
            self.initialized = True
            self.last_update_time = timestamp or datetime.now().isoformat()
            self.prediction_count = 0
            self.update_count = 0
            
            print(f"[KALMAN] 滤波器已初始化: 位置=[{initial_position[0]:.1f}, {initial_position[1]:.1f}, {initial_position[2]:.1f}]mm")
            
        except Exception as e:
            print(f"[KALMAN] 初始化失败: {e}")

    def predict(self, dt: float = 1.0):
        """
        预测步骤
        
        Args:
            dt: 时间间隔 (秒)
        """
        try:
            if not self.initialized:
                return
            
            # 更新状态转移矩阵的时间步长
            self.F[0, 3] = dt
            self.F[1, 4] = dt
            self.F[2, 5] = dt
            
            # 预测状态
            self.state = self.F @ self.state
            
            # 预测协方差
            self.P = self.F @ self.P @ self.F.T + self.Q
            
            self.prediction_count += 1
            
            print(f"[KALMAN] 预测完成: dt={dt:.3f}s, 位置=[{self.state[0]:.1f}, {self.state[1]:.1f}, {self.state[2]:.1f}]mm")
            
        except Exception as e:
            print(f"[KALMAN] 预测失败: {e}")

    def update(self, measurement: List[float], measurement_uncertainty: Optional[List[float]] = None):
        """
        更新步骤 (观测更新)
        
        Args:
            measurement: [x, y, z] 观测位置 (mm)
            measurement_uncertainty: [x_std, y_std, z_std] 观测不确定性 (mm)
        """
        try:
            if not self.initialized:
                self.initialize(measurement)
                return
            
            z = np.array(measurement)
            
            # 动态调整测量噪声
            if measurement_uncertainty:
                R_dynamic = np.diag([u**2 for u in measurement_uncertainty])
            else:
                R_dynamic = self.R
            
            # 计算卡尔曼增益
            S = self.H @ self.P @ self.H.T + R_dynamic
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            # 更新状态
            y = z - self.H @ self.state  # 观测残差
            self.state = self.state + K @ y
            
            # 更新协方差
            I = np.eye(6)
            self.P = (I - K @ self.H) @ self.P
            
            self.update_count += 1
            
            print(f"[KALMAN] 更新完成: 观测=[{measurement[0]:.1f}, {measurement[1]:.1f}, {measurement[2]:.1f}]mm, "
                  f"滤波后=[{self.state[0]:.1f}, {self.state[1]:.1f}, {self.state[2]:.1f}]mm")
            
        except Exception as e:
            print(f"[KALMAN] 更新失败: {e}")

    def get_filtered_position(self) -> List[float]:
        """获取滤波后的位置"""
        if not self.initialized:
            return [0.0, 0.0, 0.0]
        return self.state[:3].tolist()

    def get_velocity(self) -> List[float]:
        """获取估计速度"""
        if not self.initialized:
            return [0.0, 0.0, 0.0]
        return self.state[3:].tolist()

    def get_position_uncertainty(self) -> List[float]:
        """获取位置不确定性 (标准差)"""
        if not self.initialized:
            return [1000.0, 1000.0, 1000.0]
        return [math.sqrt(self.P[i, i]) for i in range(3)]

    def is_stable(self, max_velocity: float = 10.0, max_uncertainty: float = 50.0) -> bool:
        """
        检查追踪是否稳定
        
        Args:
            max_velocity: 最大允许速度 (mm/s)
            max_uncertainty: 最大允许不确定性 (mm)
        """
        try:
            if not self.initialized:
                return False
            
            velocity = self.get_velocity()
            velocity_magnitude = math.sqrt(sum(v**2 for v in velocity))
            
            uncertainty = self.get_position_uncertainty()
            max_uncertainty_current = max(uncertainty)
            
            is_stable = (velocity_magnitude <= max_velocity and 
                        max_uncertainty_current <= max_uncertainty)
            
            print(f"[KALMAN] 稳定性检查: 速度={velocity_magnitude:.1f}mm/s (≤{max_velocity}), "
                  f"不确定性={max_uncertainty_current:.1f}mm (≤{max_uncertainty}), 稳定={is_stable}")
            
            return is_stable
            
        except Exception as e:
            print(f"[KALMAN] 稳定性检查失败: {e}")
            return False

class TrackingStabilityManager:
    """追踪稳定性管理器 - 集成卡尔曼滤波和传统方法"""
    
    def __init__(self, target_id: str, config: Optional[Dict] = None):
        """
        初始化稳定性管理器
        
        Args:
            target_id: 目标ID
            config: 配置参数
        """
        self.target_id = target_id
        self.config = config or self._get_default_config()
        
        # 卡尔曼滤波器
        self.kalman = KalmanFilter3D(
            process_noise=self.config.get('process_noise', 0.1),
            measurement_noise=self.config.get('measurement_noise', 1.0)
        )
        
        # 历史记录
        self.position_history = []  # 原始位置历史
        self.filtered_history = []  # 滤波后位置历史
        self.confidence_history = []  # 置信度历史
        self.timestamp_history = []  # 时间戳历史
        
        # 稳定性状态
        self.stability_streak = 0  # 连续稳定次数
        self.instability_streak = 0  # 连续不稳定次数
        
        print(f"[STABILITY_MGR] 稳定性管理器初始化: {target_id}")

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'process_noise': 0.1,
            'measurement_noise': 1.0,
            'max_history_length': 10,
            'min_confidence_for_update': 0.5,
            'stability_velocity_threshold': 10.0,  # mm/s
            'stability_uncertainty_threshold': 50.0,  # mm
            'legacy_stability_threshold': 15.0,  # mm
            'min_stable_streak': 2,  # 连续稳定次数
            'max_instable_streak': 3  # 最大不稳定次数
        }

    def add_measurement(self, position: List[float], confidence: float, 
                       timestamp: Optional[str] = None) -> Dict:
        """
        添加新的测量值
        
        Args:
            position: [x, y, z] 位置 (mm)
            confidence: 置信度 [0-1]
            timestamp: 时间戳
            
        Returns:
            Dict: 处理结果
        """
        try:
            current_time = timestamp or datetime.now().isoformat()
            
            # 记录原始历史
            self.position_history.append(position.copy())
            self.confidence_history.append(confidence)
            self.timestamp_history.append(current_time)
            
            # 限制历史长度
            max_length = self.config['max_history_length']
            if len(self.position_history) > max_length:
                self.position_history = self.position_history[-max_length:]
                self.confidence_history = self.confidence_history[-max_length:]
                self.timestamp_history = self.timestamp_history[-max_length:]
            
            # 卡尔曼滤波处理
            kalman_result = self._process_with_kalman(position, confidence, current_time)
            
            # 稳定性评估
            stability_result = self._evaluate_stability()
            
            # 合并结果
            result = {
                'original_position': position,
                'filtered_position': kalman_result['filtered_position'],
                'velocity': kalman_result['velocity'],
                'uncertainty': kalman_result['uncertainty'],
                'is_stable_kalman': kalman_result['is_stable'],
                'is_stable_legacy': stability_result['legacy_stable'],
                'is_stable_combined': stability_result['combined_stable'],
                'stability_score': stability_result['stability_score'],
                'confidence': confidence,
                'timestamp': current_time
            }
            
            return result
            
        except Exception as e:
            print(f"[STABILITY_MGR] 添加测量值失败: {e}")
            return {
                'original_position': position,
                'filtered_position': position,
                'is_stable_combined': False,
                'confidence': confidence
            }

    def _process_with_kalman(self, position: List[float], confidence: float, 
                           timestamp: str) -> Dict:
        """使用卡尔曼滤波处理"""
        try:
            # 计算时间间隔
            dt = self._calculate_time_delta(timestamp)
            
            # 预测步骤
            if dt > 0:
                self.kalman.predict(dt)
            
            # 更新步骤 (只有置信度足够高才更新)
            if confidence >= self.config['min_confidence_for_update']:
                # 根据置信度调整测量不确定性
                measurement_uncertainty = self._calculate_measurement_uncertainty(confidence)
                self.kalman.update(position, measurement_uncertainty)
            
            # 获取结果
            filtered_position = self.kalman.get_filtered_position()
            velocity = self.kalman.get_velocity()
            uncertainty = self.kalman.get_position_uncertainty()
            is_stable = self.kalman.is_stable(
                self.config['stability_velocity_threshold'],
                self.config['stability_uncertainty_threshold']
            )
            
            # 记录滤波后历史
            self.filtered_history.append(filtered_position.copy())
            if len(self.filtered_history) > self.config['max_history_length']:
                self.filtered_history = self.filtered_history[-self.config['max_history_length']:]
            
            return {
                'filtered_position': filtered_position,
                'velocity': velocity,
                'uncertainty': uncertainty,
                'is_stable': is_stable,
                'time_delta': dt
            }
            
        except Exception as e:
            print(f"[STABILITY_MGR] 卡尔曼处理失败: {e}")
            return {
                'filtered_position': position,
                'velocity': [0, 0, 0],
                'uncertainty': [100, 100, 100],
                'is_stable': False,
                'time_delta': 0
            }

    def _calculate_time_delta(self, current_timestamp: str) -> float:
        """计算时间间隔"""
        try:
            if len(self.timestamp_history) < 2:
                return 1.0  # 默认1秒
            
            from datetime import datetime
            current_time = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))
            last_time = datetime.fromisoformat(self.timestamp_history[-2].replace('Z', '+00:00'))
            
            delta = (current_time - last_time).total_seconds()
            return max(0.1, min(10.0, delta))  # 限制在0.1-10秒之间
            
        except Exception as e:
            print(f"[STABILITY_MGR] 时间间隔计算失败: {e}")
            return 1.0

    def _calculate_measurement_uncertainty(self, confidence: float) -> List[float]:
        """根据置信度计算测量不确定性"""
        try:
            # 置信度越低，不确定性越高
            base_uncertainty = 5.0  # mm
            uncertainty_factor = (1.0 - confidence) * 10.0 + 1.0
            uncertainty = base_uncertainty * uncertainty_factor
            
            return [uncertainty, uncertainty, uncertainty]
            
        except Exception as e:
            print(f"[STABILITY_MGR] 测量不确定性计算失败: {e}")
            return [10.0, 10.0, 10.0]

    def _evaluate_stability(self) -> Dict:
        """评估稳定性 (卡尔曼 + 传统方法)"""
        try:
            # 卡尔曼稳定性
            kalman_stable = self.kalman.is_stable()
            
            # 传统方法稳定性
            legacy_stable = self._check_legacy_stability()
            
            # 置信度稳定性
            confidence_stable = self._check_confidence_stability()
            
            # 综合稳定性评估
            stability_factors = [kalman_stable, legacy_stable, confidence_stable]
            stable_count = sum(stability_factors)
            
            # 需要至少2/3的方法认为稳定
            combined_stable = stable_count >= 2
            
            # 更新稳定性计数
            if combined_stable:
                self.stability_streak += 1
                self.instability_streak = 0
            else:
                self.instability_streak += 1
                self.stability_streak = 0
            
            # 计算稳定性分数
            stability_score = self._calculate_stability_score()
            
            print(f"[STABILITY_MGR] 稳定性评估: 卡尔曼={kalman_stable}, 传统={legacy_stable}, "
                  f"置信度={confidence_stable}, 综合={combined_stable}, 分数={stability_score:.3f}")
            
            return {
                'kalman_stable': kalman_stable,
                'legacy_stable': legacy_stable,
                'confidence_stable': confidence_stable,
                'combined_stable': combined_stable,
                'stability_score': stability_score,
                'stability_streak': self.stability_streak,
                'instability_streak': self.instability_streak
            }
            
        except Exception as e:
            print(f"[STABILITY_MGR] 稳定性评估失败: {e}")
            return {
                'combined_stable': False,
                'stability_score': 0.0
            }

    def _check_legacy_stability(self) -> bool:
        """传统稳定性检查 (位置变化)"""
        try:
            if len(self.position_history) < 2:
                return False
            
            recent_positions = self.position_history[-2:]
            max_deviation = 0
            
            for i in range(1, len(recent_positions)):
                deviation = math.sqrt(
                    (recent_positions[i][0] - recent_positions[i-1][0])**2 +
                    (recent_positions[i][1] - recent_positions[i-1][1])**2
                )
                max_deviation = max(max_deviation, deviation)
            
            threshold = self.config['legacy_stability_threshold']
            return max_deviation <= threshold
            
        except Exception as e:
            print(f"[STABILITY_MGR] 传统稳定性检查失败: {e}")
            return False

    def _check_confidence_stability(self) -> bool:
        """置信度稳定性检查"""
        try:
            if len(self.confidence_history) < 2:
                return False
            
            recent_confidences = self.confidence_history[-3:]  # 最近3次
            
            # 检查置信度不能太低
            min_confidence = min(recent_confidences)
            if min_confidence < 0.5:
                return False
            
            # 检查置信度变化不能太大
            confidence_std = np.std(recent_confidences)
            if confidence_std > 0.2:  # 标准差不超过0.2
                return False
            
            return True
            
        except Exception as e:
            print(f"[STABILITY_MGR] 置信度稳定性检查失败: {e}")
            return False

    def _calculate_stability_score(self) -> float:
        """计算综合稳定性分数 [0-1]"""
        try:
            score = 0.0
            
            # 连续稳定性贡献 (0-0.4)
            stability_ratio = min(self.stability_streak / 5.0, 1.0)
            score += stability_ratio * 0.4
            
            # 卡尔曼不确定性贡献 (0-0.3)
            if self.kalman.initialized:
                uncertainty = self.kalman.get_position_uncertainty()
                avg_uncertainty = sum(uncertainty) / 3.0
                uncertainty_score = max(0, 1.0 - avg_uncertainty / 100.0)  # 100mm为满分
                score += uncertainty_score * 0.3
            
            # 置信度贡献 (0-0.3)
            if self.confidence_history:
                recent_confidence = np.mean(self.confidence_history[-3:])
                score += recent_confidence * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            print(f"[STABILITY_MGR] 稳定性分数计算失败: {e}")
            return 0.0

    def get_stability_report(self) -> Dict:
        """获取稳定性报告"""
        try:
            if not self.position_history:
                return {'status': 'no_data'}
            
            report = {
                'target_id': self.target_id,
                'total_measurements': len(self.position_history),
                'stability_streak': self.stability_streak,
                'instability_streak': self.instability_streak,
                'kalman_initialized': self.kalman.initialized,
                'current_stability_score': self._calculate_stability_score()
            }
            
            if self.kalman.initialized:
                report.update({
                    'current_position': self.kalman.get_filtered_position(),
                    'current_velocity': self.kalman.get_velocity(),
                    'current_uncertainty': self.kalman.get_position_uncertainty(),
                    'kalman_updates': self.kalman.update_count,
                    'kalman_predictions': self.kalman.prediction_count
                })
            
            if self.confidence_history:
                report.update({
                    'avg_confidence': np.mean(self.confidence_history),
                    'min_confidence': np.min(self.confidence_history),
                    'max_confidence': np.max(self.confidence_history),
                    'confidence_std': np.std(self.confidence_history)
                })
            
            return report
            
        except Exception as e:
            print(f"[STABILITY_MGR] 获取稳定性报告失败: {e}")
            return {'status': 'error', 'error': str(e)}

    def reset(self):
        """重置管理器"""
        try:
            self.kalman = KalmanFilter3D(
                self.config['process_noise'],
                self.config['measurement_noise']
            )
            self.position_history.clear()
            self.filtered_history.clear()
            self.confidence_history.clear()
            self.timestamp_history.clear()
            self.stability_streak = 0
            self.instability_streak = 0
            
            print(f"[STABILITY_MGR] 管理器已重置: {self.target_id}")
            
        except Exception as e:
            print(f"[STABILITY_MGR] 重置失败: {e}")

    def save_to_file(self, filepath: str):
        """保存稳定性数据到文件"""
        try:
            data = {
                'target_id': self.target_id,
                'config': self.config,
                'position_history': self.position_history,
                'filtered_history': self.filtered_history,
                'confidence_history': self.confidence_history,
                'timestamp_history': self.timestamp_history,
                'stability_report': self.get_stability_report()
            }
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"[STABILITY_MGR] 稳定性数据已保存: {filepath}")
            
        except Exception as e:
            print(f"[STABILITY_MGR] 保存稳定性数据失败: {e}")
# utils/adaptive_learner.py
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pickle

class AdaptiveThresholdManager:
    """自适应阈值管理器 - 支持在线学习和历史优化"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化自适应阈值管理器
        
        Args:
            config_file: 配置文件路径，用于保存学习结果
        """
        self.config_file = config_file
        
        # 基础阈值配置
        self.base_thresholds = {
            'geometric': {
                'fpfh': 0.85,
                'pca': 0.75,
                'bbox': 0.80,
                'shape_context_3d': 0.78
            },
            'appearance': {
                'histogram': 0.70,
                'color_name': 1.0  # 完全匹配
            },
            'shape': {
                'hu_moments': 0.75,
                'shape_descriptors': 0.80,
                'fourier': 0.72
            },
            'spatial': {
                'position': 0.60,
                'size': 0.70,
                'distance': 0.65
            }
        }
        
        # 特征权重（可学习）
        self.feature_weights = {
            'geometric': 0.4,
            'appearance': 0.2,
            'shape': 0.3,
            'spatial': 0.1
        }
        
        # 学习历史
        self.learning_history = {
            'successes': [],  # (feature_type, similarity, quality, timestamp)
            'failures': [],   # (feature_type, similarity, quality, timestamp)
            'adjustments': [] # (threshold_changes, performance_impact, timestamp)
        }
        
        # 性能统计
        self.performance_stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'last_update': datetime.now()
        }
        
        # 加载历史数据
        self.load_learning_data()
        
    def get_adaptive_threshold(self, feature_type: str, sub_feature: str, 
                             feature_quality: float, context: Dict = None) -> float:
        """
        获取自适应阈值
        
        Args:
            feature_type: 特征类型 ('geometric', 'appearance', 'shape', 'spatial')
            sub_feature: 子特征名称 ('fpfh', 'histogram', 'hu_moments' 等)
            feature_quality: 特征质量分数 (0-100)
            context: 上下文信息（场景、光照等）
            
        Returns:
            adaptive_threshold: 自适应调整后的阈值
        """
        # 获取基础阈值
        base_threshold = self.base_thresholds.get(feature_type, {}).get(sub_feature, 0.75)
        
        # 质量调整因子
        quality_factor = self._calculate_quality_adjustment(feature_quality)
        
        # 历史性能调整因子
        history_factor = self._calculate_history_adjustment(feature_type, sub_feature)
        
        # 上下文调整因子
        context_factor = self._calculate_context_adjustment(context)
        
        # 计算最终阈值
        adaptive_threshold = base_threshold * quality_factor * history_factor * context_factor
        
        # 限制在合理范围内
        adaptive_threshold = max(0.3, min(0.95, adaptive_threshold))
        
        return adaptive_threshold
    
    def _calculate_quality_adjustment(self, feature_quality: float) -> float:
        """基于特征质量计算调整因子"""
        quality_normalized = feature_quality / 100.0
        
        # 质量越高，阈值可以更严格（因子 > 1）
        # 质量越低，阈值应该放宽（因子 < 1）
        if quality_normalized > 0.8:
            return 1.0 + (quality_normalized - 0.8) * 0.5  # 最多提高10%
        elif quality_normalized < 0.5:
            return 0.7 + quality_normalized * 0.6  # 最多降低30%
        else:
            return 0.9 + quality_normalized * 0.2
    
    def _calculate_history_adjustment(self, feature_type: str, sub_feature: str) -> float:
        """基于历史性能计算调整因子"""
        # 获取该特征的历史记录
        successes = [h for h in self.learning_history['successes'] 
                    if h[0] == f"{feature_type}.{sub_feature}"]
        failures = [h for h in self.learning_history['failures']
                   if h[0] == f"{feature_type}.{sub_feature}"]
        
        if len(successes) < 5 and len(failures) < 5:
            return 1.0  # 数据不足，使用默认
        
        total_cases = len(successes) + len(failures)
        success_rate = len(successes) / total_cases
        
        # 成功率高 -> 可以提高阈值（更严格）
        # 成功率低 -> 应该降低阈值（更宽松）
        if success_rate > 0.8:
            return 1.1  # 提高阈值10%
        elif success_rate < 0.6:
            return 0.9  # 降低阈值10%
        else:
            return 1.0
    
    def _calculate_context_adjustment(self, context: Dict) -> float:
        """基于上下文信息计算调整因子"""
        if not context:
            return 1.0
        
        adjustment = 1.0
        
        # 光照条件调整
        lighting = context.get('lighting', 'normal')
        if lighting == 'poor':
            adjustment *= 0.9  # 光照差时放宽阈值
        elif lighting == 'excellent':
            adjustment *= 1.05  # 光照好时提高阈值
        
        # 距离调整
        distance = context.get('distance', 0.5)
        if distance > 0.8:
            adjustment *= 0.95  # 距离远时放宽阈值
        elif distance < 0.3:
            adjustment *= 1.02  # 距离近时可以更严格
        
        return adjustment
    
    def update_learning_history(self, feature_type: str, sub_feature: str,
                              similarity: float, feature_quality: float,
                              is_correct_match: bool, context: Dict = None):
        """
        更新学习历史
        
        Args:
            feature_type: 特征类型
            sub_feature: 子特征名称
            similarity: 计算出的相似度
            feature_quality: 特征质量
            is_correct_match: 是否是正确的匹配
            context: 上下文信息
        """
        timestamp = datetime.now()
        feature_key = f"{feature_type}.{sub_feature}"
        
        record = (feature_key, similarity, feature_quality, timestamp, context)
        
        if is_correct_match:
            self.learning_history['successes'].append(record)
        else:
            self.learning_history['failures'].append(record)
        
        # 更新性能统计
        self.performance_stats['total_matches'] += 1
        if is_correct_match:
            self.performance_stats['successful_matches'] += 1
        
        self.performance_stats['last_update'] = timestamp
        
        # 限制历史记录大小
        max_history = 1000
        if len(self.learning_history['successes']) > max_history:
            self.learning_history['successes'] = self.learning_history['successes'][-max_history:]
        if len(self.learning_history['failures']) > max_history:
            self.learning_history['failures'] = self.learning_history['failures'][-max_history:]
        
        # 定期保存学习数据
        if self.performance_stats['total_matches'] % 20 == 0:
            self.save_learning_data()
    
    def optimize_thresholds(self, min_samples: int = 50) -> Dict:
        """
        基于历史数据优化阈值
        
        Args:
            min_samples: 进行优化所需的最小样本数
            
        Returns:
            optimization_result: 优化结果报告
        """
        optimization_results = {}
        
        # 收集所有特征的数据
        all_features = set()
        for record in self.learning_history['successes'] + self.learning_history['failures']:
            all_features.add(record[0])
        
        for feature_key in all_features:
            # 获取该特征的历史数据
            successes = [r for r in self.learning_history['successes'] if r[0] == feature_key]
            failures = [r for r in self.learning_history['failures'] if r[0] == feature_key]
            
            if len(successes) + len(failures) < min_samples:
                continue
            
            # 计算最优阈值
            optimal_threshold = self._calculate_optimal_threshold(successes, failures)
            
            # 更新基础阈值
            feature_type, sub_feature = feature_key.split('.')
            old_threshold = self.base_thresholds.get(feature_type, {}).get(sub_feature, 0.75)
            
            if feature_type not in self.base_thresholds:
                self.base_thresholds[feature_type] = {}
            
            self.base_thresholds[feature_type][sub_feature] = optimal_threshold
            
            optimization_results[feature_key] = {
                'old_threshold': old_threshold,
                'new_threshold': optimal_threshold,
                'improvement': abs(optimal_threshold - old_threshold),
                'sample_count': len(successes) + len(failures),
                'success_rate': len(successes) / (len(successes) + len(failures))
            }
        
        # 记录优化调整
        self.learning_history['adjustments'].append({
            'timestamp': datetime.now(),
            'changes': optimization_results,
            'total_samples': self.performance_stats['total_matches']
        })
        
        # 保存优化结果
        self.save_learning_data()
        
        return optimization_results
    
    def _calculate_optimal_threshold(self, successes: List, failures: List) -> float:
        """计算最优分离阈值"""
        # 提取相似度值
        success_similarities = [record[1] for record in successes]
        failure_similarities = [record[1] for record in failures]
        
        # 寻找最佳分离点
        all_similarities = sorted(set(success_similarities + failure_similarities))
        
        best_threshold = 0.75
        best_f1_score = 0.0
        
        for threshold in all_similarities:
            # 计算混淆矩阵
            tp = sum(1 for s in success_similarities if s >= threshold)
            fp = sum(1 for f in failure_similarities if f >= threshold)
            fn = sum(1 for s in success_similarities if s < threshold)
            tn = sum(1 for f in failure_similarities if f < threshold)
            
            # 计算F1分数
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_threshold = threshold
        
        return best_threshold
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        total = self.performance_stats['total_matches']
        successful = self.performance_stats['successful_matches']
        
        if total == 0:
            return {'message': 'No matching data available'}
        
        # 计算最近7天的性能
        week_ago = datetime.now() - timedelta(days=7)
        recent_successes = [r for r in self.learning_history['successes'] if r[3] > week_ago]
        recent_failures = [r for r in self.learning_history['failures'] if r[3] > week_ago]
        recent_total = len(recent_successes) + len(recent_failures)
        
        report = {
            'overall_performance': {
                'total_matches': total,
                'success_rate': successful / total,
                'last_update': self.performance_stats['last_update'].isoformat()
            },
            'recent_performance': {
                'matches_last_7_days': recent_total,
                'success_rate_last_7_days': len(recent_successes) / recent_total if recent_total > 0 else 0
            },
            'feature_breakdown': self._get_feature_breakdown(),
            'optimization_history': len(self.learning_history['adjustments'])
        }
        
        return report
    
    def _get_feature_breakdown(self) -> Dict:
        """获取各特征的性能分解"""
        breakdown = {}
        
        # 收集所有特征
        all_features = set()
        for record in self.learning_history['successes'] + self.learning_history['failures']:
            all_features.add(record[0])
        
        for feature_key in all_features:
            successes = [r for r in self.learning_history['successes'] if r[0] == feature_key]
            failures = [r for r in self.learning_history['failures'] if r[0] == feature_key]
            total = len(successes) + len(failures)
            
            if total > 0:
                breakdown[feature_key] = {
                    'total_samples': total,
                    'success_rate': len(successes) / total,
                    'avg_success_similarity': np.mean([r[1] for r in successes]) if successes else 0,
                    'avg_failure_similarity': np.mean([r[1] for r in failures]) if failures else 0
                }
        
        return breakdown
    
    def save_learning_data(self):
        """保存学习数据到文件 - 修复日期序列化"""
        if not self.config_file:
            return
        
        try:
            # 创建目录
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # 🔧 准备保存数据，处理日期序列化
            save_data = {
                'base_thresholds': self.base_thresholds,
                'feature_weights': self.feature_weights,
                'learning_history': {
                    'successes': [],
                    'failures': [],
                    'adjustments': []
                },
                'performance_stats': {
                    'total_matches': self.performance_stats['total_matches'],
                    'successful_matches': self.performance_stats['successful_matches'],
                    'false_positives': self.performance_stats.get('false_positives', 0),
                    'false_negatives': self.performance_stats.get('false_negatives', 0),
                    'last_update': self.performance_stats['last_update'].isoformat()  # 🔧 转换为字符串
                }
            }
            
            # 🔧 处理学习历史中的日期对象
            for record in self.learning_history['successes']:
                if len(record) >= 4:
                    serialized_record = (
                        record[0],  # feature_key
                        record[1],  # similarity
                        record[2],  # feature_quality
                        record[3].isoformat() if hasattr(record[3], 'isoformat') else str(record[3]),  # timestamp
                        record[4] if len(record) > 4 else None  # context
                    )
                    save_data['learning_history']['successes'].append(serialized_record)
            
            for record in self.learning_history['failures']:
                if len(record) >= 4:
                    serialized_record = (
                        record[0],  # feature_key
                        record[1],  # similarity
                        record[2],  # feature_quality
                        record[3].isoformat() if hasattr(record[3], 'isoformat') else str(record[3]),  # timestamp
                        record[4] if len(record) > 4 else None  # context
                    )
                    save_data['learning_history']['failures'].append(serialized_record)
            
            # 🔧 处理adjustments中的日期
            for adjustment in self.learning_history['adjustments']:
                if isinstance(adjustment, dict) and 'timestamp' in adjustment:
                    adj_copy = adjustment.copy()
                    if hasattr(adj_copy['timestamp'], 'isoformat'):
                        adj_copy['timestamp'] = adj_copy['timestamp'].isoformat()
                    save_data['learning_history']['adjustments'].append(adj_copy)
                else:
                    save_data['learning_history']['adjustments'].append(adjustment)
            
            # 保存到文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
            print(f"[ADAPTIVE] 学习数据保存成功: {self.config_file}")
                
        except Exception as e:
            print(f"[ADAPTIVE] 保存学习数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def load_learning_data(self):
        """从文件加载学习数据"""
        if not self.config_file or not os.path.exists(self.config_file):
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载基础阈值
            if 'base_thresholds' in data:
                self.base_thresholds.update(data['base_thresholds'])
            
            # 加载特征权重
            if 'feature_weights' in data:
                self.feature_weights.update(data['feature_weights'])
            
            # 加载学习历史
            if 'learning_history' in data:
                history = data['learning_history']
                
                # 转换时间戳
                self.learning_history['successes'] = [
                    (r[0], r[1], r[2], datetime.fromisoformat(r[3]), r[4])
                    for r in history.get('successes', [])
                ]
                self.learning_history['failures'] = [
                    (r[0], r[1], r[2], datetime.fromisoformat(r[3]), r[4])
                    for r in history.get('failures', [])
                ]
                self.learning_history['adjustments'] = history.get('adjustments', [])
            
            # 加载性能统计
            if 'performance_stats' in data:
                stats = data['performance_stats']
                self.performance_stats.update(stats)
                if 'last_update' in stats:
                    self.performance_stats['last_update'] = datetime.fromisoformat(stats['last_update'])
            
            print(f"[ADAPTIVE] 成功加载学习数据: {len(self.learning_history['successes'])} 成功, {len(self.learning_history['failures'])} 失败")
            
        except Exception as e:
            print(f"[ADAPTIVE] 加载学习数据失败: {e}")
    
    def reset_learning_data(self):
        """重置学习数据"""
        self.learning_history = {
            'successes': [],
            'failures': [],
            'adjustments': []
        }
        
        self.performance_stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'last_update': datetime.now()
        }
        
        print("[ADAPTIVE] 学习数据已重置")
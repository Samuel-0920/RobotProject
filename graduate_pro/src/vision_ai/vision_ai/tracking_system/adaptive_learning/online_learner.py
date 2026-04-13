# adaptive_learning/online_learner.py
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

class OnlineLearner:
    """在线学习管理器 - 实现边训练边应用的自适应权重调整"""
    
    def __init__(self, user_id: str, similarity_calculator):
        """
        初始化在线学习管理器
        
        Args:
            user_id: 用户ID
            similarity_calculator: 相似度计算器引用
        """
        self.user_id = user_id
        self.similarity_calculator = similarity_calculator
        
        # 学习参数
        self.learning_rate = 0.1
        self.micro_update_frequency = 3  # 每3次反馈微调
        self.major_update_frequency = 10  # 每10次反馈重新优化
        self.max_history_size = 100  # 最大历史记录数
        
        # 数据存储
        self.feedback_history = deque(maxlen=self.max_history_size)
        self.class_specific_data = {}  # 每个类别的独立数据
        self.update_counter = 0
        
        # 文件路径
        self.data_dir = self._setup_data_directory()
        self.learning_file = os.path.join(self.data_dir, f'{user_id}_online_learning.json')
        
        # 加载历史学习数据
        self._load_learning_data()
        
        print(f"[ONLINE_LEARNER] 用户 {user_id} 在线学习管理器初始化完成")
    
    def _setup_data_directory(self) -> str:
        """设置数据目录"""
        data_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'data', 'user_profiles', 
            self.user_id, 'adaptive_learning'
        )
        os.makedirs(data_dir, exist_ok=True)
        return data_dir
    
    def update_with_feedback(self, target_id: str, tracking_record: Dict, is_correct: bool):
        """
        使用反馈更新学习数据
        
        Args:
            target_id: 目标ID
            tracking_record: 追踪记录
            is_correct: 反馈是否正确
        """
        try:
            # 提取类别信息
            class_name = self._extract_class_name(target_id)
            
            # 构建学习记录
            learning_record = self._create_learning_record(
                target_id, tracking_record, is_correct
            )
            
            # 添加到历史记录
            self.feedback_history.append(learning_record)
            
            # 更新类别特定数据
            self._update_class_data(class_name, learning_record)
            
            # 增加更新计数
            self.update_counter += 1
            
            # 根据频率决定更新策略
            if self.update_counter % self.micro_update_frequency == 0:
                self._perform_micro_update(class_name)
            
            if self.update_counter % self.major_update_frequency == 0:
                self._perform_major_update(class_name)
            
            print(f"[ONLINE_LEARNER] 已更新学习数据: {target_id} ({'正确' if is_correct else '错误'})")
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 更新反馈失败: {e}")
    
    def _extract_class_name(self, target_id: str) -> str:
        """从目标ID提取类别名称"""
        # target_id格式: "potato_0", "banana_1" 等
        return target_id.split('_')[0] if '_' in target_id else target_id
    
    def _create_learning_record(self, target_id: str, tracking_record: Dict, is_correct: bool) -> Dict:
        """创建学习记录 - 修复版"""
        try:
            # 🔧 安全地提取相似度信息
            if not tracking_record:
                print("[ONLINE_LEARNER] tracking_record 为空")
                return {
                    'target_id': target_id,
                    'class_name': self._extract_class_name(target_id),
                    'timestamp': datetime.now().isoformat(),
                    'is_correct': is_correct,
                    'overall_similarity': 0.0,
                    'feature_similarities': {},
                    'step_number': 0,
                    'tracking_confidence': 0.0
                }
            
            # 🔧 安全地访问嵌套字典
            tracking_result = tracking_record.get('tracking_result', {})
            if not tracking_result:
                # 如果没有嵌套结构，直接使用 tracking_record
                tracking_result = tracking_record
            
            similarity_breakdown = tracking_result.get('similarity_breakdown', {})
            overall_similarity = tracking_result.get('tracking_confidence', 0.0)
            
            # 创建学习记录
            record = {
                'target_id': target_id,
                'class_name': self._extract_class_name(target_id),
                'timestamp': datetime.now().isoformat(),
                'is_correct': is_correct,
                'overall_similarity': float(overall_similarity),
                'feature_similarities': similarity_breakdown,
                'step_number': tracking_record.get('step_number', 0),
                'tracking_confidence': float(overall_similarity)
            }
            
            return record
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 创建学习记录失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回基本记录
            return {
                'target_id': target_id,
                'class_name': self._extract_class_name(target_id),
                'timestamp': datetime.now().isoformat(),
                'is_correct': is_correct,
                'overall_similarity': 0.0,
                'feature_similarities': {},
                'step_number': 0,
                'tracking_confidence': 0.0
            }
        
    def _update_class_data(self, class_name: str, learning_record: Dict):
        """更新类别特定数据 - 修复版"""
        try:
            if class_name not in self.class_specific_data:
                self.class_specific_data[class_name] = {
                    'correct_records': deque(maxlen=50),
                    'incorrect_records': deque(maxlen=50),
                    'current_weights': self.similarity_calculator.feature_weights.copy(),
                    'weight_history': [],
                    'performance_metrics': {
                        'total_feedback': 0,
                        'correct_count': 0,
                        'recent_accuracy': 0.0,
                        'best_accuracy': 0.0
                    }
                }
            
            class_data = self.class_specific_data[class_name]
            
            # 🔧 安全地检查 is_correct 字段
            is_correct = learning_record.get('is_correct', False)
            
            # 添加记录到对应列表
            if is_correct:
                class_data['correct_records'].append(learning_record)
                class_data['performance_metrics']['correct_count'] += 1
            else:
                class_data['incorrect_records'].append(learning_record)
            
            # 更新性能指标
            class_data['performance_metrics']['total_feedback'] += 1
            
            # 计算最近10次的准确率
            recent_correct_records = list(class_data['correct_records'])[-10:]
            recent_incorrect_records = list(class_data['incorrect_records'])[-10:]
            recent_total = len(recent_correct_records) + len(recent_incorrect_records)
            
            if recent_total > 0:
                recent_correct_count = len(recent_correct_records)
                class_data['performance_metrics']['recent_accuracy'] = recent_correct_count / recent_total
                
                # 更新最佳准确率
                current_accuracy = class_data['performance_metrics']['recent_accuracy']
                if current_accuracy > class_data['performance_metrics']['best_accuracy']:
                    class_data['performance_metrics']['best_accuracy'] = current_accuracy
            
            print(f"[ONLINE_LEARNER] 类别数据更新完成: {class_name}")
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 更新类别数据失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _perform_micro_update(self, class_name: str):
        """执行微调更新 - 小幅调整权重"""
        try:
            if class_name not in self.class_specific_data:
                return
            
            class_data = self.class_specific_data[class_name]
            correct_records = list(class_data['correct_records'])
            incorrect_records = list(class_data['incorrect_records'])
            
            if len(correct_records) < 2 or len(incorrect_records) < 2:
                return
            
            # 分析最近的反馈模式
            recent_correct = correct_records[-3:]  # 最近3个正确案例
            recent_incorrect = incorrect_records[-3:]  # 最近3个错误案例
            
            # 计算特征表现差异
            feature_adjustments = self._calculate_feature_adjustments(
                recent_correct, recent_incorrect
            )
            
            # 应用微调
            current_weights = class_data['current_weights']
            for feature_type, adjustment in feature_adjustments.items():
                if feature_type in current_weights:
                    # 小幅调整权重
                    old_weight = current_weights[feature_type]
                    new_weight = old_weight + (adjustment * self.learning_rate * 0.1)  # 微调系数0.1
                    current_weights[feature_type] = max(0.01, min(0.8, new_weight))  # 限制范围
            
            # 归一化权重
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                current_weights = {k: v/total_weight for k, v in current_weights.items()}
                class_data['current_weights'] = current_weights
            
            # 更新相似度计算器的权重
            self.similarity_calculator.update_weights(current_weights)
            
            print(f"[ONLINE_LEARNER] 执行 {class_name} 微调更新")
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 微调更新失败: {e}")
    
    def _perform_major_update(self, class_name: str):
        """执行重大更新 - 基于历史数据重新优化权重"""
        try:
            if class_name not in self.class_specific_data:
                return
            
            class_data = self.class_specific_data[class_name]
            correct_records = list(class_data['correct_records'])
            incorrect_records = list(class_data['incorrect_records'])
            
            if len(correct_records) < 5 or len(incorrect_records) < 5:
                return
            
            # 执行全面的权重优化
            optimized_weights = self._optimize_weights_from_history(
                correct_records, incorrect_records
            )
            
            if optimized_weights:
                # 记录权重变化历史
                weight_change = {
                    'timestamp': datetime.now().isoformat(),
                    'old_weights': class_data['current_weights'].copy(),
                    'new_weights': optimized_weights.copy(),
                    'performance_before': class_data['performance_metrics']['recent_accuracy'],
                    'sample_count': len(correct_records) + len(incorrect_records)
                }
                
                class_data['weight_history'].append(weight_change)
                
                # 限制历史记录大小
                if len(class_data['weight_history']) > 20:
                    class_data['weight_history'] = class_data['weight_history'][-20:]
                
                # 应用新权重
                class_data['current_weights'] = optimized_weights
                self.similarity_calculator.update_weights(optimized_weights)
                
                print(f"[ONLINE_LEARNER] 执行 {class_name} 重大更新")
                
        except Exception as e:
            print(f"[ONLINE_LEARNER] 重大更新失败: {e}")
    
    def _calculate_feature_adjustments(self, correct_records: List, incorrect_records: List) -> Dict:
        """计算特征调整量 - 修复版，处理复杂的相似度结构"""
        try:
            adjustments = {}
            
            # 计算正确案例的特征平均表现
            correct_features = {}
            for record in correct_records:
                feature_similarities = record.get('feature_similarities', {})
                
                for feature_type, similarity_data in feature_similarities.items():
                    if feature_type not in correct_features:
                        correct_features[feature_type] = []
                    
                    # 🔧 修复：处理不同格式的相似度数据
                    similarity_value = self._extract_similarity_value(similarity_data)
                    if similarity_value is not None:
                        correct_features[feature_type].append(similarity_value)
            
            # 计算错误案例的特征平均表现
            incorrect_features = {}
            for record in incorrect_records:
                feature_similarities = record.get('feature_similarities', {})
                
                for feature_type, similarity_data in feature_similarities.items():
                    if feature_type not in incorrect_features:
                        incorrect_features[feature_type] = []
                    
                    # 🔧 修复：处理不同格式的相似度数据
                    similarity_value = self._extract_similarity_value(similarity_data)
                    if similarity_value is not None:
                        incorrect_features[feature_type].append(similarity_value)
            
            # 计算调整量
            for feature_type in set(correct_features.keys()) | set(incorrect_features.keys()):
                correct_values = correct_features.get(feature_type, [])
                incorrect_values = incorrect_features.get(feature_type, [])
                
                if correct_values and incorrect_values:
                    correct_avg = np.mean(correct_values)
                    incorrect_avg = np.mean(incorrect_values)
                    
                    # 如果正确案例中该特征表现更好，增加权重；否则减少权重
                    adjustments[feature_type] = (correct_avg - incorrect_avg) * 0.5
                elif correct_values:
                    # 只有正确案例有数据，轻微增加权重
                    adjustments[feature_type] = np.mean(correct_values) * 0.1
                elif incorrect_values:
                    # 只有错误案例有数据，轻微减少权重
                    adjustments[feature_type] = -np.mean(incorrect_values) * 0.1
            
            return adjustments
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 计算特征调整失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    # 2. 新增辅助方法：提取相似度值
    def _extract_similarity_value(self, similarity_data) -> Optional[float]:
        """
        从相似度数据中提取数值 - 处理多种格式
        """
        try:
            if similarity_data is None:
                return None
            
            # 如果是数字，直接返回
            if isinstance(similarity_data, (int, float)):
                return float(similarity_data)
            
            # 如果是numpy数值类型
            if isinstance(similarity_data, (np.integer, np.floating)):
                return float(similarity_data)
            
            # 如果是字典，尝试提取常见的数值字段
            if isinstance(similarity_data, dict):
                # 按优先级尝试不同字段
                value_fields = [
                    'final_score', 'average_score', 'contribution', 
                    'overall_score', 'score', 'value', 'similarity'
                ]
                
                for field in value_fields:
                    if field in similarity_data:
                        value = similarity_data[field]
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            return float(value)
                
                # 如果没有找到标准字段，尝试找到第一个数值
                for key, value in similarity_data.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        return float(value)
                
                return None
            
            # 如果是列表，取平均值
            if isinstance(similarity_data, (list, tuple)):
                numeric_values = []
                for item in similarity_data:
                    if isinstance(item, (int, float, np.integer, np.floating)):
                        numeric_values.append(float(item))
                
                return np.mean(numeric_values) if numeric_values else None
            
            # 尝试转换为float
            try:
                return float(similarity_data)
            except (ValueError, TypeError):
                return None
                
        except Exception as e:
            print(f"[ONLINE_LEARNER] 提取相似度值失败: {e}")
            return None

    
    def _optimize_weights_from_history(self, correct_records: List, incorrect_records: List) -> Optional[Dict]:
        """基于历史数据优化权重 - 修复版"""
        try:
            # 收集所有特征数据
            all_features = set()
            for record in correct_records + incorrect_records:
                all_features.update(record.get('feature_similarities', {}).keys())
            
            if not all_features:
                return None
            
            # 为每个特征计算分离性能指标
            feature_performances = {}
            
            for feature_type in all_features:
                correct_similarities = []
                incorrect_similarities = []
                
                # 🔧 修复：使用新的相似度值提取方法
                for record in correct_records:
                    similarity_data = record.get('feature_similarities', {}).get(feature_type)
                    similarity_value = self._extract_similarity_value(similarity_data)
                    if similarity_value is not None:
                        correct_similarities.append(similarity_value)
                
                for record in incorrect_records:
                    similarity_data = record.get('feature_similarities', {}).get(feature_type)
                    similarity_value = self._extract_similarity_value(similarity_data)
                    if similarity_value is not None:
                        incorrect_similarities.append(similarity_value)
                
                if correct_similarities and incorrect_similarities:
                    # 计算分离度 - 正确案例均值与错误案例均值的差异
                    correct_mean = np.mean(correct_similarities)
                    incorrect_mean = np.mean(incorrect_similarities)
                    separation = correct_mean - incorrect_mean
                    
                    # 计算稳定性 - 标准差的倒数
                    correct_std = np.std(correct_similarities) + 1e-6
                    incorrect_std = np.std(incorrect_similarities) + 1e-6
                    stability = 2.0 / (correct_std + incorrect_std)
                    
                    # 综合性能指标
                    performance = separation * stability
                    feature_performances[feature_type] = max(0.01, performance)
                elif correct_similarities:
                    # 只有正确样本，使用均值作为性能
                    performance = np.mean(correct_similarities)
                    feature_performances[feature_type] = max(0.01, performance)
            
            if not feature_performances:
                return None
            
            # 将性能转换为权重（归一化）
            total_performance = sum(feature_performances.values())
            if total_performance <= 0:
                return None
            
            optimized_weights = {
                feature: performance / total_performance 
                for feature, performance in feature_performances.items()
            }
            
            return optimized_weights
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] Historical weight optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_class_performance_report(self, class_name: str) -> Dict:
        """获取类别性能报告"""
        try:
            if class_name not in self.class_specific_data:
                return {'message': f'No data for class {class_name}'}
            
            class_data = self.class_specific_data[class_name]
            metrics = class_data['performance_metrics']
            
            report = {
                'class_name': class_name,
                'total_feedback': metrics['total_feedback'],
                'correct_count': metrics['correct_count'],
                'accuracy': metrics['correct_count'] / metrics['total_feedback'] if metrics['total_feedback'] > 0 else 0,
                'recent_accuracy': metrics['recent_accuracy'],
                'best_accuracy': metrics['best_accuracy'],
                'current_weights': class_data['current_weights'],
                'weight_updates': len(class_data['weight_history'])
            }
            
            return report
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 获取性能报告失败: {e}")
            return {'error': str(e)}
    
    def save_learning_data(self):
        """保存学习数据"""
        try:
            # 准备保存数据
            save_data = {
                'user_id': self.user_id,
                'last_updated': datetime.now().isoformat(),
                'update_counter': self.update_counter,
                'learning_parameters': {
                    'learning_rate': self.learning_rate,
                    'micro_update_frequency': self.micro_update_frequency,
                    'major_update_frequency': self.major_update_frequency
                },
                'class_specific_data': {},
                'recent_feedback_history': list(self.feedback_history)[-20:]  # 最近20条记录
            }
            
            # 序列化类别数据
            for class_name, class_data in self.class_specific_data.items():
                save_data['class_specific_data'][class_name] = {
                    'current_weights': class_data['current_weights'],
                    'performance_metrics': class_data['performance_metrics'],
                    'weight_history': class_data['weight_history'][-10:],  # 最近10次权重变化
                    'recent_correct_count': len(class_data['correct_records']),
                    'recent_incorrect_count': len(class_data['incorrect_records'])
                }
            
            # 保存到文件
            with open(self.learning_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"[ONLINE_LEARNER] 学习数据已保存: {self.learning_file}")
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 保存学习数据失败: {e}")
    
    def _load_learning_data(self):
        """加载历史学习数据"""
        try:
            if not os.path.exists(self.learning_file):
                print(f"[ONLINE_LEARNER] 无历史学习数据，使用默认设置")
                return
            
            with open(self.learning_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复参数
            self.update_counter = data.get('update_counter', 0)
            
            learning_params = data.get('learning_parameters', {})
            self.learning_rate = learning_params.get('learning_rate', self.learning_rate)
            self.micro_update_frequency = learning_params.get('micro_update_frequency', self.micro_update_frequency)
            self.major_update_frequency = learning_params.get('major_update_frequency', self.major_update_frequency)
            
            # 恢复类别数据
            class_data = data.get('class_specific_data', {})
            for class_name, class_info in class_data.items():
                self.class_specific_data[class_name] = {
                    'correct_records': deque(maxlen=50),
                    'incorrect_records': deque(maxlen=50),
                    'current_weights': class_info.get('current_weights', self.similarity_calculator.feature_weights.copy()),
                    'weight_history': class_info.get('weight_history', []),
                    'performance_metrics': class_info.get('performance_metrics', {
                        'total_feedback': 0,
                        'correct_count': 0,
                        'recent_accuracy': 0.0,
                        'best_accuracy': 0.0
                    })
                }
            
            print(f"[ONLINE_LEARNER] 历史学习数据加载成功，更新计数: {self.update_counter}")
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 加载学习数据失败: {e}")
    
    def get_learning_summary(self) -> Dict:
        """获取学习摘要"""
        try:
            summary = {
                'user_id': self.user_id,
                'total_updates': self.update_counter,
                'classes_learned': len(self.class_specific_data),
                'total_feedback_count': len(self.feedback_history),
                'class_performances': {}
            }
            
            for class_name in self.class_specific_data.keys():
                summary['class_performances'][class_name] = self.get_class_performance_report(class_name)
            
            return summary
            
        except Exception as e:
            print(f"[ONLINE_LEARNER] 获取学习摘要失败: {e}")
            return {'error': str(e)}
# utils/user_profile_manager.py
import os
import json
from datetime import datetime
from typing import Dict, List, Optional

class UserProfileManager:
    """用户配置管理器 - 管理多用户数据隔离和个性化设置"""
    
    def __init__(self, user_id: str, base_data_dir: str = None):
        """
        初始化用户配置管理器
        
        Args:
            user_id: 用户ID
            base_data_dir: 基础数据目录，默认为当前目录下的data文件夹
        """
        self.user_id = user_id
        
        # 设置数据目录结构
        if base_data_dir is None:
            base_data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        
        self.base_data_dir = os.path.abspath(base_data_dir)
        self.user_data_dir = os.path.join(self.base_data_dir, 'user_profiles', user_id)
        
        # 用户特定的子目录
        self.adaptive_learning_dir = os.path.join(self.user_data_dir, 'adaptive_learning')
        self.tracking_history_dir = os.path.join(self.user_data_dir, 'tracking_history')
        self.class_weights_dir = os.path.join(self.user_data_dir, 'class_weights')
        self.user_config_file = os.path.join(self.user_data_dir, 'user_config.json')
        
        # 创建目录结构
        self._create_directory_structure()
        
        # 加载或创建用户配置
        self.user_config = self._load_or_create_user_config()
        
        print(f"[USER_PROFILE] 用户 {user_id} 配置管理器初始化完成")
        print(f"[USER_PROFILE] 用户数据目录: {self.user_data_dir}")
    
    def _create_directory_structure(self):
        """创建用户数据目录结构"""
        try:
            directories = [
                self.user_data_dir,
                self.adaptive_learning_dir,
                self.tracking_history_dir,
                self.class_weights_dir
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            print(f"[USER_PROFILE] 目录结构创建完成")
            
        except Exception as e:
            print(f"[USER_PROFILE] 创建目录结构失败: {e}")
            raise
    
    def _load_or_create_user_config(self) -> Dict:
        """加载或创建用户配置"""
        try:
            if os.path.exists(self.user_config_file):
                with open(self.user_config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"[USER_PROFILE] 已有用户配置加载成功")
            else:
                config = self._create_default_user_config()
                self._save_user_config(config)
                print(f"[USER_PROFILE] 创建新用户配置")
            
            return config
            
        except Exception as e:
            print(f"[USER_PROFILE] 加载用户配置失败: {e}")
            return self._create_default_user_config()
    
    def _create_default_user_config(self) -> Dict:
        """创建默认用户配置"""
        return {
            'user_info': {
                'user_id': self.user_id,
                'created_at': datetime.now().isoformat(),
                'last_active': datetime.now().isoformat(),
                'total_sessions': 0
            },
            'preferences': {
                'feedback_timeout': 30.0,  # 反馈超时时间（秒）
                'auto_save_interval': 10,   # 自动保存间隔（步数）
                'max_tracking_steps': 10,   # 最大追踪步数
                'visualization_enabled': True
            },
            'performance_stats': {
                'total_tracking_steps': 0,
                'successful_steps': 0,
                'avg_accuracy': 0.0,
                'best_accuracy': 0.0,
                'favorite_classes': []
            },
            'adaptive_learning': {
                'enabled': True,
                'learning_rate': 0.1,
                'update_frequency': 3,  # 每3次反馈更新一次权重
                'optimization_frequency': 10  # 每10次反馈重新优化一次
            }
        }
    
    def _save_user_config(self, config: Dict):
        """保存用户配置"""
        try:
            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[USER_PROFILE] 保存用户配置失败: {e}")
    
    def update_user_activity(self):
        """更新用户活动时间"""
        try:
            self.user_config['user_info']['last_active'] = datetime.now().isoformat()
            self.user_config['user_info']['total_sessions'] += 1
            self._save_user_config(self.user_config)
            
        except Exception as e:
            print(f"[USER_PROFILE] 更新用户活动失败: {e}")
    
    def get_class_weights_file(self, class_name: str) -> str:
        """获取指定类别的权重文件路径"""
        return os.path.join(self.class_weights_dir, f'{class_name}_weights.json')
    
    def get_adaptive_learning_file(self, class_name: str) -> str:
        """获取指定类别的自适应学习文件路径"""
        return os.path.join(self.adaptive_learning_dir, f'{class_name}_learning.json')
    
    def save_class_weights(self, class_name: str, weights: Dict):
        """保存类别特定的特征权重"""
        try:
            weights_file = self.get_class_weights_file(class_name)
            
            weights_data = {
                'class_name': class_name,
                'user_id': self.user_id,
                'last_updated': datetime.now().isoformat(),
                'weights': weights,
                'update_count': self._get_current_update_count(weights_file) + 1
            }
            
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights_data, f, indent=2, ensure_ascii=False)
            
            print(f"[USER_PROFILE] 已保存 {class_name} 类别权重")
            
        except Exception as e:
            print(f"[USER_PROFILE] 保存类别权重失败: {e}")
    
    def load_class_weights(self, class_name: str) -> Optional[Dict]:
        """加载类别特定的特征权重"""
        try:
            weights_file = self.get_class_weights_file(class_name)
            
            if not os.path.exists(weights_file):
                return None
            
            with open(weights_file, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
            
            return weights_data.get('weights')
            
        except Exception as e:
            print(f"[USER_PROFILE] 加载类别权重失败: {e}")
            return None
    
    def _get_current_update_count(self, weights_file: str) -> int:
        """获取当前更新次数"""
        try:
            if not os.path.exists(weights_file):
                return 0
            
            with open(weights_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get('update_count', 0)
            
        except Exception:
            return 0
    
    def record_tracking_session(self, session_data: Dict):
        """记录追踪会话数据"""
        try:
            session_file = os.path.join(
                self.tracking_history_dir,
                f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            session_data['user_id'] = self.user_id
            session_data['session_file'] = session_file
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            # 更新性能统计
            self._update_performance_stats(session_data)
            
            print(f"[USER_PROFILE] 追踪会话已记录: {session_file}")
            
        except Exception as e:
            print(f"[USER_PROFILE] 记录追踪会话失败: {e}")
    
    def _update_performance_stats(self, session_data: Dict):
        """更新性能统计"""
        try:
            statistics = session_data.get('statistics', {})
            
            # 更新总体统计
            self.user_config['performance_stats']['total_tracking_steps'] += statistics.get('total_steps', 0)
            self.user_config['performance_stats']['successful_steps'] += statistics.get('successful_steps', 0)
            
            # 更新平均准确率
            human_accuracy = statistics.get('human_accuracy', 0)
            if human_accuracy > 0:
                current_avg = self.user_config['performance_stats']['avg_accuracy']
                total_sessions = self.user_config['user_info']['total_sessions']
                
                # 计算新的平均值
                new_avg = (current_avg * (total_sessions - 1) + human_accuracy) / total_sessions
                self.user_config['performance_stats']['avg_accuracy'] = new_avg
                
                # 更新最佳准确率
                if human_accuracy > self.user_config['performance_stats']['best_accuracy']:
                    self.user_config['performance_stats']['best_accuracy'] = human_accuracy
            
            # 更新常用类别
            target_class = session_data.get('session_info', {}).get('target_class')
            if target_class:
                favorite_classes = self.user_config['performance_stats']['favorite_classes']
                class_found = False
                
                for class_info in favorite_classes:
                    if class_info['class_name'] == target_class:
                        class_info['usage_count'] += 1
                        class_found = True
                        break
                
                if not class_found:
                    favorite_classes.append({
                        'class_name': target_class,
                        'usage_count': 1,
                        'avg_accuracy': human_accuracy
                    })
                
                # 按使用次数排序，保持前5个
                favorite_classes.sort(key=lambda x: x['usage_count'], reverse=True)
                self.user_config['performance_stats']['favorite_classes'] = favorite_classes[:5]
            
            # 保存更新的配置
            self._save_user_config(self.user_config)
            
        except Exception as e:
            print(f"[USER_PROFILE] 更新性能统计失败: {e}")
    
    def get_user_preferences(self) -> Dict:
        """获取用户偏好设置"""
        return self.user_config.get('preferences', {})
    
    def update_user_preferences(self, preferences: Dict):
        """更新用户偏好设置"""
        try:
            self.user_config['preferences'].update(preferences)
            self._save_user_config(self.user_config)
            
            print(f"[USER_PROFILE] 用户偏好已更新")
            
        except Exception as e:
            print(f"[USER_PROFILE] 更新用户偏好失败: {e}")
    
    def get_adaptive_learning_config(self) -> Dict:
        """获取自适应学习配置"""
        return self.user_config.get('adaptive_learning', {})
    
    def get_performance_summary(self) -> Dict:
        """获取用户性能摘要"""
        try:
            stats = self.user_config['performance_stats']
            user_info = self.user_config['user_info']
            
            # 计算总成功率
            total_steps = stats['total_tracking_steps']
            successful_steps = stats['successful_steps']
            success_rate = successful_steps / total_steps if total_steps > 0 else 0
            
            summary = {
                'user_id': self.user_id,
                'total_sessions': user_info['total_sessions'],
                'total_tracking_steps': total_steps,
                'overall_success_rate': success_rate,
                'avg_human_accuracy': stats['avg_accuracy'],
                'best_accuracy': stats['best_accuracy'],
                'favorite_classes': stats['favorite_classes'],
                'member_since': user_info['created_at'],
                'last_active': user_info['last_active']
            }
            
            return summary
            
        except Exception as e:
            print(f"[USER_PROFILE] 获取性能摘要失败: {e}")
            return {}
    
    def list_user_sessions(self) -> List[str]:
        """列出用户的历史会话文件"""
        try:
            if not os.path.exists(self.tracking_history_dir):
                return []
            
            session_files = []
            for filename in os.listdir(self.tracking_history_dir):
                if filename.startswith('session_') and filename.endswith('.json'):
                    session_files.append(os.path.join(self.tracking_history_dir, filename))
            
            # 按时间排序
            session_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            return session_files
            
        except Exception as e:
            print(f"[USER_PROFILE] 列出用户会话失败: {e}")
            return []
    
    def get_class_learning_history(self, class_name: str) -> Optional[Dict]:
        """获取指定类别的学习历史"""
        try:
            learning_file = self.get_adaptive_learning_file(class_name)
            
            if not os.path.exists(learning_file):
                return None
            
            with open(learning_file, 'r', encoding='utf-8') as f:
                learning_data = json.load(f)
            
            return learning_data
            
        except Exception as e:
            print(f"[USER_PROFILE] 获取类别学习历史失败: {e}")
            return None
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """清理旧的数据文件"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            cleaned_count = 0
            
            # 清理历史会话文件
            for session_file in self.list_user_sessions():
                if os.path.getmtime(session_file) < cutoff_time:
                    os.remove(session_file)
                    cleaned_count += 1
            
            # 清理过期的学习数据（可选）
            # 这里可以添加更多清理逻辑
            
            if cleaned_count > 0:
                print(f"[USER_PROFILE] 清理了 {cleaned_count} 个过期文件")
            
        except Exception as e:
            print(f"[USER_PROFILE] 清理旧数据失败: {e}")
    
    def export_user_data(self, export_path: str) -> bool:
        """导出用户数据"""
        try:
            import shutil
            
            # 创建导出目录
            export_user_dir = os.path.join(export_path, f'user_{self.user_id}_export')
            os.makedirs(export_user_dir, exist_ok=True)
            
            # 复制用户数据目录
            shutil.copytree(
                self.user_data_dir,
                os.path.join(export_user_dir, 'user_data'),
                dirs_exist_ok=True
            )
            
            # 创建导出摘要
            export_summary = {
                'export_time': datetime.now().isoformat(),
                'user_id': self.user_id,
                'data_directory': self.user_data_dir,
                'performance_summary': self.get_performance_summary()
            }
            
            with open(os.path.join(export_user_dir, 'export_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(export_summary, f, indent=2, ensure_ascii=False)
            
            print(f"[USER_PROFILE] 用户数据已导出到: {export_user_dir}")
            return True
            
        except Exception as e:
            print(f"[USER_PROFILE] 导出用户数据失败: {e}")
            return False
    
    def get_user_data_size(self) -> Dict[str, int]:
        """获取用户数据大小统计"""
        try:
            def get_dir_size(directory):
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(directory):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                return total_size
            
            size_info = {
                'total_size_bytes': get_dir_size(self.user_data_dir),
                'adaptive_learning_size': get_dir_size(self.adaptive_learning_dir),
                'tracking_history_size': get_dir_size(self.tracking_history_dir),
                'class_weights_size': get_dir_size(self.class_weights_dir),
                'session_count': len(self.list_user_sessions())
            }
            
            # 转换为可读格式
            size_info['total_size_mb'] = size_info['total_size_bytes'] / (1024 * 1024)
            
            return size_info
            
        except Exception as e:
            print(f"[USER_PROFILE] 获取数据大小失败: {e}")
            return {}
    
    def __str__(self):
        """字符串表示"""
        return f"UserProfileManager(user_id='{self.user_id}', data_dir='{self.user_data_dir}')"
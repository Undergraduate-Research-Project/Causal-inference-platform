import hashlib
import logging

class LocalAuthDB:
    """本地用户认证数据库"""
    
    def __init__(self, config=None):
        # 本地用户数据
        self.local_users = {
            "admin": {
                "password": "21232f297a57a5a743894a0e4a801fc3",  # admin的MD5
                "user_id": "admin",
                "email": "admin@localhost.com",
                "role": "admin"
            },
            "test": {
                "password": "098f6bcd4621d373cade4e832627b4f6",  # test的MD5
                "user_id": "local_test_001", 
                "email": "test@localhost.com",
                "role": "user"
            },
            "user": {
                "password": "ee11cbb19052e40b07aac0ca060c23ee",  # user的MD5
                "user_id": "local_user_001",
                "email": "user@localhost.com", 
                "role": "user"
            }
        }
    
    def authenticate_user(self, username, password):
        """用户认证方法"""
        try:
            # 对密码进行MD5加密以匹配存储格式
            password_hash = hashlib.md5(password.encode()).hexdigest()
            
            if username in self.local_users:
                stored_user = self.local_users[username]
                if stored_user['password'] == password_hash:
                    logging.info(f"用户认证成功: {username}")
                    return {
                        'success': True,
                        'user': {
                            'username': username,
                            'user_id': stored_user['user_id'],
                            'email': stored_user.get('email', ''),
                            'role': stored_user.get('role', 'user'),
                            'auth_method': 'local'
                        }
                    }
                else:
                    return {
                        'success': False,
                        'message': '用户名或密码错误'
                    }
            else:
                return {
                    'success': False,
                    'message': '用户名或密码错误'
                }
        except Exception as e:
            logging.error(f"用户认证失败: {str(e)}")
            return {
                'success': False,
                'message': '认证过程中发生错误',
                'error': str(e)
            }
    
    def register_user(self, username, password, additional_data=None):
        """用户注册方法 - 不支持注册，只能使用预设账户"""
        logging.warning(f"用户尝试注册: {username}，但系统不支持注册")
        return {
            'success': False,
            'message': '系统不支持注册新用户。请使用预设账户登录。',
            'local_accounts_hint': '可用账户: admin/admin, test/test, user/user'
        }
    
    def get_user_by_username(self, username):
        """根据用户名获取用户信息"""
        try:
            if username in self.local_users:
                user_data = self.local_users[username]
                logging.info(f"获取用户信息成功: {username}")
                return {
                    'success': True,
                    'user': {
                        'username': username,
                        'user_id': user_data['user_id'],
                        'email': user_data.get('email', ''),
                        'role': user_data.get('role', 'user'),
                        'auth_method': 'local'
                    }
                }
            else:
                return {
                    'success': False,
                    'message': '用户不存在'
                }
        except Exception as e:
            logging.error(f"获取用户信息失败: {str(e)}")
            return {
                'success': False,
                'message': '获取用户信息时发生错误',
                'error': str(e)
            }

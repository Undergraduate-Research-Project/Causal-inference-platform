import http.client
import json
import hashlib
import logging
import socket

class HuobanDB:
    """火伴API数据库操作类"""
    
    def __init__(self, config=None):
        if config:
            self.host = getattr(config, 'HUOBAN_API_HOST', "api.huoban.com")
            self.table_id = getattr(config, 'HUOBAN_TABLE_ID', "2100000066422526")
            self.api_key = getattr(config, 'HUOBAN_API_KEY', "9pTFg4AxdFRKsTb1y9667Rq1uoF2kCAtRsjXmVEe")
        else:
            # 使用默认配置
            self.host = "api.huoban.com"
            self.table_id = "2100000066422526"
            self.api_key = "9pTFg4AxdFRKsTb1y9667Rq1uoF2kCAtRsjXmVEe"
            
        self.headers = {
            'Open-Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        
        # 本地硬编码用户数据 - 作为备份认证方案
        self.local_users = {
            "admin": {
                "password": "password",  # admin的MD5
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
    
    def _make_request(self, method, endpoint, payload=None, timeout=10):
        """发起HTTP请求的通用方法"""
        try:
            conn = http.client.HTTPSConnection(self.host, timeout=timeout)
            
            if payload:
                payload_json = json.dumps(payload)
                conn.request(method, endpoint, payload_json, self.headers)
            else:
                conn.request(method, endpoint, headers=self.headers)
            
            res = conn.getresponse()
            data = res.read()
            result = json.loads(data.decode("utf-8"))
            conn.close()
            print("API请求成功")
            print(result)
            return {
                'success': res.status == 200,
                'status_code': res.status,
                'data': result
            }
        except (http.client.HTTPException, socket.timeout, socket.gaierror, 
                ConnectionRefusedError, OSError) as e:
            logging.error(f"网络连接失败: {str(e)}")
            return {
                'success': False,
                'error': f'网络连接失败: {str(e)}',
                'network_error': True
            }
        except Exception as e:
            logging.error(f"API请求失败: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'network_error': False
            }
    
    def _authenticate_local(self, username, password):
        """本地用户认证 - 备份方案"""
        try:
            # 对密码进行MD5加密以匹配存储格式
            password_hash = hashlib.md5(password.encode()).hexdigest()
            
            if username in self.local_users:
                stored_user = self.local_users[username]
                if stored_user['password'] == password_hash:
                    logging.info(f"本地用户认证成功: {username}")
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
            logging.error(f"本地用户认证失败: {str(e)}")
            return {
                'success': False,
                'message': '本地认证过程中发生错误',
                'error': str(e)
            }
    
    def authenticate_user(self, username, password):
        """用户认证方法 - 优先使用火伴API，失败时使用本地验证"""
        # 首先尝试火伴API认证
        try:
            logging.info(f"尝试使用火伴云API认证用户: {username}")
            
            # 对密码进行MD5加密
            password_hash = password
            
            # 构建查询条件，查找用户名匹配的记录
            payload = {
                "table_id": self.table_id,
                "filter": {
                    "and": [
                        {
                            "field": "username",  # 假设用户名字段名为username
                            "query": {
                                "eq": [username]
                            }
                        }
                    ]
                },
                "limit": 1,
                "offset": 0,
                "with_field_config": 0
            }
            
            response = self._make_request("POST", "/openapi/v1/item/list", payload)
            
            # 检查是否是网络错误
            if not response['success'] and response.get('network_error'):
                logging.warning(f"火伴云API连接失败，切换到本地认证: {response.get('error')}")
                return self._authenticate_local(username, password)
            
            if response['success']:
                data = response['data']
                if data.get('data') and len(data['data']) > 0:
                    # 找到用户，验证密码
                    user_info = data['data']['items'][0]
                    stored_password = user_info.get('fields', {}).get('2200000533373883', '')
                    
                    if stored_password == password_hash:
                        logging.info(f"火伴云API认证成功: {username}")
                        return {
                            'success': True,
                            'user': {
                                'username': username,
                                'user_id': user_info.get('item_id'),
                                'data': user_info.get('data', {}),
                                'auth_method': 'huoban'
                            }
                        }
                    else:
                        return {
                            'success': False,
                            'message': '用户名或密码错误'
                        }
                else:
                    # 用户在火伴云中不存在，尝试本地认证
                    logging.info(f"用户在火伴云中不存在，尝试本地认证: {username}")
                    return self._authenticate_local(username, password)
            else:
                # API请求失败，尝试本地认证
                logging.warning(f"火伴云API请求失败，切换到本地认证: {response.get('error')}")
                return self._authenticate_local(username, password)
                
        except Exception as e:
            logging.error(f"火伴云用户认证异常: {str(e)}")
            # 发生异常时，尝试本地认证
            logging.info(f"火伴云认证异常，切换到本地认证: {username}")
            return self._authenticate_local(username, password)
    
    def register_user(self, username, password, additional_data=None):
        """用户注册方法 - 优先使用火伴API，失败时提示使用本地账户"""
        try:
            logging.info(f"尝试使用火伴云API注册用户: {username}")
            
            # 首先检查用户名是否已存在
            check_payload = {
                "table_id": self.table_id,
                "filter": {
                    "and": [
                        {
                            "field": "username",
                            "query": {
                                "eq": [username]
                            }
                        }
                    ]
                },
                "limit": 1,
                "offset": 0,
                "with_field_config": 0
            }
            
            check_response = self._make_request("POST", "/openapi/v1/item/list", check_payload)
            
            # 检查是否是网络错误
            if not check_response['success'] and check_response.get('network_error'):
                logging.warning(f"火伴云API连接失败，无法注册新用户: {check_response.get('error')}")
                return {
                    'success': False,
                    'message': '网络连接失败，无法注册新用户。请联系管理员或使用现有本地账户登录。',
                    'local_accounts_hint': '可用的本地测试账户: admin/admin, test/test, user/user'
                }
            
        except Exception as e:
            logging.error(f"用户注册失败: {str(e)}")
            return {
                'success': False,
                'message': '网络连接失败，无法注册新用户。请联系管理员或使用现有本地账户登录。',
                'local_accounts_hint': '可用的本地测试账户: admin/admin, test/test, user/user',
                'error': str(e)
            }
            logging.error(f"用户注册失败: {str(e)}")
            return {
                'success': False,
                'message': '网络连接失败，无法注册新用户。请联系管理员或使用现有本地账户登录。',
                'local_accounts_hint': '可用的本地测试账户: admin/admin, test/test, user/user',
                'error': str(e)
            }
            
            if check_response['success']:
                data = check_response['data']
                if data.get('items') and len(data['items']) > 0:
                    return {
                        'success': False,
                        'message': '用户名已存在'
                    }
            
            # 用户名不存在，可以注册
            password_hash = hashlib.md5(password.encode()).hexdigest()
            
            # 构建注册数据
            register_data = {
                "username": username,
                "password_hash": password_hash
            }
            
            # 如果有额外数据，添加到注册数据中
            if additional_data:
                register_data.update(additional_data)
            
            register_payload = {
                "table_id": self.table_id,
                "data": register_data
            }
            
            response = self._make_request("POST", "/openapi/v1/item/create", register_payload)
            
            # 检查是否是网络错误
            if not response['success'] and response.get('network_error'):
                logging.warning(f"火伴云API连接失败，无法注册新用户: {response.get('error')}")
                return {
                    'success': False,
                    'message': '网络连接失败，无法注册新用户。请联系管理员或使用现有本地账户登录。',
                    'local_accounts_hint': '可用的本地测试账户: admin/admin, test/test, user/user'
                }
            
            if response['success']:
                logging.info(f"火伴云API注册成功: {username}")
                return {
                    'success': True,
                    'message': '注册成功',
                    'user_id': response['data'].get('item_id')
                }
            else:
                return {
                    'success': False,
                    'message': '注册失败',
                    'error': response.get('error', '未知错误')
                }
                
        except Exception as e:
            logging.error(f"用户注册失败: {str(e)}")
            return {
                'success': False,
                'message': '网络连接失败，无法注册新用户。请联系管理员或使用现有本地账户登录。',
                'local_accounts_hint': '可用的本地测试账户: admin/admin, test/test, user/user',
                'error': str(e)
            }
    def get_user_by_username(self, username):
        """根据用户名获取用户信息 - 优先使用火伴API，失败时使用本地查询"""
        try:
            logging.info(f"尝试使用火伴云API获取用户信息: {username}")
            
            payload = {
                "table_id": self.table_id,
                "filter": {
                    "and": [
                        {
                            "field": "username",
                            "query": {
                                "eq": [username]
                            }
                        }
                    ]
                },
                "limit": 1,
                "offset": 0,
                "with_field_config": 0
            }
            
            response = self._make_request("POST", "/openapi/v1/item/list", payload)
            
            # 检查是否是网络错误
            if not response['success'] and response.get('network_error'):
                logging.warning(f"火伴云API连接失败，尝试本地查询: {response.get('error')}")
                # 尝试本地用户查询
                if username in self.local_users:
                    user_data = self.local_users[username]
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
            
            if response['success']:
                data = response['data']
                if data.get('items') and len(data['items']) > 0:
                    user_info = data['items'][0]
                    logging.info(f"火伴云API获取用户信息成功: {username}")
                    return {
                        'success': True,
                        'user': {
                            'username': username,
                            'user_id': user_info.get('item_id'),
                            'data': user_info.get('data', {}),
                            'auth_method': 'huoban'
                        }
                    }
                else:
                    # 火伴云中没有用户，尝试本地查询
                    if username in self.local_users:
                        user_data = self.local_users[username]
                        logging.info(f"火伴云中无用户，使用本地用户信息: {username}")
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
            else:
                # API请求失败，尝试本地查询
                logging.warning(f"火伴云API请求失败，尝试本地查询: {response.get('error')}")
                if username in self.local_users:
                    user_data = self.local_users[username]
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
            # 发生异常时，尝试本地查询
            if username in self.local_users:
                user_data = self.local_users[username]
                logging.info(f"异常情况下使用本地用户信息: {username}")
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
                    'message': '获取用户信息时发生错误',
                    'error': str(e)
                }

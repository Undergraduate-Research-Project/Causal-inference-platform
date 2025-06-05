import http.client
import json
import hashlib
import logging

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
    
    def _make_request(self, method, endpoint, payload=None):
        """发起HTTP请求的通用方法"""
        try:
            conn = http.client.HTTPSConnection(self.host)
            
            if payload:
                payload_json = json.dumps(payload)
                conn.request(method, endpoint, payload_json, self.headers)
            else:
                conn.request(method, endpoint, headers=self.headers)
            
            res = conn.getresponse()
            data = res.read()
            result = json.loads(data.decode("utf-8"))
            conn.close()
            print("finish!!!!!")
            print(result)
            return {
                'success': res.status == 200,
                'status_code': res.status,
                'data': result
            }
        except Exception as e:
            logging.error(f"API请求失败: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def authenticate_user(self, username, password):
        """用户认证方法 - 从火伴API查询用户信息"""
        try:
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
            print("*************")
            print(payload)
            response = self._make_request("POST", "/openapi/v1/item/list", payload)
            print("responce")
            
            if response['success']:
                print("success!!!")
                data = response['data']
                print(data)
                print(len(data['data']))
                if data.get('data') and len(data['data']) > 0:
                    # 找到用户，验证密码
                    user_info = data['data']['items'][0]
                    stored_password = user_info.get('fields', {}).get('2200000533373883', '')
                    print("*********")
                    print(stored_password)
                    print(password_hash)
                    
                    if stored_password == password_hash:
                        return {
                            'success': True,
                            'user': {
                                'username': username,
                                'user_id': user_info.get('item_id'),
                                'data': user_info.get('data', {})
                            }
                        }
                    else:
                        return {
                            'success': False,
                            'message': '用户名或密码错误'
                        }
                else:
                    # 用户不存在
                    return {
                        'success': False,
                        'message': '用户名或密码错误'
                    }
            else:
                return {
                    'success': False,
                    'message': 'API请求失败',
                    'error': response.get('error', '未知错误')
                }
                
        except Exception as e:
            logging.error(f"用户认证失败: {str(e)}")
            return {
                'success': False,
                'message': '认证过程中发生错误',
                'error': str(e)
            }
    
    def register_user(self, username, password, additional_data=None):
        """用户注册方法"""
        try:
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
            
            if response['success']:
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
                'message': '注册过程中发生错误',
                'error': str(e)
            }
    
    def get_user_by_username(self, username):
        """根据用户名获取用户信息"""
        try:
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
            
            if response['success']:
                data = response['data']
                if data.get('items') and len(data['items']) > 0:
                    user_info = data['items'][0]
                    return {
                        'success': True,
                        'user': {
                            'username': username,
                            'user_id': user_info.get('item_id'),
                            'data': user_info.get('data', {})
                        }
                    }
                else:
                    return {
                        'success': False,
                        'message': '用户不存在'
                    }
            else:
                return {
                    'success': False,
                    'message': 'API请求失败',
                    'error': response.get('error', '未知错误')
                }
                
        except Exception as e:
            logging.error(f"获取用户信息失败: {str(e)}")
            return {
                'success': False,
                'message': '获取用户信息时发生错误',
                'error': str(e)
            }

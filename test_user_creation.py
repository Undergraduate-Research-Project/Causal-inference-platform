"""
测试用户创建脚本
用于在火伴数据库中创建测试用户
"""

from api.huoban_db import HuobanDB
import sys

def create_test_user():
    """创建测试用户"""
    db = HuobanDB()
    
    # 创建测试用户
    test_users = [
        {
            'username': 'admin',
            'password': 'password',
            'email': 'admin@test.com'
        },
        {
            'username': 'test_user',
            'password': 'test123',
            'email': 'test@test.com'
        }
    ]
    
    for user_data in test_users:
        print(f"创建用户: {user_data['username']}")
        result = db.register_user(
            username=user_data['username'],
            password=user_data['password'],
            additional_data={'email': user_data['email']}
        )
        
        if result['success']:
            print(f"✅ 用户 {user_data['username']} 创建成功")
            print(f"   用户ID: {result.get('user_id', 'N/A')}")
        else:
            print(f"❌ 用户 {user_data['username']} 创建失败: {result['message']}")
        print("-" * 50)

def test_user_login():
    """测试用户登录"""
    db = HuobanDB()
    
    # 测试登录
    test_credentials = [
        {'username': 'admin', 'password': 'password'},
        {'username': 'test_user', 'password': 'test123'},
        {'username': 'admin', 'password': 'wrong_password'},  # 错误密码测试
    ]
    
    for creds in test_credentials:
        print(f"测试登录: {creds['username']}")
        result = db.authenticate_user(creds['username'], creds['password'])
        
        if result['success']:
            print(f"✅ 登录成功")
            print(f"   用户ID: {result['user']['user_id']}")
            print(f"   用户数据: {result['user']['data']}")
        else:
            print(f"❌ 登录失败: {result['message']}")
        print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            create_test_user()
        elif sys.argv[1] == "test":
            test_user_login()
        else:
            print("使用方法:")
            print("  python test_user_creation.py create  # 创建测试用户")
            print("  python test_user_creation.py test    # 测试用户登录")
    else:
        print("使用方法:")
        print("  python test_user_creation.py create  # 创建测试用户")
        print("  python test_user_creation.py test    # 测试用户登录")

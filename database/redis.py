# @Time : 2023/2/6 16:37 
# @Author : CaoXiang
# @Description: 数据库交互接口
import redis
import os
from config.conf_loader import YamlConfigLoader


def get_redis(loader: YamlConfigLoader):
    try:
        host = loader.attempt_load_param("feature_service_ip")
    except:
        host = os.getenv("feature_service_ip", "127.0.0.1")
    try:
        port = loader.attempt_load_param("feature_service_port")
    except:
        port = os.getenv("feature_service_port", "5555")
    pool = redis.ConnectionPool(host=host, port=port, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    return r










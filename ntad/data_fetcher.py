"""
MITRE ATT&CK 数据获取模块
从 MITRE 官方 GitHub 下载 Enterprise ATT&CK 的 STIX 2.0 数据文件
"""

import os
import time
import requests

# 数据源 URL
ENTERPRISE_ATTACK_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

# 缓存配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_FILE = os.path.join(DATA_DIR, "enterprise-attack.json")
CACHE_MAX_AGE = 7 * 24 * 3600  # 7 天（秒）


def _is_cache_valid() -> bool:
    """检查缓存文件是否存在且未过期"""
    if not os.path.exists(CACHE_FILE):
        return False
    mtime = os.path.getmtime(CACHE_FILE)
    age = time.time() - mtime
    return age < CACHE_MAX_AGE


def fetch_enterprise_attack() -> dict:
    """
    获取 Enterprise ATT&CK STIX 数据

    优先使用本地缓存（7 天内有效），否则从 GitHub 下载最新版本。
    返回解析后的 JSON 字典。
    """
    if _is_cache_valid():
        print("使用缓存数据")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return __import__("json").load(f)

    print("正在从 MITRE GitHub 下载最新数据...")
    try:
        response = requests.get(ENTERPRISE_ATTACK_URL, timeout=60)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError("网络请求超时，请检查网络连接后重试")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"网络请求失败: {e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(response.text)
    except PermissionError:
        raise RuntimeError(f"没有写入权限: {CACHE_FILE}")

    print("下载完成，已缓存到本地")
    return response.json()


if __name__ == "__main__":
    data = fetch_enterprise_attack()
    print(f"数据类型: {data.get('type', '未知')}")
    print(f"攻击对象数量: {len(data.get('objects', []))}")

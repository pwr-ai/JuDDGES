import requests
from requests.adapters import HTTPAdapter
from tor_python_easy.tor_control_port_client import TorControlPortClient
from tor_python_easy.tor_socks_get_ip_client import TorSocksGetIpClient


class TorClient:
    def __init__(self, password: str, tor_port: int = 9050, tor_config_port: int = 9051) -> None:
        self.proxy_config = {
            "http": f"socks5://localhost:{tor_port}",
            "https": f"socks5://localhost:{tor_port}",
        }
        self.ip_client = TorSocksGetIpClient(self.proxy_config)
        self.tor_control_port_client = TorControlPortClient("127.0.0.1", tor_config_port, password)

    def get_current_ip(self, session: requests.Session) -> str:
        result = session.get("http://ip-api.com/json/").json()
        assert isinstance(result, dict)
        return result["query"]

    def get_session(self, **kwargs) -> requests.Session:
        adapter = HTTPAdapter(**kwargs)
        http = requests.Session()
        http.proxies.update(self.tor_client.proxy_config)
        http.mount("https://", adapter)
        http.mount("http://", adapter)
        return http



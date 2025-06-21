import json
import httpx
from typing import Optional, Dict, Any, List


class McpClient:
    """MCP客户端，处理所有与MCP相关的网络请求"""
    
    def __init__(self, api_host: str, api_key: str, device_id: str, timeout: int = 10):
        self.api_host = api_host
        self.api_key = api_key
        self.device_id = device_id
        self.timeout = timeout
        self.base_url = f"{api_host}"
        
        # 通用请求头
        self.headers = {
            'X-API-Key': api_key,
            'X-Device-ID': device_id
        }
    
    def fetch_device_states(self) -> Dict[str, Any]:
        """
        获取IoT设备状态
        
        Returns:
            Dict[str, Any]: API响应数据
        """
        url = f"{self.base_url}/open/iot/device/deviceState"
        
        try:
            response = httpx.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.TimeoutException:
            return {
                "code": -1,
                "message": f"API request timed out for IoT device states URL: {url}",
                "error_type": "timeout"
            }
        except httpx.RequestError as e:
            return {
                "code": -1,
                "message": f"API request failed for IoT device states: {e}",
                "error_type": "request_error"
            }
        except json.JSONDecodeError as e:
            return {
                "code": -1,
                "message": f"Failed to decode JSON response from IoT device states API: {e}",
                "error_type": "json_decode_error"
            }
        except Exception as e:
            return {
                "code": -1,
                "message": f"An unexpected error occurred while fetching device states: {e}",
                "error_type": "unexpected_error"
            }
    
    def fetch_mcp_tools(self) -> Dict[str, Any]:
        """
        获取MCP支持的工具列表
        
        Returns:
            Dict[str, Any]: API响应数据
        """
        url = f"{self.base_url}/open/iot/device/mcpTools"
        
        try:
            response = httpx.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.TimeoutException:
            return {
                "code": -1,
                "message": f"API request timed out for MCP tools URL: {url}",
                "error_type": "timeout"
            }
        except httpx.RequestError as e:
            return {
                "code": -1,
                "message": f"API request failed for MCP tools: {e}",
                "error_type": "request_error"
            }
        except json.JSONDecodeError as e:
            return {
                "code": -1,
                "message": f"Failed to decode JSON response from MCP tools API: {e}",
                "error_type": "json_decode_error"
            }
        except Exception as e:
            return {
                "code": -1,
                "message": f"An unexpected error occurred while fetching MCP tools: {e}",
                "error_type": "unexpected_error"
            }
    
    def execute_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行MCP工具
        
        Args:
            tool_name (str): 工具名称
            params (Dict[str, Any]): 工具参数
            
        Returns:
            Dict[str, Any]: API响应数据
        """
        url = f"{self.base_url}/open/iot/device/executeMcpTool"
        
        # 构建请求数据
        payload = {
            "toolName": tool_name,
            "params": params
        }
        
        # 设置Content-Type头
        headers = {
            **self.headers,
            'Content-Type': 'application/json'
        }
        
        try:
            response = httpx.post(
                url, 
                headers=headers, 
                data=json.dumps(payload),
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except httpx.TimeoutException:
            return {
                "code": -1,
                "message": "MCP tool execution request timed out",
                "error_type": "timeout"
            }
        except httpx.RequestError as e:
            return {
                "code": -1,
                "message": f"MCP tool execution request failed: {str(e)}",
                "error_type": "request_error"
            }
        except json.JSONDecodeError:
            return {
                "code": -1,
                "message": "Invalid response from MCP tool execution API",
                "error_type": "json_decode_error"
            }
        except Exception as e:
            return {
                "code": -1,
                "message": f"Error executing MCP tool: {str(e)}",
                "error_type": "unexpected_error"
            }


def create_mcp_client(api_host: str, api_key: str, device_id: str, timeout: int = 10) -> Optional[McpClient]:
    """
    创建MCP客户端的工厂函数
    
    Args:
        api_host (str): API主机地址
        api_key (str): API密钥
        device_id (str): 设备ID
        timeout (int): 请求超时时间，默认10秒
        
    Returns:
        Optional[McpClient]: 如果api_key和device_id都提供则返回客户端实例，否则返回None
    """
    if not api_key or not device_id:
        return None
    
    return McpClient(api_host, api_key, device_id, timeout) 
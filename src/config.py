import json
import os

class Config:
    def __init__(self):
        self.load_config()
    
    def load_config(self,):
        with open('config.json', 'r') as f:
            config = json.load(f)

            # 加载 LLM 相关配置
            llm_config = config.get('llm', {})
            self.llm_model_type = llm_config.get('model_type', 'ollama')
            self.ollama_model_name = llm_config.get('ollama_model_name', 'llama3')
            self.ollama_api_url = llm_config.get('ollama_api_url', 'http://localhost:11434/api/chat')
            self.openai_model_name = llm_config.get('openai_model_name', 'gpt-4o-mini')
            # load server config
            server_config = config.get('server',{})
            self.port = server_config.get('port',7860)
            self.name = server_config.get('name',"0.0.0.0")

def main():

    config = Config()  # 创建配置实例

    print(config.llm_model_type)
    print(config.openai_model_name)
    print(config.ollama_model_name)
    print(config.ollama_api_url)
    print(config.name)
    print(config.port)

if __name__ == "__main__":
    main()
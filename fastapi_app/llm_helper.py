from openai import OpenAI

class OpenAILLM():
    def __init__(self, model_name = "gpt-4o"):
        self.api_key = None
        self.model_name = model_name
        self.agent = None

    def login_and_create_client(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key = api_key)
        response = self.client.chat.completions.create(model =  self.model_name, messages=[
            {"role": "user", "content": "Hi! if this message reached you, just reply 'LLM Live!'"} ]).choices[0].message.content.strip()
        return response


    def hit_llm(self, msg):
        response = self.client.chat.completions.create(model = self.model_name, messages=[{"role": "user", "content": msg} ])
        return response.choices[0].message.content.strip()
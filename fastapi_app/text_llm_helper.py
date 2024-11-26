from openai import OpenAI
import aiohttp
from fastapi import HTTPException

class OpenAILLM():
    def __init__(self, model_name = "gpt-4o"):
        self.api_key = None
        self.model_name = model_name
        self.agent = None
        self.history = {"prev_transcriptions":[], "prev_answers":[]}
        self.system_prompt = ""

    def login_and_create_client(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key = api_key)
        answer = self.client.chat.completions.create(model =  self.model_name, messages=[
            {"role": "user", "content": "Hi! if this message reached you, just reply 'LLM Live!'"} ]).choices[0].message.content.strip()
        
        return answer


    def hit_llm(self, msg):
        """
        TO-DO: Add async support. find ways
        """
        messages=[
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": msg}]

        response = self.client.chat.completions.create(model = self.model_name, messages = messages)
        result = response.choices[0].message.content.strip()
        self.history["prev_transcriptions"].append(msg)
        self.history["prev_answers"].append(result)

        return result
    

async def get_rag(query: str, rag_url:str) -> dict:
    """
    Sends query to RAG for context
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(rag_url, json={'query': query}, timeout=20) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail=f"RAG service returned status {response.status}")
                return await response.json()
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in getting RAG: {str(e)}")
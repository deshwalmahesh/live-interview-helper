from openai import OpenAI

SYS_PROMPT = """You are a helpful assistant in providing answer in the field of System Design, AI, ML,  NLP, Computer Vision, statistics, Python, Hugging-Face, scikit-learn and Programming questions in Python.

You are given an transcription of an interview. Identitity of Interviewer and the Candidate is not clear. You have to look at the transcription and figure who is the interviewer and who is the answerer. 

Then you have to find whether there is any question asked by the interviewer or not. If there is, then provide the answer to help the candidate.

The question will either be related to programming which you have to solve in Python or you will be given a question related to the filed of AI, ML, Deep Learning, NLP, stats or System design etc.

Note: There can be  minimum info about the problem and query. There can be spelling mistakes and half information. You need to figure out what the actual question that is asked. It'll always be in the field of AI, ML, Deep Learning, System Design etc. 

After figuring out the query, write the answer to that question with bullet points.

If it is a process of there are sub steps, write all of the steps.

Write all the info in detail and if there needs to be, use the example too in the end explaining the working."""


class OpenAILLM():
    def __init__(self, model_name = "gpt-4o"):
        self.api_key = None
        self.model_name = model_name
        self.agent = None
        self.system_prompt = SYS_PROMPT
        self.history = {"prev_transcriptions":[], "prev_answers":[]}

    def login_and_create_client(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key = api_key)
        response = self.client.chat.completions.create(model =  self.model_name, messages=[
            {"role": "user", "content": "Hi! if this message reached you, just reply 'LLM Live!'"} ]).choices[0].message.content.strip()
        return response


    def hit_llm(self, msg):
        messages=[
        {"role": "system", "content": self.system_prompt},
        {"role": "user", "content": msg}]

        response = self.client.chat.completions.create(model = self.model_name, messages = messages)
        result = response.choices[0].message.content.strip()
        self.history["prev_transcriptions"].append(msg)
        self.history["prev_answers"].append(result)

        return result
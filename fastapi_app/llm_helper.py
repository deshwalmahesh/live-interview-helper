from openai import OpenAI

class OpenAILLM():
    def __init__(self, model_name = "gpt-4o"):
        self.api_key = None
        self.model_name = model_name
        self.agent = None
        self.history = {"prev_transcriptions":[], "prev_answers":[]}
        self.system_prompt = """You are a helpful assistant in providing helpful answers in a technical interview.

You are given an transcription of an interview. Identitity of Interviewer and the Candidate is not clear so you have to look at the transcription and figure who is the interviewer and who is the answerer. 

Then you have to find whether there is any question asked by the interviewer or not. If there is, then provide the answer to EACH and EVERY question asked within that span to help the candidate.

NOTE: There can be  minimum info about the problem and query as it's a transcription and isn't perfect and there can be half or broken information. You need to figure out what the actual technical questions are.

After figuring out the questions, write the systenatic answer to that question with headings and bullet points in least words possible.

If it is a process of there are sub steps, write all of the steps. Use the example too in the end explaining the working.

MUST FOLLOW: 
1. Remember that the questions will only be technical from the field of System Design, AI, ML,  NLP, Computer Vision, statistics, Python, Hugging-Face, scikit-learn, Opencv and Programming questions in Python.
2. Don't write opening or closing statements or your thoughts about the conversation. 
3. Provide only the answer to the questions which are to the point and be concise so that you provide answer in least words possible. 
4. Use Markdown format to properly display the heading, sub heading, bullet points, spacing etc etc and use MathJax to render equations properly
5. Try to infer from the words which topic resembles from System Design, AI, ML,  NLP, Computer Vision, statistics, Python, Hugging-Face, scikit-learn, Opencv and Programming questions in Python if tere is a question apart from these topics. These could be synonym or homophones too"""


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
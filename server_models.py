from pydantic import BaseModel

class Prompt(BaseModel):
    prompt:str
class Prompt_Language(BaseModel):
    prompt:str
    language:str



class IntentAnalysis(BaseModel):
    query:str
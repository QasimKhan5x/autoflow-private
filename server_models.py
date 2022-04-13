from typing import Optional, List
from pydantic import BaseModel

class Prompt(BaseModel):
    prompt:str
class Prompt_Context(BaseModel):
    prompt:str
    context:Optional[str]
class Prompt_Language(BaseModel):
    prompt:str
    language:str
class Prompt_Language_Context(BaseModel):
    prompt:str
    language:str
    context:Optional[str]

class Code_Task(BaseModel):
    code:str
    task:str

class Code_Task_Context(BaseModel):
    code:str
    task:str
    context:Optional[str]

class API_Req(BaseModel):
    api_name:str
    task:str
    params:str
    token: Optional[str]
class IntentAnalysis(BaseModel):
    query:str

class QueryInfo(BaseModel):
    task:str
    tableName:str
    columnName:str

class JsonData(BaseModel):
    fp:str
    content:str

class SearchCode(BaseModel):
    prompt:str
    code: List[JsonData]
    recreate:bool

class SearchCode_Language(BaseModel):
    prompt:str
    code: List[JsonData]
    recreate:bool
    language:Optional[str]

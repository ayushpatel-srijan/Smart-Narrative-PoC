import logging
import requests
import os
from uuid import uuid4
from typing import Any, List, Mapping, Optional
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Set
from pydantic import Extra, Field, root_validator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
import requests
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#ds-internal-dev@srijan.net
# $rij4n@123
os.environ["API_USERNAME"] = "ds-internal-dev@srijan.net"
os.environ["API_PASSWORD"] = "$rij4n@123"
from langchain.llms.base import LLM
#from api import *

from typing import Any, Union
from asyncio.queues import Queue
from langchain.callbacks.base import AsyncCallbackHandler
from typing import Any, Dict, List, Optional
from langchain.schema import LLMResult
import json
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler
DEFAULT_ANSWER_PREFIX_TOKENS = ["\n", "AI", ":"]

#----------

class CustomAsyncCallBackHandler(AsyncCallbackHandler):
    queue: Queue
    
    
    def __init__(self,queue:Queue, answer_prefix_tokens: Optional[List[str]] = None) -> None:
        super().__init__()
        if answer_prefix_tokens is None:
            answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        self.answer_prefix_tokens = answer_prefix_tokens
        self.last_tokens = [""] * len(answer_prefix_tokens)
        self.answer_reached = False
        self.queue=queue


    async def put_message(self,json_str):
        await self.queue.put(json.dumps(json_str))
        await self.queue.join()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
         # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.last_tokens == self.answer_prefix_tokens:
            self.answer_reached = True
            # Do not print the last token in answer_prefix_tokens,
            # as it's not part of the answer yet
            return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            response = str(token)
            await self.put_message(response)

        

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        if self.answer_reached:
            response = "[DONE]"
            await self.put_message(response)

    async def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""


#---------------------

class Inference:
    
    def __init__(self) -> None:
        self.SESSION_ID = None
        self.headers = None

    def _login(self):
        login_url = f"{Config().API_BASE_URL}/login"
        payload = {
            "username": os.environ.get("API_USERNAME"),
            "password": os.environ.get("API_PASSWORD"),
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try :
            response = requests.post(login_url, data=payload, headers=headers)
            json_data = response.json()
        except Exception as e:
            print(f"Exception in login : {e}")
            response = requests.post(login_url, data=payload, headers=headers)
            print(response.status_code)
            json_data = response.json()
            

        #print(json_data)
        bearer_token = json_data["access_token"]
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"}
        logger.info("Logged in successfully")
        print("Logged in successfully")
        return self.headers
    

    def _validate_session_id(self, url, payload):
        response, validation = None, False
        try:
            print("posting request...")
            #start_time = time.time()
            response = requests.post(f"{url}/{self.SESSION_ID}", json=payload, headers=self.headers, timeout=40)
            print(response.status_code)
            #elapsed_time = time.time() - start_time
            if response.status_code == 200:
                validation = True

            return response, validation
        
        except requests.exceptions.Timeout:
            #if elapsed_time > 40:
            print("Request timed out. Resending the request...")
            response, validation = self._validate_session_id(url, payload)
            return response, validation
        except Exception as e:
            logger.info(f"Error After Validation: {response}")
            return response, validation

        
    def _get_session_id(self, chat_name, headers):
        print("getting session_id...")
        self._login()
        url = f"{Config().API_BASE_URL}/chat/new?chat_name={chat_name}"
        response = requests.post(
                f"{url}" ,headers=headers
            )
        response_data = response.json()
        return response_data['session_id']
    
    def _nlq_to_pandas(self, nlq_question):
        self._login()
        input_prompt = nlq_question
        url = f"{Config().API_BASE_URL}/chat/interact"
        payload = {"message": input_prompt}
        print("got prompt..")
        response, validation = self._validate_session_id(url=url, payload=payload)
        print("got response..")
        if not validation:
            print(response.status_code)
            print("getting new session id")
            self.SESSION_ID = self._get_session_id(chat_name=uuid4().hex, headers=self.headers)
            print(f"Logged in afresh, new Session ID : {self.SESSION_ID} ")
            logger.info(f"Logged in afresh, new Session ID : {self.SESSION_ID} ")
            response, validation = self._validate_session_id(url=url, payload=payload)
        try :

            response_data = response.json()
            
        except Exception as e:
            print("Error : ",e)
            print("response : " ,response)

        print(response_data)
        return response_data['content']
       
        # Add any additional logic or processing here
        
class CustomLLM(LLM):
    i: object
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self,prompt,stop=None,run_manager: Optional[CustomAsyncCallBackHandler] = None, **kwargs):
        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)
        text = ""
        try :
            temp =self.i._nlq_to_pandas(prompt)
        except Exception as e:
            print("Exception in 'temp =self.i._nlq_to_pandas(prompt) ' : ",e)
            os.system("ssh -fNT CHATGPT_API")
            temp =self.i._nlq_to_pandas(prompt)

        for token in temp:
            if text_callback:
                text_callback(token)
            text += token
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
        #try:
        #    return self.i._nlq_to_pandas(prompt)
        #except:
        #    os.system("ssh -fNT CHATGPT_API")
        #    return self.i._nlq_to_pandas(prompt)
class Config:
    def __init__(self) -> None:
        self.API_BASE_URL = "http://localhost:8080"

llm = CustomLLM(i=Inference())
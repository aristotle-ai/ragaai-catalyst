import litellm
import json
import requests
import os
from google import genai
from google.genai.types import GenerateContentConfig
from typing import Optional, List, Dict, Any
import logging
logger = logging.getLogger('LiteLLM')
logger.setLevel(logging.ERROR)

class GuardExecutor:

    def __init__(self,guard_manager,input_deployment_id = None,output_deployment_id=None,field_map={}):
        self.field_map = field_map
        self.guard_manager = guard_manager
        try:
            if input_deployment_id:
                self.input_deployment_id = input_deployment_id
                self.input_deployment_details = self.guard_manager.get_deployment(input_deployment_id)
            if output_deployment_id:
                self.output_deployment_id = output_deployment_id
                self.output_deployment_details = self.guard_manager.get_deployment(output_deployment_id)
            if input_deployment_id and output_deployment_id:
                # check if 2 deployments are mapped to same dataset
                if self.input_deployment_details['data']['datasetId'] != self.output_deployment_details['data']['datasetId']:
                    logger.error('Input deployment and output deployment should be mapped to same dataset')
            for guardrail in self.input_deployment_details['data']['guardrailsResponse']:
                maps = guardrail['metricSpec']['config']['mappings']
                for _map in maps:
                    if _map['schemaName']=='Response':
                        logger.error('Response field should be mapped only in output guardrails')
        except Exception as e:
            logger.error(str(e))
        self.base_url = guard_manager.base_url
        for key in field_map.keys():
            if key not in ['prompt','context','response','instruction']:
                logger.error('Keys in field map should be in ["prompt","context","response","instruction"]')
        self.id_2_doc = {}

    def execute_deployment(self, deployment_id, payload):
        api = self.base_url + f'/guardrail/deployment/{deployment_id}/ingest'
        payload = json.dumps(payload)
        headers = {
            'x-project-id': str(self.guard_manager.project_id),
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
        }
        try:
            response = requests.request("POST", api, headers=headers, data=payload,timeout=self.guard_manager.timeout)
        except Exception as e:
            logger.error(f'Failed running guardrail: {str(e)}')
            return None
        if response.status_code!=200:
            logger.error(f'Error in running deployment {response.json()["message"]}')
            return None
        if response.json()['success']:
            return response.json()
        else:
            print(response.json()['message'])
            return None

    def llm_executor(self,prompt,model_params,llm_caller):
        messages = [{
                    'role':'user',
                    'content':prompt
                    }]
        if llm_caller == 'litellm':
            model_params['messages'] = messages
            response = litellm.completion(**model_params)
            return response['choices'][0].message.content
        elif llm_caller == 'genai':
            genai_client = genai.Client(api_key=os.getenv('GENAI_API_KEY'))
            model_params['messages'] = messages
            response = genai_client.models.generate(**model_params)
            return response.text
        else:
            print(f"{llm_caller} not supported currently, use litellm as llm caller")
        '''
        elif llm_caller == 'anthropic':
            response = anthropic.completion(prompt=messages, **model_params)
            return response['completion']
        elif llm_caller == 'langchain':
            response = langchain.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_caller == 'azure_openai':
            response = azure_openai.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_caller == 'aws_bedrock':
            response = aws_bedrock.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_caller == 'meta':
            response = meta.completion(prompt=messages, **model_params)
            return response['choices'][0].text
        elif llm_csller == 'llamaindex':
            response = llamaindex.completion(prompt=messages, **model_params)
            return response['choices'][0].text'''

    def set_input_params(self, prompt: None, context: None, instruction: None,  **kwargs):
        if 'latest' not in self.id_2_doc:
            self.id_2_doc['latest'] = {}
        if prompt:
            self.id_2_doc['latest']['prompt'] = prompt
        if context:
            self.id_2_doc['latest']['context'] = context
        if instruction:
            self.id_2_doc['latest']['instruction'] = instruction

    
    def __call__(self,prompt,prompt_params,model_params,llm_caller='litellm'):
        '''for key in self.field_map:
            if key not in ['prompt','response']:
                if self.field_map[key] not in prompt_params:
                    raise ValueError(f'{key} added as field map but not passed as prompt parameter')
        context_var = self.field_map.get('context',None)
        prompt = None
        for msg in messages:
            if 'role' in msg:
                if msg['role'] == 'user':
                    prompt = msg['content']
                    if not context_var:
                        msg['content'] += '\n' + prompt_params[context_var]
        doc = dict()
        doc['prompt'] = prompt
        doc['context'] = prompt_params[context_var]'''
        
        # Run the input guardrails
        alternate_response,input_deployment_response = self.execute_input_guardrails(prompt,prompt_params)
        if input_deployment_response and input_deployment_response['data']['status'].lower() == 'fail':
            return alternate_response, None, input_deployment_response
        
        # activate only guardrails that require response
        try:
            llm_response = self.llm_executor(prompt,model_params,llm_caller)
        except Exception as e:
            print('Error in running llm:',str(e))
            return None, None, input_deployment_response
        if 'instruction' in self.field_map:
            instruction = prompt_params[self.field_map['instruction']]
        alternate_op_response,output_deployment_response = self.execute_output_guardrails(llm_response)
        if output_deployment_response and output_deployment_response['data']['status'].lower() == 'fail':
            return alternate_op_response,llm_response,output_deployment_response
        else:
            return None,llm_response,output_deployment_response

    def set_variables(self,prompt,prompt_params):
        for key in self.field_map:
            if key not in ['prompt', 'response']:
                if self.field_map[key] not in prompt_params:
                    logger.error(f'{key} added as field map but not passed as prompt parameter')
        context_var = self.field_map.get('context', None)
        
        doc = dict()
        doc['prompt'] = prompt
        doc['context'] = prompt_params[context_var]
        if 'instruction' in self.field_map:
            instruction = prompt_params[self.field_map['instruction']]
            doc['instruction'] = instruction
        return doc

    def execute_input_guardrails(self, prompt, prompt_params):
        doc = self.set_variables(prompt,prompt_params)
        deployment_response = self.execute_deployment(self.input_deployment_id,doc)
        trace_id = deployment_response['data']['results'][0]['executionId']
        self.id_2_doc[trace_id] = doc
        if deployment_response and deployment_response['data']['status'].lower() == 'fail':
            return deployment_response['data']['alternateResponse'], deployment_response
        elif deployment_response:
            return None, deployment_response

    def execute_output_guardrails(self, llm_response: str, trace_id:str, prompt=None, prompt_params=None) -> None:
        if not prompt: # user has not passed input
            if trace_id not in self.id_2_doc:
                logger.error(f'No input doc found for trace_id: {trace_id}')
            else:
                doc = self.id_2_doc[trace_id]
                doc['response'] = llm_response
        else:
            doc = self.set_variables(prompt,prompt_params)
            doc['response'] = llm_response
        doc['traceId'] = trace_id
        deployment_response = self.execute_deployment(self.output_deployment_id,doc)
        del self.id_2_doc[trace_id]
        if deployment_response and deployment_response['data']['status'].lower() == 'fail':
            return deployment_response['data']['alternateResponse'], deployment_response
        elif deployment_response:
            return None, deployment_response

    @staticmethod
    def execute_deployment_static(deployment_id, payload, gdm):
        api = gdm.base_url + f'/guardrail/deployment/{deployment_id}/ingest'
        payload = json.dumps(payload)
        headers = {
            'x-project-id': str(gdm.project_id),
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {os.getenv("RAGAAI_CATALYST_TOKEN")}'
        }
        try:
            response = requests.request("POST", api, headers=headers, data=payload,timeout=gdm.timeout)
        except Exception as e:
            logger.error(f'Failed running guardrail: {str(e)}')
            return None
        if response.status_code!=200:
            logger.error(f'Error in running deployment {response.json()["message"]}')
            return None
        if response.json()['success']:
            return response.json()
        else:
            print(response.json()['message'])
            return None

    @staticmethod
    def execute_input_guardrail(input_deployment_id, prompt, context, gdm):
        doc = {
                'prompt':prompt,
                'context':context
               }
        deployment_response = GuardExecutor.execute_deployment_static(input_deployment_id, doc, gdm)
        if deployment_response and deployment_response['data']['status'].lower() == 'fail':
            return deployment_response['data']['alternateResponse'], deployment_response
        elif deployment_response:
            return None, deployment_response
        else:
            return None, None
    
    @staticmethod
    def execute_output_guardrail(output_deployment_id, prompt, context, response, gdm, trace_id):
        doc = {
                'prompt':prompt,
                'context':context,
                'response':response,
                'traceId':trace_id
               }
        deployment_response = GuardExecutor.execute_deployment_static(output_deployment_id, doc, gdm)
        if deployment_response and deployment_response['data']['status'].lower() == 'fail':
            return deployment_response['data']['alternateResponse'], deployment_response
        elif deployment_response:
            return None, deployment_response
        else:
            return None, None
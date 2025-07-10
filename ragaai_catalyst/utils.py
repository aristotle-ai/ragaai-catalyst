import requests
import logging
from ragaai_catalyst.proto import trace_pb2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def get_token():
#     access_key = os.getenv("RAGA-TRACER_ACCESS_KEY")
#     secret_key = os.getenv("RAGA-TRACER_SECRET_KEY")
#     headers = {"Content-Type": "application/json"}
#     json_data = {
#         "accessKey": access_key,
#         "secretKey": secret_key,
#     }
#     response = requests.post(
#         "https://backend.dev3.ragaai.ai/api/token", headers=headers, json=json_data
#     )
#     token_response = response.json()
#     token = token_response.get("data", {}).get("token", None)
#     if token is not None:
#         os.environ["RAGAAI_CATALYST_TOKEN"] = token
#     return token_response


def response_checker(response, context=""):
    """
    Checks the response status code and logs the appropriate message.

    Args:
        response (requests.Response): The response object.
        context (str, optional): The context in which the response is being checked. Defaults to "".

    Returns:
        int: The status code of the response.
    """
    logger.debug(f" Response : {response}")
    if response.status_code == 200:
        logger.debug(
            f"{context} - Successful Request. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 400:
        logger.debug(
            f"{context} - Bad Request. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 401:
        logger.debug(
            f"{context} - Unauthorized. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 403:
        logger.debug(
            f"{context} - Forbidden. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 404:
        logger.debug(
            f"{context} - Not Found. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 408:
        logger.debug(
            f"{context} - Request Timeout. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 500:
        logger.debug(
            f"{context} - Internal Server Error. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 502:
        logger.debug(
            f"{context} - Bad Gateway. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 503:
        logger.debug(
            f"{context} - Service Unavailable. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    elif response.status_code == 504:
        logger.debug(
            f"{context} - Gateway Timeout. Response Code: {response.status_code}, Response Text: {(response.json()['message'])}"
        )
        return response.status_code
    else:
        error_message = response.json().get("message", "No message returned.")
        logger.debug(
            f"{context}{response.reason}. Response Code: {response.status_code}, Response Text: {error_message}"
        )
        return response.status_code

def json_to_protobuf(json_data):
    trace = trace_pb2.Trace()
    trace.id = json_data['id']
    trace.project_name = json_data['project_name']
    trace.start_time = json_data['start_time']
    trace.end_time = json_data['end_time']
    trace.tracer_type = json_data['tracer_type']

    # Metadata
    metadata = trace.metadata
    metadata.tokens.prompt_tokens = json_data['metadata']['tokens']['prompt_tokens']
    metadata.tokens.completion_tokens = json_data['metadata']['tokens']['completion_tokens']
    metadata.tokens.total_tokens = json_data['metadata']['tokens']['total_tokens']

    metadata.cost.input_cost = json_data['metadata']['cost']['input_cost']
    metadata.cost.output_cost = json_data['metadata']['cost']['output_cost']
    metadata.cost.total_cost = json_data['metadata']['cost']['total_cost']

    system_info = metadata.system_info
    system_info.id = json_data['metadata']['system_info']['id']
    system_info.os.name = json_data['metadata']['system_info']['os']['name']
    system_info.os.version = json_data['metadata']['system_info']['os']['version']
    system_info.os.platform = json_data['metadata']['system_info']['os']['platform']
    system_info.os.kernel_version = json_data['metadata']['system_info']['os']['kernel_version']
    system_info.environment.name = json_data['metadata']['system_info']['environment']['name']
    system_info.environment.version = json_data['metadata']['system_info']['environment']['version']
    system_info.environment.packages.extend(json_data['metadata']['system_info']['environment']['packages'])
    system_info.environment.env_path = json_data['metadata']['system_info']['environment']['env_path']
    system_info.environment.command_to_run = json_data['metadata']['system_info']['environment']['command_to_run']
    system_info.source_code = json_data['metadata']['system_info']['source_code']

    resources = metadata.resources
    resources.cpu.info.name = json_data['metadata']['resources']['cpu']['info']['name']
    resources.cpu.info.cores = json_data['metadata']['resources']['cpu']['info']['cores']
    resources.cpu.info.threads = json_data['metadata']['resources']['cpu']['info']['threads']
    resources.cpu.interval = json_data['metadata']['resources']['cpu']['interval']
    resources.cpu.values.extend(json_data['metadata']['resources']['cpu']['values'])

    resources.memory.info.total = json_data['metadata']['resources']['memory']['info']['total']
    resources.memory.info.free = json_data['metadata']['resources']['memory']['info']['free']
    resources.memory.interval = json_data['metadata']['resources']['memory']['interval']
    resources.memory.values.extend(json_data['metadata']['resources']['memory']['values'])

    resources.disk.info.total = json_data['metadata']['resources']['disk']['info']['total']
    resources.disk.info.free = json_data['metadata']['resources']['disk']['info']['free']
    resources.disk.interval = json_data['metadata']['resources']['disk']['interval']
    resources.disk.read.extend(json_data['metadata']['resources']['disk']['read'])
    resources.disk.write.extend(json_data['metadata']['resources']['disk']['write'])

    resources.network.info.upload_speed = json_data['metadata']['resources']['network']['info']['upload_speed']
    resources.network.info.download_speed = json_data['metadata']['resources']['network']['info']['download_speed']
    resources.network.interval = json_data['metadata']['resources']['network']['interval']
    resources.network.uploads.extend(json_data['metadata']['resources']['network']['uploads'])
    resources.network.downloads.extend(json_data['metadata']['resources']['network']['downloads'])

    # Spans
    for span_data_json in json_data['data']:
        span_data = trace.data.add()
        span_data.start_time = span_data_json['start_time']
        span_data.end_time = span_data_json['end_time']
        for span_json in span_data_json['spans']:
            span = span_data.spans.add()
            span.name = span_json['name']
            span.context.trace_id = span_json['context']['trace_id']
            span.context.span_id = span_json['context']['span_id']
            span.context.trace_state = span_json['context']['trace_state']
            span.kind = span_json['kind']
            span.parent_id = span_json.get('parent_id') or ""
            span.start_time = span_json['start_time']
            span.end_time = span_json['end_time']
            span.status.status_code = span_json['status']['status_code']
            for key, value in span_json['attributes'].items():
                span.attributes[key] = str(value)
            for key, value in span_json['resource']['attributes'].items():
                span.resource.attributes[key] = str(value)
            span.resource.schema_url = span_json['resource']['schema_url']
            span.name_occurrences = span_json['name_occurrences']
            span.hash_id = span_json['hash_id']

    # Workflow steps
    for step_json in json_data['workflow']:
        step = trace.workflow.add()
        step.id = step_json['id']
        step.span_id = step_json['span_id'] if step_json['span_id'] else ''
        step.interaction_type = step_json['interaction_type']
        step.name = step_json['name']
        step.content = step_json['content'] if step_json['content'] else ''
        step.timestamp = step_json['timestamp']
        step.error = step_json['error'] if step_json['error'] else ''

    return trace

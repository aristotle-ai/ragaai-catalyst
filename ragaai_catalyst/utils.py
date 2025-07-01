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
    try:
        trace = trace_pb2.Trace()
        trace.id = get_safe_str(json_data, 'id')
        trace.external_id = get_safe_str(json_data, 'external_id')
        trace.project_name = get_safe_str(json_data, "project_name")
        trace.start_time = get_safe_str(json_data, "start_time")
        trace.end_time = get_safe_str(json_data, "end_time")
        trace.tracer_type = get_safe_str(json_data, "tracer_type")


        # Metadata
        metadata = trace.metadata

        metadata_json = json_data.get('metadata', {})
        tokens = metadata_json.get('tokens', {})

        metadata.tokens.prompt_tokens = tokens.get('prompt_tokens', 0)
        metadata.tokens.completion_tokens = tokens.get('completion_tokens', 0)
        metadata.tokens.total_tokens = tokens.get('total_tokens', 0)

        cost = metadata_json.get('cost', {})
        metadata.cost.input_cost = cost.get('input_cost', 0.0)
        metadata.cost.output_cost = cost.get('output_cost', 0.0)
        metadata.cost.total_cost = cost.get('total_cost', 0.0)

        custom_fields = json_data.get("metadata", {}).get("custom_fields", {})
        if isinstance(custom_fields, dict):
            for key, value in custom_fields.items():
                metadata.custom_fields[key] = str(value)


        system_info = metadata.system_info
        system_info_json = metadata_json.get('system_info', {})

        system_info.id = get_safe_str(system_info_json, 'id')

        os_info = system_info_json.get('os', {})
        system_info.os.name = get_safe_str(os_info, 'name')
        system_info.os.version = get_safe_str(os_info, 'version')
        system_info.os.platform = get_safe_str(os_info, 'platform')
        system_info.os.kernel_version = get_safe_str(os_info, 'kernel_version')

        env = system_info_json.get('environment', {})
        system_info.environment.name = get_safe_str(env, 'name')
        system_info.environment.version = get_safe_str(env, 'version')
        system_info.environment.env_path = get_safe_str(env, 'env_path')
        system_info.environment.command_to_run = get_safe_str(env, 'command_to_run')
        if isinstance(env.get('packages'), list):
            system_info.environment.packages.extend(env['packages'])
        system_info.source_code = get_safe_str(system_info_json, 'source_code')

        resources = metadata.resources

        resources_json = metadata_json.get('resources', {})
        cpu = resources_json.get('cpu', {})
        cpu_info = cpu.get('info', {})
        resources.cpu.info.name = get_safe_str(cpu_info, 'name')
        resources.cpu.info.cores = cpu_info.get('cores', 0)
        resources.cpu.info.threads = cpu_info.get('threads', 0)
        resources.cpu.interval = get_safe_str(cpu, 'interval')
        if isinstance(cpu.get('values'), list):
            resources.cpu.values.extend(cpu['values'])

        memory = resources_json.get('memory', {})
        mem_info = memory.get('info', {})
        resources.memory.info.total = mem_info.get('total', 0.0)
        resources.memory.info.free = mem_info.get('free', 0.0)
        trace.metadata.resources.memory.interval = get_safe_str(memory, 'interval')
        if isinstance(memory.get('values'), list):
            resources.memory.values.extend(memory['values'])

        disk = resources_json.get('disk', {})
        disk_info = disk.get('info', {})
        resources.disk.info.total = disk_info.get('total', 0)
        resources.disk.info.free = disk_info.get('free', 0)
        resources.disk.interval = disk.get('interval', "")
        if isinstance(disk.get('read'), list):
            resources.disk.read.extend(disk['read'])
        if isinstance(disk.get('write'), list):
            resources.disk.write.extend(disk['write'])

        network = resources_json.get('network', {})
        net_info = network.get('info', {})
        resources.network.info.upload_speed = net_info.get('upload_speed', 0.0)
        resources.network.info.download_speed = net_info.get('download_speed', 0.0)
        resources.network.interval = get_safe_str(network, 'interval')
        if isinstance(network.get('uploads'), list):
            resources.network.uploads.extend(network['uploads'])
        if isinstance(network.get('downloads'), list):
            trace.metadata.resources.network.downloads.extend(network['downloads'])
            resources.network.downloads.extend(network['downloads'])

        # Spans
        for span_data_json in json_data['data']:
            span_data = trace.data.add()
            span_data.start_time = span_data_json['start_time']
            span_data.end_time = span_data_json['end_time']
            for span_json in span_data_json['spans']:
                span = span_data.spans.add()
                span.name = get_safe_str(span_json, 'name')
                context = span_json.get('context', {})
                span.context.trace_id = get_safe_str(context, 'trace_id')
                span.context.span_id = get_safe_str(context, 'span_id')
                span.context.trace_state = get_safe_str(context, 'trace_state')
                span.kind = get_safe_str(span_json, 'kind')
                span.parent_id = get_safe_str(span_json, 'parent_id')
                span.start_time = get_safe_str(span_json, 'start_time')
                span.end_time = get_safe_str(span_json, 'end_time')
                status = span_json.get('status', {})
                span.status.status_code = get_safe_str(status, 'status_code')
                span.hash_id = get_safe_str(span_json, 'hash_id')

                attributes = span_json.get('attributes', {})
                if isinstance(attributes, dict):
                    for key, value in attributes.items():
                        span.attributes[key] = str(value) if value is not None else ''

                resource = span_json.get("resource", {})
                resource_attributes = resource.get("attributes", {})
                if isinstance(resource_attributes, dict):
                    for key, value in resource_attributes.items():
                        span.resource.attributes[key] = str(value) if value is not None else ''

                span.resource.schema_url = resource.get("schema_url") or ""
        # Workflow steps
        for step_json in json_data['workflow']:
            step = trace.workflow.add()
            step.id = get_safe_str(step_json, 'id')
            step.span_id = get_safe_str(step_json, 'span_id')
            step.interaction_type = get_safe_str(step_json, 'interaction_type')
            step.name = get_safe_str(step_json, 'name')
            step.content = get_safe_str(step_json, 'content')
            step.timestamp = get_safe_str(step_json, 'timestamp')
            step.error = get_safe_str(step_json, 'error')

        print("✅ Successfully transformed JSON to Protobuf")
        return trace
    except Exception as e:
        print(f"❌ Error transforming JSON to Protobuf: {e}")
        raise


def get_safe_str(json_obj, key, default=""):
    """
    Safely retrieve a string from a JSON object.
    Logs if key is missing or value is None/empty.
    """
    value = json_obj.get(key, default)
    if value in [None, ""]:
        logger.debug(f"[STR] Missing or empty key '{key}', defaulting to '{default}'")
        return default
    return str(value)
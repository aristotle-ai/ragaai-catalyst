import json
import os
import requests
import logging
from importlib import resources
from dataclasses import asdict

logger = logging.getLogger(__name__)

def convert_usage_to_dict(usage):
    # Initialize the token_usage dictionary with default values
    token_usage = {
        "input": 0,
        "completion": 0,
        "reasoning": 0,  # Default reasoning tokens to 0 unless specified
    }

    if usage:
        if isinstance(usage, dict):
            # Access usage data as dictionary keys
            token_usage["input"] = usage.get("prompt_tokens", 0)
            token_usage["completion"] = usage.get("completion_tokens", 0)
            # If reasoning tokens are provided, adjust accordingly
            token_usage["reasoning"] = usage.get("reasoning_tokens", 0)
        else:
            # Handle the case where usage is not a dictionary
            # This could be an object with attributes, or something else
            try:
                token_usage["input"] = getattr(usage, "prompt_tokens", 0)
                token_usage["completion"] = getattr(usage, "completion_tokens", 0)
                token_usage["reasoning"] = getattr(usage, "reasoning_tokens", 0)
            except AttributeError:
                # If attributes are not found, log or handle the error as needed
                logger.warning(f"Unexpected usage type: {type(usage)}")

    return token_usage


def calculate_cost(
    token_usage,
    input_cost_per_token=0.0,
    output_cost_per_token=0.0,
    reasoning_cost_per_token=0.0,
):
    input_tokens = token_usage.get("prompt_tokens", 0)
    output_tokens = token_usage.get("completion_tokens", 0)
    reasoning_tokens = token_usage.get("reasoning_tokens", 0)

    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * output_cost_per_token
    reasoning_cost = reasoning_tokens * reasoning_cost_per_token

    total_cost = input_cost + output_cost + reasoning_cost

    return {
        "input": input_cost,
        "completion": output_cost,
        "reasoning": reasoning_cost,
        "total": total_cost,
    }

def log_event(event_data, log_file_path):
    event_data = asdict(event_data)
    with open(log_file_path, "a") as f:
        f.write(json.dumps(event_data) + "\n")


def process_child_interactions(child, interaction_id, interactions):
    """
    Helper method to process child interactions recursively.
    
    Args:
        child (dict): The child span to process
        interaction_id (int): Current interaction ID
        interactions (list): List of interactions to append to
        
    Returns:
        int: Next interaction ID to use
    """
    child_type = child.get("type")

    if child_type == "tool":
        # Tool call start
        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": "tool_call_start",
                "name": child.get("name"),
                "content": {
                    "parameters": [
                        child.get("data", {}).get("input", {}).get("args"),
                        child.get("data", {}).get("input", {}).get("kwargs"),
                    ]
                },
                "timestamp": child.get("start_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

        # Tool call end
        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": "tool_call_end",
                "name": child.get("name"),
                "content": {
                    "returns": child.get("data", {}).get("output"),
                },
                "timestamp": child.get("end_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

    elif child_type == "llm":
        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": "llm_call_start",
                "name": child.get("name"),
                "content": {
                    "prompt": child.get("data", {}).get("input"),
                },
                "timestamp": child.get("start_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": "llm_call_end",
                "name": child.get("name"),
                "content": {"response": child.get("data", {}).get("output")},
                "timestamp": child.get("end_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

    elif child_type == "agent":
        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": "agent_call_start",
                "name": child.get("name"),
                "content": None,
                "timestamp": child.get("start_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

        # Process nested children recursively
        if "children" in child.get("data", {}):
            for nested_child in child["data"]["children"]:
                interaction_id = process_child_interactions(
                    nested_child, interaction_id, interactions
                )

        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": "agent_call_end",
                "name": child.get("name"),
                "content": child.get("data", {}).get("output"),
                "timestamp": child.get("end_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

    else:
        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": f"{child_type}_call_start",
                "name": child.get("name"),
                "content": child.get("data", {}),
                "timestamp": child.get("start_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

        interactions.append(
            {
                "id": str(interaction_id),
                "span_id": child.get("id"),
                "interaction_type": f"{child_type}_call_end",
                "name": child.get("name"),
                "content": child.get("data", {}),
                "timestamp": child.get("end_time"),
                "error": child.get("error"),
            }
        )
        interaction_id += 1

    # Process additional interactions and network calls
    if "interactions" in child:
        for interaction in child["interactions"]:
            interaction["id"] = str(interaction_id)
            interaction["span_id"] = child.get("id")
            interaction["error"] = None
            interactions.append(interaction)
            interaction_id += 1

    if "network_calls" in child:
        for child_network_call in child["network_calls"]:
            network_call = {}
            network_call["id"] = str(interaction_id)
            network_call["span_id"] = child.get("id")
            network_call["interaction_type"] = "network_call"
            network_call["name"] = None
            network_call["content"] = {
                "request": {
                    "url": child_network_call.get("url"),
                    "method": child_network_call.get("method"),
                    "headers": child_network_call.get("headers"),
                },
                "response": {
                    "status_code": child_network_call.get("status_code"),
                    "headers": child_network_call.get("response_headers"),
                    "body": child_network_call.get("response_body"),
                },
            }
            network_call["timestamp"] = child_network_call.get("start_time")
            network_call["error"] = child_network_call.get("error")
            interactions.append(network_call)
            interaction_id += 1

    return interaction_id


def format_interactions(trace) -> dict:
    """
    Format interactions from trace data into a standardized format.
    Returns a dictionary containing formatted interactions based on trace data.

    The function processes spans from self.trace and formats them into interactions
    of various types including: agent_start, agent_end, input, output, tool_call_start,
    tool_call_end, llm_call, file_read, file_write, network_call.

    Returns:
        dict: A dictionary with "workflow" key containing a list of interactions
                sorted by timestamp.
    """
    interactions = []
    interaction_id = 1

    try:
        if 'data' not in trace or not trace['data'][0].get("spans"):
            return {"workflow": []}
    except Exception as e:
        logger.error(f"Error in checking data or spans: {str(e)}")
        return {"workflow": []}


    for span in trace['data'][0].get("spans", []):
        try:
            # Process agent spans
            if span.get('type') == "agent":
                # Add agent_start interaction
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": "agent_call_start",
                        "name": span.get('name'),
                        "content": None,
                        "timestamp": span.get('start_time'),
                        "error": span.get('error'),
                    }
                )
                interaction_id += 1

                # Process children of agent recursively
                if "children" in span.get('data', {}):
                    for child in span['data'].get("children", []):
                        interaction_id = process_child_interactions(
                            child, interaction_id, interactions
                        )

                # Add agent_end interaction
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": "agent_call_end",
                        "name": span.get('name'),
                        "content": span.get('data', {}).get("output"),
                        "timestamp": span.get('end_time'),
                        "error": span.get('error'),
                    }
                )
                interaction_id += 1

            elif span.get('type') == "tool":
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": "tool_call_start",
                        "name": span.get('name'),
                        "content": {
                            "prompt": span.get('data', {}).get("input"),
                            "response": span.get('data', {}).get("output"),
                        },
                        "timestamp": span.get('start_time'),
                        "error": span.get('error'),
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": "tool_call_end",
                        "name": span.get('name'),
                        "content": {
                            "prompt": span.get('data', {}).get("input"),
                            "response": span.get('data', {}).get("output"),
                        },
                        "timestamp": span.get('end_time'),
                        "error": span.get('error'),
                    }
                )
                interaction_id += 1

            elif span.get('type') == "llm":
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": "llm_call_start",
                        "name": span.get('name'),
                        "content": {
                            "prompt": span.get('data', {}).get("input"),
                        },
                        "timestamp": span.get('start_time'),
                        "error": span.get('error')
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": "llm_call_end",
                        "name": span.get('name'),
                        "content": {"response": span.get('data', {}).get("output")},
                        "timestamp": span.get('end_time'),
                        "error": span.get('error'),
                    }
                )
                interaction_id += 1

            else:
                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": f"{span.get('type')}_call_start",
                        "name": span.get('name'),
                        "content": span.get('data'),
                        "timestamp": span.get('start_time'),
                        "error": span.get('error'),
                    }
                )
                interaction_id += 1

                interactions.append(
                    {
                        "id": str(interaction_id),
                        "span_id": span.get('id'),
                        "interaction_type": f"{span.get('type')}_call_end",
                        "name": span.get('name'),
                        "content": span.get('data'),
                        "timestamp": span.get('end_time'),
                        "error": span.get('error'),
                    }
                )
                interaction_id += 1

            # Process interactions from span.data if they exist
            if 'interactions' in span:
                for span_interaction in span['interactions']:
                    interaction = {}
                    interaction["id"] = str(interaction_id)
                    interaction["span_id"] = span.get('id')
                    interaction["interaction_type"] = span_interaction.get('type')
                    interaction["content"] = span_interaction.get('content')
                    interaction["timestamp"] = span_interaction.get('timestamp')
                    interaction["error"] = span.get('error')
                    interactions.append(interaction)
                    interaction_id += 1

            if 'network_calls' in span:
                for span_network_call in span['network_calls']:
                    network_call = {}
                    network_call["id"] = str(interaction_id)
                    network_call["span_id"] = span.get('id')
                    network_call["interaction_type"] = "network_call"
                    network_call["name"] = None
                    network_call["content"] = {
                        "request": {
                            "url": span_network_call.get("url"),
                            "method": span_network_call.get("method"),
                            "headers": span_network_call.get("headers"),
                        },
                        "response": {
                            "status_code": span_network_call.get("status_code"),
                            "headers": span_network_call.get("response_headers"),
                            "body": span_network_call.get("response_body"),
                        },
                    }
                    network_call["timestamp"] = span_network_call.get("timestamp")
                    network_call["error"] = span_network_call.get("error")
                    interactions.append(network_call)
                    interaction_id += 1
        
        except Exception as e:
            logger.warning(f"Found issue processing span, skipping")
            continue
    try:
        # Sort interactions by timestamp
        sorted_interactions = sorted(
            interactions, key=lambda x: x.get("timestamp") if x.get("timestamp") else ""
        )
    except Exception as e:
        logger.error(f"Error in sorting interactions: {str(e)}")
        sorted_interactions = interactions

    try:
        # Reassign IDs to maintain sequential order after sorting
        for idx, interaction in enumerate(sorted_interactions, 1):
            interaction["id"] = str(idx)
    except Exception as e:
        logger.error(f"Error in reassigning IDs: {str(e)}")
        

    return {"workflow": sorted_interactions}
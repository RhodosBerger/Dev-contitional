"""
Workflow Execution Engine for Dev-conditional Server Engine
Executes node-based workflows with conditional logic and data flow
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

from ..storage.database import AsyncSessionLocal
from ..storage.models import Workflow, WorkflowExecution
from ..websocket.ws_handler import notify_workflow_update
from ..config import settings

logger = logging.getLogger(__name__)


class NodeExecutionStatus(Enum):
    """Node execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowExecutor:
    """Main workflow execution engine"""

    def __init__(self):
        self.node_executors = {
            "trigger": self._execute_trigger_node,
            "api_call": self._execute_api_call_node,
            "condition": self._execute_condition_node,
            "data_transform": self._execute_data_transform_node,
            "llm_prompt": self._execute_llm_prompt_node,
            "code_execution": self._execute_code_execution_node,
            "notification": self._execute_notification_node,
            "database_query": self._execute_database_query_node,
            "file_operation": self._execute_file_operation_node
        }

    @classmethod
    async def execute_workflow_async(
        cls,
        workflow_id: int,
        execution_id: int,
        input_data: Dict[str, Any]
    ):
        """Execute workflow asynchronously"""
        executor = cls()
        await executor._execute_workflow(workflow_id, execution_id, input_data)

    async def _execute_workflow(
        self,
        workflow_id: int,
        execution_id: int,
        input_data: Dict[str, Any]
    ):
        """Internal workflow execution method"""
        execution_context = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "input_data": input_data,
            "output_data": {},
            "node_results": {},
            "start_time": datetime.now(),
            "status": "running"
        }

        try:
            # Load workflow from database
            workflow = await self._load_workflow(workflow_id)
            if not workflow:
                raise Exception(f"Workflow {workflow_id} not found")

            # Update execution status
            await self._update_execution_status(execution_id, "running")

            # Parse workflow data
            workflow_data = workflow.workflow_data
            nodes = workflow_data.get("nodes", [])
            edges = workflow_data.get("edges", [])

            # Build execution graph
            execution_graph = self._build_execution_graph(nodes, edges)

            # Execute workflow
            execution_context = await self._execute_graph(execution_graph, execution_context)

            # Mark execution as completed
            await self._update_execution_status(
                execution_id,
                "completed",
                execution_context["output_data"]
            )

            # Notify completion
            await notify_workflow_update(str(workflow_id), {
                "type": "execution_completed",
                "execution_id": execution_id,
                "output_data": execution_context["output_data"],
                "duration": (datetime.now() - execution_context["start_time"]).total_seconds()
            })

            logger.info(f"Workflow {workflow_id} execution {execution_id} completed successfully")

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            execution_context["error"] = str(e)

            # Mark execution as failed
            await self._update_execution_status(execution_id, "failed", error_message=str(e))

            # Notify failure
            await notify_workflow_update(str(workflow_id), {
                "type": "execution_failed",
                "execution_id": execution_id,
                "error": str(e),
                "duration": (datetime.now() - execution_context["start_time"]).total_seconds()
            })

    async def _load_workflow(self, workflow_id: int) -> Optional[Workflow]:
        """Load workflow from database"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select

            query = select(Workflow).where(Workflow.id == workflow_id)
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def _update_execution_status(
        self,
        execution_id: int,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Update execution status in database"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, update

            query = select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
            result = await session.execute(query)
            execution = result.scalar_one_or_none()

            if execution:
                execution.status = status
                if output_data:
                    execution.output_data = output_data
                if error_message:
                    execution.error_message = error_message
                if status in ["completed", "failed"]:
                    execution.completed_at = datetime.now()

                await session.commit()

    def _build_execution_graph(self, nodes: List[Dict], edges: List[Dict]) -> Dict[str, Any]:
        """Build execution graph from nodes and edges"""
        # Create node map
        node_map = {}
        for node in nodes:
            node_map[node["id"]] = {
                "node": node,
                "dependencies": [],
                "dependents": [],
                "status": NodeExecutionStatus.PENDING
            }

        # Build dependencies from edges
        for edge in edges:
            source = edge["source"]
            target = edge["target"]

            if source in node_map and target in node_map:
                node_map[target]["dependencies"].append(source)
                node_map[source]["dependents"].append(target)

        # Find starting nodes (no dependencies)
        start_nodes = [
            node_id for node_id, node_info in node_map.items()
            if not node_info["dependencies"]
        ]

        return {
            "nodes": node_map,
            "start_nodes": start_nodes,
            "execution_order": []
        }

    async def _execute_graph(
        self,
        graph: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the workflow graph"""
        nodes = graph["nodes"]
        start_nodes = graph["start_nodes"]
        execution_order = graph["execution_order"]

        # Queue of nodes ready to execute
        ready_queue = asyncio.Queue()

        # Add start nodes to queue
        for node_id in start_nodes:
            await ready_queue.put(node_id)

        # Track completed nodes
        completed_nodes = set()
        failed_nodes = set()

        # Execute until no more nodes can be processed
        while not ready_queue.empty() and len(completed_nodes) + len(failed_nodes) < len(nodes):
            try:
                # Get node to execute (with timeout)
                node_id = await asyncio.wait_for(ready_queue.get(), timeout=1.0)

                if node_id in completed_nodes or node_id in failed_nodes:
                    continue

                node_info = nodes[node_id]
                node = node_info["node"]

                # Check if all dependencies are completed
                dependencies_completed = all(
                    dep_id in completed_nodes
                    for dep_id in node_info["dependencies"]
                )

                if not dependencies_completed:
                    # Put back in queue and continue
                    await ready_queue.put(node_id)
                    await asyncio.sleep(0.1)
                    continue

                # Execute node
                try:
                    await self._execute_node(node_id, node, context)
                    completed_nodes.add(node_id)
                    node_info["status"] = NodeExecutionStatus.COMPLETED

                    # Add dependent nodes to queue
                    for dependent_id in node_info["dependents"]:
                        if dependent_id not in completed_nodes and dependent_id not in failed_nodes:
                            await ready_queue.put(dependent_id)

                    # Notify node completion
                    await notify_workflow_update(str(context["workflow_id"]), {
                        "type": "node_completed",
                        "node_id": node_id,
                        "node_name": node.get("data", {}).get("name", "Unknown"),
                        "execution_id": context["execution_id"]
                    })

                except Exception as e:
                    logger.error(f"Node {node_id} execution failed: {str(e)}")
                    failed_nodes.add(node_id)
                    node_info["status"] = NodeExecutionStatus.FAILED
                    context["output_data"]["error"] = str(e)

                    # Notify node failure
                    await notify_workflow_update(str(context["workflow_id"]), {
                        "type": "node_failed",
                        "node_id": node_id,
                        "node_name": node.get("data", {}).get("name", "Unknown"),
                        "error": str(e),
                        "execution_id": context["execution_id"]
                    })

            except asyncio.TimeoutError:
                # Check if there are still nodes being processed
                if ready_queue.empty():
                    break
                continue

        return context

    async def _execute_node(
        self,
        node_id: str,
        node: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Execute a single node"""
        node_type = node.get("type", "unknown")
        node_data = node.get("data", {})

        logger.info(f"Executing node {node_id} of type {node_type}")

        # Get the appropriate executor
        executor = self.node_executors.get(node_type)
        if not executor:
            raise Exception(f"No executor found for node type: {node_type}")

        # Execute the node
        result = await executor(node_id, node_data, context)

        # Store node result
        context["node_results"][node_id] = result

        # Merge result into context output data
        if isinstance(result, dict):
            context["output_data"].update(result)

    # Node executors
    async def _execute_trigger_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute trigger node"""
        trigger_type = node_data.get("trigger_type", "manual")

        if trigger_type == "manual":
            # Manual trigger - pass through input data
            return context["input_data"]
        elif trigger_type == "webhook":
            # Webhook trigger - return webhook data
            return {"webhook_data": context.get("webhook_data", {})}
        elif trigger_type == "schedule":
            # Scheduled trigger - add timestamp
            return {"triggered_at": datetime.now().isoformat()}
        else:
            return {"trigger_type": trigger_type}

    async def _execute_api_call_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute API call node"""
        import httpx

        url = node_data.get("url", "")
        method = node_data.get("method", "GET")
        headers = node_data.get("headers", {})
        body = node_data.get("body", {})

        # Substitute variables in URL and body
        url = self._substitute_variables(url, context)
        if isinstance(body, dict):
            body = self._substitute_variables_dict(body, context)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if body else None
                )

                return {
                    "api_response": {
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
                    }
                }
        except Exception as e:
            return {"api_error": str(e)}

    async def _execute_condition_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute condition node"""
        condition = node_data.get("condition", "")
        true_path = node_data.get("true_path", [])
        false_path = node_data.get("false_path", [])

        # Evaluate condition (simple implementation)
        try:
            # For now, support simple key-value comparisons
            if "==" in condition:
                key, value = condition.split("==", 1)
                key = key.strip()
                value = value.strip().strip('"\'')

                context_value = self._get_nested_value(key.strip(), context)
                result = str(context_value) == value

                return {
                    "condition_result": result,
                    "condition": condition,
                    "next_path": true_path if result else false_path
                }
            else:
                return {"condition_result": True, "condition": condition}
        except Exception as e:
            return {"condition_error": str(e), "condition_result": False}

    async def _execute_data_transform_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data transformation node"""
        transformation_type = node_data.get("transformation_type", "map")
        source_data = node_data.get("source_data", "")
        target_format = node_data.get("target_format", {})

        # Get source data from context
        data = self._get_nested_value(source_data, context)

        if transformation_type == "map":
            # Simple field mapping
            if isinstance(data, dict) and isinstance(target_format, dict):
                return {"transformed_data": self._map_fields(data, target_format)}
            else:
                return {"transformed_data": data}
        elif transformation_type == "filter":
            # Filter data based on criteria
            filter_criteria = node_data.get("filter_criteria", {})
            return {"filtered_data": self._filter_data(data, filter_criteria, context)}
        else:
            return {"transformed_data": data}

    async def _execute_llm_prompt_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute LLM prompt node"""
        from ..llm.service import LLMService

        prompt_template = node_data.get("prompt_template", "")
        system_prompt = node_data.get("system_prompt", "")

        # Substitute variables in prompt
        prompt = self._substitute_variables(prompt_template, context)

        try:
            llm_service = LLMService()
            response = await llm_service.chat(
                message=prompt,
                context={"system_prompt": system_prompt}
            )

            return {"llm_response": response["response"]}
        except Exception as e:
            return {"llm_error": str(e)}

    async def _execute_code_execution_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute code node (with safety restrictions)"""
        code = node_data.get("code", "")
        language = node_data.get("language", "python")

        # For now, only support simple Python expressions
        if language == "python":
            try:
                # Create a safe execution environment
                safe_globals = {
                    "__builtins__": {
                        "len": len,
                        "str": str,
                        "int": int,
                        "float": float,
                        "bool": bool,
                        "list": list,
                        "dict": dict,
                    }
                }

                # Add context variables
                safe_locals = {"context": context, "input_data": context["input_data"]}

                # Execute code (limited to expressions for safety)
                result = eval(code, safe_globals, safe_locals)
                return {"code_result": result}
            except Exception as e:
                return {"code_error": str(e)}
        else:
            return {"error": f"Language {language} not supported"}

    async def _execute_notification_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute notification node"""
        message = node_data.get("message", "")
        notification_type = node_data.get("notification_type", "log")
        recipients = node_data.get("recipients", [])

        # Substitute variables in message
        message = self._substitute_variables(message, context)

        if notification_type == "log":
            logger.info(f"Notification: {message}")
            return {"notification_sent": True, "type": "log"}
        elif notification_type == "email":
            # TODO: Implement email sending
            logger.info(f"Email notification to {recipients}: {message}")
            return {"notification_sent": True, "type": "email", "recipients": recipients}
        else:
            return {"notification_sent": False, "error": "Unknown notification type"}

    async def _execute_database_query_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute database query node"""
        query = node_data.get("query", "")
        query_type = node_data.get("query_type", "select")

        # For now, just return the query information
        # TODO: Implement actual database execution
        return {
            "database_query": query,
            "query_type": query_type,
            "executed": False,
            "message": "Database execution not yet implemented"
        }

    async def _execute_file_operation_node(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute file operation node"""
        operation = node_data.get("operation", "read")
        file_path = node_data.get("file_path", "")

        # Substitute variables in file path
        file_path = self._substitute_variables(file_path, context)

        try:
            if operation == "read":
                import os
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    return {"file_content": content, "file_path": file_path}
                else:
                    return {"file_error": f"File not found: {file_path}"}
            elif operation == "write":
                content = node_data.get("content", "")
                content = self._substitute_variables(content, context)

                import os
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(content)
                return {"file_written": True, "file_path": file_path}
            else:
                return {"file_error": f"Unknown operation: {operation}"}
        except Exception as e:
            return {"file_error": str(e)}

    # Helper methods
    def _substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Substitute variables in text using context"""
        import re

        def replace_var(match):
            var_path = match.group(1)
            value = self._get_nested_value(var_path, context)
            return str(value) if value is not None else ""

        return re.sub(r'\{\{([^}]+)\}\}', replace_var, text)

    def _substitute_variables_dict(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute variables in dictionary values"""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._substitute_variables(value, context)
            elif isinstance(value, dict):
                result[key] = self._substitute_variables_dict(value, context)
            else:
                result[key] = value
        return result

    def _get_nested_value(self, path: str, data: Dict[str, Any]) -> Any:
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _map_fields(self, source_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map fields from source data using mapping configuration"""
        result = {}
        for target_field, source_field in mapping.items():
            if source_field in source_data:
                result[target_field] = source_data[source_field]
        return result

    def _filter_data(self, data: Any, criteria: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Filter data based on criteria"""
        # Simple filtering implementation
        if isinstance(data, list) and criteria:
            filtered_items = []
            for item in data:
                if isinstance(item, dict):
                    match = True
                    for key, value in criteria.items():
                        if key in item and item[key] != value:
                            match = False
                            break
                    if match:
                        filtered_items.append(item)
            return filtered_items
        return data
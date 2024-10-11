from abc import abstractmethod
from typing import TYPE_CHECKING, cast

from langchain.agents import AgentExecutor, BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.agent import RunnableAgent
from langchain_core.runnables import Runnable

from langflow.base.agents.callback import AgentAsyncHandler
from langflow.base.agents.utils import data_to_messages
from langflow.custom import Component
from langflow.field_typing import Text
from langflow.inputs.inputs import InputTypes
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput
from langflow.schema import Data
from langflow.schema.message import Message
from langflow.template import Output
from langflow.utils.constants import MESSAGE_SENDER_AI

# Imports
import os
import time
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import tiktoken  # For accurate token counting


if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


MAX_TOTAL_TOKENS = 2048   # Adjust based on OpenAI's token limit per request
RETRY_LIMIT = 3
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff factor
MAX_REFINEMENT_ATTEMPTS = 3
MAX_CHAT_HISTORY_TOKENS = 4096  # Max tokens for the chat mode

client = OpenAI()


class LCAgentComponent(Component):
    trace_type = "agent"
    _base_inputs: list[InputTypes] = [
        MessageTextInput(name="input_value", display_name="Input"),
        BoolInput(
            name="handle_parsing_errors",
            display_name="Handle Parse Errors",
            value=True,
            advanced=True,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose",
            value=True,
            advanced=True,
        ),
        IntInput(
            name="max_iterations",
            display_name="Max Iterations",
            value=15,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Agent", name="agent", method="build_agent"),
        Output(display_name="Response", name="response", method="message_response"),
    ]

    @abstractmethod
    def build_agent(self) -> AgentExecutor:
        """Create the agent."""

    async def message_response(self) -> Message:
        """Run the agent and return the response."""
        agent = self.build_agent()
        result = await self.run_agent(agent=agent)

        if isinstance(result, list):
            result = "\n".join([result_dict["text"] for result_dict in result])
        message = Message(text=result, sender=MESSAGE_SENDER_AI)
        self.status = message
        return message

    def _validate_outputs(self):
        required_output_methods = ["build_agent"]
        output_names = [output.name for output in self.outputs]
        for method_name in required_output_methods:
            if method_name not in output_names:
                msg = f"Output with name '{method_name}' must be defined."
                raise ValueError(msg)
            if not hasattr(self, method_name):
                msg = f"Method '{method_name}' must be defined."
                raise ValueError(msg)

    def get_agent_kwargs(self, flatten: bool = False) -> dict:
        base = {
            "handle_parsing_errors": self.handle_parsing_errors,
            "verbose": self.verbose,
            "allow_dangerous_code": True,
        }
        agent_kwargs = {
            "handle_parsing_errors": self.handle_parsing_errors,
            "max_iterations": self.max_iterations,
        }
        if flatten:
            return {
                **base,
                **agent_kwargs,
            }
        return {**base, "agent_executor_kwargs": agent_kwargs}

    def get_chat_history_data(self) -> list[Data] | None:
        # might be overridden in subclasses
        return None

    async def run_agent(self, agent: AgentExecutor) -> Text:
        input_dict: dict[str, str | list[BaseMessage]] = {"input": self.input_value}
        self.chat_history = self.get_chat_history_data()
        if self.chat_history:
            input_dict["chat_history"] = data_to_messages(self.chat_history)
        result = agent.invoke(
            input_dict, config={"callbacks": [AgentAsyncHandler(self.log), *self.get_langchain_callbacks()]}
        )
        self.status = result
        if "output" not in result:
            msg = "Output key not found in result. Tried 'output'."
            raise ValueError(msg)

        return cast(str, result.get("output"))


class LCToolsAgentComponent(LCAgentComponent):
    _base_inputs = [
        *LCAgentComponent._base_inputs,
        HandleInput(name="tools", display_name="Tools", input_types=["Tool", "BaseTool"], is_list=True),
    ]

    def build_agent(self) -> AgentExecutor:
        agent = self.create_agent_runnable()
        return AgentExecutor.from_agent_and_tools(
            agent=RunnableAgent(runnable=agent, input_keys_arg=["input"], return_keys_arg=["output"]),
            tools=self.tools,
            **self.get_agent_kwargs(flatten=True),
        )

    async def run_agent(
        self,
        agent: Runnable | BaseSingleActionAgent | BaseMultiActionAgent | AgentExecutor,
    ) -> Text:
        if isinstance(agent, AgentExecutor):
            runnable = agent
        else:
            runnable = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=self.tools,
                handle_parsing_errors=self.handle_parsing_errors,
                verbose=self.verbose,
                max_iterations=self.max_iterations,
            )
        input_dict: dict[str, str | list[BaseMessage]] = {"input": self.input_value}
        if self.chat_history:
            input_dict["chat_history"] = data_to_messages(self.chat_history)

        result = runnable.invoke(
            input_dict, config={"callbacks": [AgentAsyncHandler(self.log), *self.get_langchain_callbacks()]}
        )
        self.status = result
        if "output" not in result:
            msg = "Output key not found in result. Tried 'output'."
            raise ValueError(msg)

        return cast(str, result.get("output"))

    @abstractmethod
    def create_agent_runnable(self) -> Runnable:
        """Create the agent."""

class CustomAgent:
    """
    Represents an agent that can perform various reasoning actions.
    """
    ACTION_DESCRIPTIONS = {
        'discuss': "formulating a response",
        'verify': "verifying data",
        'refine': "refining the response",
        'critique': "critiquing another agent's response"
    }

    def __init__(self, color, **kwargs):
        """
        Initialize an agent with custom instructions.
        """
        self.name = kwargs.get('name', 'Unnamed Agent')
        self.color = color
        self.messages = []
        self.chat_history = []  # For chat mode
        self.lock = None
        self.system_purpose = kwargs.get('system_purpose', '')
        self.personality_traits = kwargs.get('personality', {})
        self.interaction_style = kwargs.get('interaction_style', {})
        self.ethical_conduct = kwargs.get('ethical_conduct', {})
        self.capabilities_limitations = kwargs.get('capabilities_limitations', {})
        self.context_awareness = kwargs.get('context_awareness', {})
        self.adaptability_engagement = kwargs.get('adaptability_engagement', {})
        self.responsiveness = kwargs.get('responsiveness', {})
        self.additional_tools_modules = kwargs.get('additional_tools_modules', {})
        self.custom_instructions = kwargs  # Store all other custom instructions
        self.other_agents_info = ""  # Will be set after all agents are initialized

    def _add_message(self, role, content, mode='reasoning'):
        """
        Adds a message to the agent's message history and ensures token limit is not exceeded.
        """
        if mode == 'chat':
            self.chat_history.append({"role": role, "content": content})
            # Enforce maximum token limit for chat history
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logging.error(f"Error getting encoding: {e}")
                raise e
            total_tokens = sum(len(encoding.encode(msg['content'])) for msg in self.chat_history)
            if total_tokens > MAX_CHAT_HISTORY_TOKENS:
                # Trim messages from the beginning
                while total_tokens > MAX_CHAT_HISTORY_TOKENS and len(self.chat_history) > 1:
                    self.chat_history.pop(0)
                    total_tokens = sum(len(encoding.encode(msg['content'])) for msg in self.chat_history)
        else:
            self.messages.append({"role": role, "content": content})
            # Enforce a maximum message history length based on token count
            try:
                # Use a known encoding compatible with chat models
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logging.error(f"Error getting encoding: {e}")
                raise e
            total_tokens = sum(len(encoding.encode(msg['content'])) for msg in self.messages)
            if total_tokens > MAX_TOTAL_TOKENS:
                # Trim messages from the beginning, but keep the initial instruction
                while total_tokens > MAX_TOTAL_TOKENS and len(self.messages) > 1:
                    if self.messages[0]['content'] == self.system_purpose:
                        # Keep the initial instruction message
                        self.messages.pop(1)
                    else:
                        self.messages.pop(0)
                    total_tokens = sum(len(encoding.encode(msg['content'])) for msg in self.messages)

    def _handle_chat_response(self, prompt):
        """
        Handles the chat response for reasoning logic using o1-preview model.
        """
        self._add_message("user", prompt)

        # Prepare the messages array including the initial instruction and conversation history
        messages = self.messages.copy()

        # Start timing
        start_time = time.time()

        # Initialize retry parameters
        retries = 0
        backoff = 1  # Initial backoff time in seconds

        while retries < RETRY_LIMIT:
            try:
                # Agent generates a response
                # response = client.chat.completions.create(
                #     model="o1-preview-2024-09-12",
                #     messages=messages
                # )
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )

                # End timing
                end_time = time.time()
                duration = end_time - start_time

                # Extract and return reply
                assistant_reply = response.choices[0].message.content.strip()
                self._add_message("assistant", assistant_reply)

                return assistant_reply, duration

            except Exception as e:
                error_type = type(e).__name__
                logging.error(f"Error in agent '{self.name}': {error_type}: {e}")
                retries += 1
                if retries >= RETRY_LIMIT:
                    logging.error(f"Agent '{self.name}' reached maximum retry limit.")
                    break
                backoff_time = backoff * (RETRY_BACKOFF_FACTOR ** (retries - 1))
                logging.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)

        return "An error occurred while generating a response.", time.time() - start_time

    def _handle_chat_interaction(self, user_message):
        """
        Handles chat interaction with the agent using gpt-4o model.
        """
        # Prepare the system message with agent's personality and quirks
        system_message = self._build_system_message()

        # Add system message at the beginning of the conversation
        messages = [{"role": "system", "content": system_message}] + self.chat_history

        # Add user's message
        messages.append({"role": "user", "content": user_message})

        # Start timing
        start_time = time.time()

        # Initialize retry parameters
        retries = 0
        backoff = 1

        while retries < RETRY_LIMIT:
            try:
                # Agent generates a response
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )

                # End timing
                end_time = time.time()
                duration = end_time - start_time

                # Extract and return reply
                assistant_reply = response.choices[0].message.content.strip()
                self._add_message("assistant", assistant_reply, mode='chat')

                return assistant_reply, duration

            except Exception as e:
                error_type = type(e).__name__
                logging.error(f"Error in chat with agent '{self.name}': {error_type}: {e}")
                retries += 1
                if retries >= RETRY_LIMIT:
                    logging.error(f"Agent '{self.name}' reached maximum retry limit in chat.")
                    break
                backoff_time = backoff * (RETRY_BACKOFF_FACTOR ** (retries - 1))
                logging.info(f"Retrying chat in {backoff_time} seconds...")
                time.sleep(backoff_time)

        return "An error occurred while generating a response.", time.time() - start_time

    def _build_system_message(self):
        """
        Builds the system message incorporating the agent's personality and quirks,
        as well as information about other agents.
        """
        personality_description = f"Your name is {self.name}. {self.system_purpose}"

        # Include interaction style
        if self.interaction_style:
            interaction_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.interaction_style.items())
            personality_description += f"\n\nInteraction Style:\n{interaction_details}"

        # Include personality traits
        if self.personality_traits:
            personality_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.personality_traits.items())
            personality_description += f"\n\nPersonality Traits:\n{personality_details}"

        # Include ethical conduct
        if self.ethical_conduct:
            ethical_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.ethical_conduct.items())
            personality_description += f"\n\nEthical Conduct:\n{ethical_details}"

        # Include capabilities and limitations
        if self.capabilities_limitations:
            capabilities_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.capabilities_limitations.items())
            personality_description += f"\n\nCapabilities and Limitations:\n{capabilities_details}"

        # Include context awareness
        if self.context_awareness:
            context_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.context_awareness.items())
            personality_description += f"\n\nContext Awareness:\n{context_details}"

        # Include adaptability and engagement
        if self.adaptability_engagement:
            adaptability_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.adaptability_engagement.items())
            personality_description += f"\n\nAdaptability and Engagement:\n{adaptability_details}"

        # Include responsiveness
        if self.responsiveness:
            responsiveness_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.responsiveness.items())
            personality_description += f"\n\nResponsiveness:\n{responsiveness_details}"

        # Include additional tools and modules
        if self.additional_tools_modules:
            tools_details = "\n".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in self.additional_tools_modules.items())
            personality_description += f"\n\nAdditional Tools and Modules:\n{tools_details}"

        # Include information about other agents
        if self.other_agents_info:
            personality_description += f"\n\nYou are aware of the following other agents:\n{self.other_agents_info}"

        return personality_description

    def discuss(self, prompt):
        """
        Agent formulates a response to the user's prompt.
        """
        return self._handle_chat_response(prompt)

    def verify(self, data):
        """
        Agent verifies the accuracy of the provided data.
        """
        verification_prompt = f"Verify the accuracy of the following information:\n\n{data}"
        return self._handle_chat_response(verification_prompt)

    def refine(self, data, more_time=False, iterations=2):
        """
        Agent refines the response to improve its accuracy and completeness.
        """
        refinement_prompt = f"Please refine the following response to improve its accuracy and completeness:\n\n{data}"
        if more_time:
            refinement_prompt += "\nTake additional time to improve the response thoroughly."

        total_duration = 0
        refined_response = data
        for i in range(iterations):
            refined_response, duration = self._handle_chat_response(refinement_prompt)
            total_duration += duration
            # Update the prompt for the next iteration
            refinement_prompt = f"Please further refine the following response:\n\n{refined_response}"

        return refined_response, total_duration

    def critique(self, other_agent_response):
        """
        Agent critiques another agent's response for accuracy and completeness.
        """
        critique_prompt = f"Critique the following response for accuracy and completeness:\n\n{other_agent_response}"
        return self._handle_chat_response(critique_prompt)
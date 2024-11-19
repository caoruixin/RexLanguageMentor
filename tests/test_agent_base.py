import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import sys
import os

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.agents.agent_base import AgentBase  # Correct import path for AgentBase
from langchain_core.prompts import MessagesPlaceholder


class TestAgentBase(unittest.TestCase):
    def setUp(self):
        self.mock_prompt_file = "mock_prompt_file.txt"
        self.mock_intro_file = "mock_intro_file.json"
        self.agent_name = "TestAgent"
        self.session_id = "test_session"
        self.mock_prompt_content = "This is a mock system prompt."
        self.mock_intro_content = [{"role": "system", "content": "Hello, I'm here to assist you."}]

    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock system prompt.")
    def test_load_prompt_success(self, mock_file):
        agent = AgentBase(name=self.agent_name, prompt_file=self.mock_prompt_file)
        self.assertEqual(agent.prompt, self.mock_prompt_content)
        mock_file.assert_called_once_with(self.mock_prompt_file, "r", encoding="utf-8")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_prompt_file_not_found(self, mock_file):
        with self.assertRaises(FileNotFoundError):
            AgentBase(name=self.agent_name, prompt_file=self.mock_prompt_file)

    @patch("builtins.open", new_callable=mock_open)
    def test_load_intro_success(self, mock_file):
        mock_file.side_effect = [
            mock_open(read_data="This is a mock system prompt.").return_value,
            mock_open(read_data=json.dumps([{"role": "system", "content": "Hello, I'm here to assist you."}])).return_value,
        ]

        agent = AgentBase(
            name=self.agent_name,
            prompt_file=self.mock_prompt_file,
            intro_file=self.mock_intro_file,
        )

        mock_file.assert_any_call(self.mock_prompt_file, "r", encoding="utf-8")
        mock_file.assert_any_call(self.mock_intro_file, "r", encoding="utf-8")
        self.assertEqual(agent.intro_messages, self.mock_intro_content)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_load_intro_file_not_found(self, mock_file):
        with self.assertRaises(FileNotFoundError):
            AgentBase(name=self.agent_name, prompt_file=self.mock_prompt_file, intro_file=self.mock_intro_file)

    @patch("builtins.open", new_callable=mock_open, read_data="Invalid JSON")
    def test_load_intro_invalid_json(self, mock_file):
        with self.assertRaises(ValueError):
            AgentBase(name=self.agent_name, prompt_file=self.mock_prompt_file, intro_file=self.mock_intro_file)

    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock system prompt.")
    @patch("src.agents.agent_base.ChatOllama")
    @patch("src.agents.agent_base.ChatPromptTemplate.from_messages")
    def test_create_chatbot(self, mock_prompt_template, mock_chat_ollama, mock_file):
        mock_prompt_template.return_value = MagicMock()
        mock_chat_ollama.return_value = MagicMock()

        agent = AgentBase(name=self.agent_name, prompt_file=self.mock_prompt_file)
        agent.create_chatbot()

        # Verify that from_messages is called with the correct arguments
        mock_prompt_template.assert_any_call([
            ("system", "This is a mock system prompt."),
            MessagesPlaceholder(variable_name="messages"),
        ])


    @patch("src.agents.agent_base.RunnableWithMessageHistory")
    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock system prompt.")
    def test_chat_with_history(self, mock_file, mock_runnable):
        mock_runnable_instance = MagicMock()
        mock_runnable.return_value = mock_runnable_instance
        mock_runnable_instance.invoke.return_value = MagicMock(content="Mocked response")

        agent = AgentBase(name=self.agent_name, prompt_file=self.mock_prompt_file)
        response = agent.chat_with_history(user_input="Test input", session_id=self.session_id)

        self.assertEqual(response, "Mocked response")
        mock_runnable_instance.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()


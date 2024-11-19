import unittest
from unittest.mock import patch, MagicMock, mock_open
import random
import sys
import os
import json

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.agents.scenario_agent import ScenarioAgent  # Import ScenarioAgent
from src.agents.session_history import get_session_history  # Mock session history
from langchain_core.messages import AIMessage


class TestScenarioAgent(unittest.TestCase):
    def setUp(self):
        self.scenario_name = "test_scenario"
        self.session_id = "test_session"
        self.mock_prompt_file = f"prompts/{self.scenario_name}_prompt.txt"
        self.mock_intro_file = f"content/intro/{self.scenario_name}.json"
        self.mock_intro_messages = ["Hello, how can I help?", "Hi there! Welcome to the scenario."]
        self.mock_prompt_content = "This is a mock scenario prompt."

    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock scenario prompt.")
    @patch("src.agents.scenario_agent.random.choice")
    @patch("src.agents.session_history.get_session_history")
    def test_start_new_session_no_history(self, mock_get_session_history, mock_random_choice, mock_file):
        """
        Test starting a new session with no history.
        """
        # Mock session history to have no messages
        mock_history = MagicMock()
        mock_history.messages = []
        mock_get_session_history.return_value = mock_history

        # Mock random choice for intro messages
        mock_random_choice.return_value = "Hello, how can I help?"

        # Mock intro file to contain intro messages
        with patch("builtins.open", mock_open(read_data=json.dumps(self.mock_intro_messages))):
            agent = ScenarioAgent(scenario_name=self.scenario_name, session_id=self.session_id)

        initial_message = agent.start_new_session(session_id=self.session_id)

        # Validate the initial AI message
        self.assertEqual(initial_message, "Hello, how can I help?")

        # Ensure add_message was called with the correct AIMessage
        mock_history.add_message.assert_called_once_with(AIMessage(content="Hello, how can I help?"))


    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock scenario prompt.")
    @patch("src.agents.session_history.get_session_history")
    def test_start_new_session_with_history(self, mock_get_session_history, mock_file):
        """
        Test starting a new session when history already exists.
        """
        # Mock session history to have existing messages
        mock_history = MagicMock()
        mock_history.messages = [MagicMock(content="Existing message")]
        mock_get_session_history.return_value = mock_history

        # Mock intro file to contain intro messages
        with patch("builtins.open", mock_open(read_data=json.dumps(self.mock_intro_messages))):
            agent = ScenarioAgent(scenario_name=self.scenario_name, session_id=self.session_id)

        initial_message = agent.start_new_session(session_id=self.session_id)

        # Validate the last message in history is returned
        self.assertEqual(initial_message, "Existing message")

        # Ensure no new message was added to history
        mock_history.add_message.assert_not_called()

    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock scenario prompt.")
    def test_load_prompt_file_not_found(self, mock_file):
        """
        Test behavior when the prompt file is not found.
        """
        mock_file.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            ScenarioAgent(scenario_name=self.scenario_name)

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(["Invalid JSON"]))
    def test_load_intro_invalid_json(self, mock_file):
        """
        Test behavior when the intro file contains invalid JSON.
        """
        mock_file.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        with self.assertRaises(ValueError):
            ScenarioAgent(scenario_name=self.scenario_name)

    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock scenario prompt.")
    @patch("src.agents.session_history.get_session_history")
    def test_start_new_session_default_session_id(self, mock_get_session_history, mock_file):
        """
        Test starting a session with a default session ID.
        """
        # Mock session history to have no messages
        mock_history = MagicMock()
        mock_history.messages = []
        mock_get_session_history.return_value = mock_history

        # Mock intro file to contain intro messages
        with patch("builtins.open", mock_open(read_data=json.dumps(self.mock_intro_messages))):
            agent = ScenarioAgent(scenario_name=self.scenario_name)

        initial_message = agent.start_new_session()

        # Validate the session ID defaults to the agent's session ID
        mock_get_session_history.assert_called_once_with(agent.session_id)

        # Validate initial message behavior
        self.assertIn(initial_message, self.mock_intro_messages)


if __name__ == "__main__":
    unittest.main()


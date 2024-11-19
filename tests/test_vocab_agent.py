import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.agents.vocab_agent import VocabAgent  # Import VocabAgent
from src.agents.session_history import get_session_history  # Correct path for session_history
from langchain_core.messages import AIMessage


class TestVocabAgent(unittest.TestCase):
    def setUp(self):
        self.session_id = "test_session"
        self.mock_prompt_file = "prompts/vocab_study_prompt.txt"
        self.mock_prompt_content = "This is a mock vocab study prompt."

    @patch("builtins.open", new_callable=mock_open, read_data="This is a mock vocab study prompt.")
    def test_init_vocab_agent(self, mock_file):
        """
        Test the initialization of the VocabAgent.
        """
        agent = VocabAgent(session_id=self.session_id)
        self.assertEqual(agent.name, "vocab_study")
        self.assertEqual(agent.prompt_file, self.mock_prompt_file)
        self.assertEqual(agent.session_id, self.session_id)
        mock_file.assert_called_once_with(self.mock_prompt_file, "r", encoding="utf-8")

    @patch("src.agents.session_history.get_session_history")
    def test_restart_session(self, mock_get_session_history):
        """
        Test restarting a session by clearing the session history.
        """
        # Mock session history object
        mock_history = MagicMock()
        mock_get_session_history.return_value = mock_history

        agent = VocabAgent(session_id=self.session_id)
        cleared_history = agent.restart_session()

        # Verify the session history is cleared
        mock_history.clear.assert_called_once()
        # Ensure the cleared history object is returned
        self.assertEqual(cleared_history, mock_history)

        # Verify get_session_history is called with the correct session_id
        mock_get_session_history.assert_called_once_with(self.session_id)

    @patch("src.agents.session_history.get_session_history")
    def test_restart_session_default_session_id(self, mock_get_session_history):
        """
        Test restarting a session with the default session ID.
        """
        # Mock session history object
        mock_history = MagicMock()
        mock_get_session_history.return_value = mock_history

        agent = VocabAgent()
        cleared_history = agent.restart_session()

        # Verify the session ID defaults to the agent's session ID
        mock_get_session_history.assert_called_once_with(agent.session_id)
        # Verify the session history is cleared
        mock_history.clear.assert_called_once()
        # Ensure the cleared history object is returned
        self.assertEqual(cleared_history, mock_history)

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_prompt_file_not_found(self, mock_file):
        """
        Test behavior when the prompt file is not found.
        """
        with self.assertRaises(FileNotFoundError):
            VocabAgent(session_id=self.session_id)


if __name__ == "__main__":
    unittest.main()

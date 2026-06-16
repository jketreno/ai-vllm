from __future__ import annotations

import json
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import app.main as main


class SpamClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.token_path = pathlib.Path(self.temp.name) / "token"
        self.token_path.write_text("test-token", encoding="utf-8")
        self.original_token_file = main.TOKEN_FILE
        main.TOKEN_FILE = str(self.token_path)
        self.client = TestClient(main.app)

    def tearDown(self):
        main.TOKEN_FILE = self.original_token_file
        self.temp.cleanup()

    def test_requires_bearer_token(self):
        response = self.client.post(
            "/v1/classify",
            json={"subject": "Hello", "text_body": "Normal message"},
        )
        self.assertEqual(response.status_code, 401)

    def test_applies_threshold_to_structured_model_response(self):
        upstream = Mock()
        upstream.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reasons": [
                                    "Urgent credential-verification request",
                                    "Sender domain does not match the claimed brand",
                                ],
                                "classification": "SPAM",
                                "spam_score": 0.93,
                            }
                        )
                    }
                }
            ]
        }
        with patch("app.main.httpx.post", return_value=upstream) as post:
            response = self.client.post(
                "/v1/classify",
                headers={"Authorization": "Bearer test-token"},
                json={
                    "envelope_from": "billing@lookalike.example",
                    "headers": [{"name": "Authentication-Results", "value": "spf=fail"}],
                    "subject": "Your account will be closed today",
                    "text_body": "Verify your password immediately.",
                },
            )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["classification"], "SPAM")
        self.assertEqual(result["spam_score"], 0.93)
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["temperature"], 0)
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(request["response_format"]["type"], "json_schema")
        upstream.raise_for_status.assert_called_once()

    def test_rejects_empty_message(self):
        response = self.client.post(
            "/v1/classify",
            headers={"Authorization": "Bearer test-token"},
            json={},
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_upstream_output_fails_closed_as_gateway_error(self):
        upstream = Mock()
        upstream.json.return_value = {
            "choices": [{"message": {"content": '{"spam_score": 2}'}}]
        }
        with patch("app.main.httpx.post", return_value=upstream):
            response = self.client.post(
                "/v1/classify",
                headers={"Authorization": "Bearer test-token"},
                json={"subject": "Hello"},
        )
        self.assertEqual(response.status_code, 502)

    def test_normalizes_ham_score_that_contradicts_classification(self):
        upstream = Mock()
        upstream.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reasons": [
                                    "Valid DKIM and SPF authentication signatures confirm legitimate sender identity.",
                                    "No indicators of phishing, credential theft, or unsolicited advertising present.",
                                ],
                                "classification": "HAM",
                                "spam_score": 1.0,
                            }
                        )
                    }
                }
            ]
        }
        with patch("app.main.httpx.post", return_value=upstream):
            response = self.client.post(
                "/v1/classify",
                headers={"Authorization": "Bearer test-token"},
                json={"subject": "Funeral notice", "text_body": "May his memory be eternal."},
            )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["classification"], "HAM")
        self.assertLess(result["spam_score"], result["threshold"])

    def test_normalizes_spam_score_that_contradicts_classification(self):
        upstream = Mock()
        upstream.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "reasons": [
                                    "Urgent credential-verification request",
                                    "Sender domain does not match the claimed brand",
                                ],
                                "classification": "SPAM",
                                "spam_score": 0.1,
                            }
                        )
                    }
                }
            ]
        }
        with patch("app.main.httpx.post", return_value=upstream):
            response = self.client.post(
                "/v1/classify",
                headers={"Authorization": "Bearer test-token"},
                json={"subject": "Urgent password reset", "text_body": "Send your password now."},
            )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["classification"], "SPAM")
        self.assertGreaterEqual(result["spam_score"], result["threshold"])


if __name__ == "__main__":
    unittest.main()

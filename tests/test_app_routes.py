from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import app as genresense_app


class AuthRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        genresense_app.app.config["TESTING"] = True
        genresense_app.app.config["SECRET_KEY"] = "test-secret"
        self.client = genresense_app.app.test_client()

    def test_callback_redirects_to_start_page_after_successful_auth(self) -> None:
        with self.client.session_transaction() as session:
            session[genresense_app.STATE_KEY] = "expected-state"

        fake_oauth = Mock()
        fake_oauth.get_access_token.return_value = {"access_token": "token"}

        with (
            patch.object(genresense_app, "_settings_from_env", return_value=object()),
            patch.object(genresense_app, "_build_oauth", return_value=fake_oauth),
        ):
            response = self.client.get("/callback?code=spotify-code&state=expected-state")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith("/start"))

        with self.client.session_transaction() as session:
            self.assertEqual(session[genresense_app.TOKEN_INFO_KEY]["access_token"], "token")
            self.assertIn("Starting ingestion and feature prep", session["status_message"])

    def test_start_page_renders_auto_submit_form_for_authenticated_user(self) -> None:
        with (
            patch.object(genresense_app, "_settings_from_env", return_value=object()),
            patch.object(
                genresense_app,
                "_get_authenticated_client",
                return_value=(object(), {"display_name": "Test Listener"}),
            ),
        ):
            response = self.client.get("/start")

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Starting your GenreSense process", body)
        self.assertIn('action="/run-pipeline"', body)
        self.assertIn("requestSubmit", body)

    def test_start_page_redirects_home_when_user_is_not_authenticated(self) -> None:
        with (
            patch.object(genresense_app, "_settings_from_env", return_value=object()),
            patch.object(genresense_app, "_get_authenticated_client", return_value=(None, None)),
        ):
            response = self.client.get("/start")

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith("/"))

        with self.client.session_transaction() as session:
            self.assertEqual(session["status_message"], "Connect your Spotify account to start the process.")


if __name__ == "__main__":
    unittest.main()

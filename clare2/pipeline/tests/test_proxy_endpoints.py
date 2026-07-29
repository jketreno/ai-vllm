import unittest

from app.proxy import ALLOWED_ENDPOINTS, parse_endpoint_and_route


class CapacityEndpointRoutingTests(unittest.TestCase):
    def test_v1_prefixed_capacity_request_resolves_to_the_capacity_endpoint(self):
        endpoint, route_id = parse_endpoint_and_route("v1/capacity", None)

        self.assertEqual(endpoint, "/v1/capacity")
        self.assertIsNone(route_id)
        self.assertIn(endpoint, ALLOWED_ENDPOINTS)

    def test_v1_prefixed_health_request_resolves_to_the_health_endpoint(self):
        endpoint, _route_id = parse_endpoint_and_route("v1/health", None)

        self.assertEqual(endpoint, "/v1/health")
        self.assertIn(endpoint, ALLOWED_ENDPOINTS)

    def test_route_in_path_form_still_resolves_for_chat_completions(self):
        endpoint, route_id = parse_endpoint_and_route(
            "my-route/v1/chat/completions", None
        )

        self.assertEqual(endpoint, "/v1/chat/completions")
        self.assertEqual(route_id, "my-route")


if __name__ == "__main__":
    unittest.main()

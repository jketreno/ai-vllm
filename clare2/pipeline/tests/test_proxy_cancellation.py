import asyncio
import unittest

from app.admission import ClientDisconnected
from app.proxy_transport import request_until_disconnected


class ProxyCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_downstream_disconnect_cancels_non_streaming_upstream(self):
        cancelled = asyncio.Event()

        class Downstream:
            async def is_disconnected(self):
                return True

        class Upstream:
            async def request(self, *_args, **_kwargs):
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        with self.assertRaises(ClientDisconnected):
            await request_until_disconnected(
                Upstream(),
                Downstream(),
                "POST",
                "http://vllm/v1/chat/completions",
                b"{}",
                {},
            )
        self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()

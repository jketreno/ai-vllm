import unittest

from exporter import GIB, decode_docker_log, parse_allocations


class ExporterTests(unittest.TestCase):
    def test_parses_latest_profiler_run(self):
        text = """
        Model loading took 1.0 GiB memory
        Available KV cache memory: 2.0 GiB
        Graph capturing finished in 1 secs, took 0.1 GiB
        Model loading took 29.95 GiB memory
        Available KV cache memory: 48.62 GiB
        Graph capturing finished in 18 secs, took 0.33 GiB
        """
        self.assertEqual(
            parse_allocations(text),
            {
                "weights": 29.95 * GIB,
                "kv_cache": 48.62 * GIB,
                "cuda_graphs": 0.33 * GIB,
            },
        )

    def test_decodes_multiplexed_docker_logs(self):
        message = b"Model loading took 29.95 GiB memory\n"
        frame = bytes([1, 0, 0, 0]) + len(message).to_bytes(4, "big") + message
        self.assertEqual(decode_docker_log(frame), message.decode())


if __name__ == "__main__":
    unittest.main()

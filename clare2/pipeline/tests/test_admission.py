import asyncio
import unittest

from app.admission import (
    AdmissionController,
    AdmissionRejected,
    ClientDisconnected,
)


async def connected() -> bool:
    return False


class AdmissionControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_instead_of_building_an_unbounded_queue(self):
        controller = AdmissionController(1, 0, 0, 45)
        lease = await controller.acquire("semantic", connected)

        with self.assertRaises(AdmissionRejected) as rejected:
            await controller.acquire("semantic", connected)

        self.assertEqual(rejected.exception.retry_after, 45)
        self.assertEqual(controller.active, 1)
        self.assertEqual(controller.waiting, 0)
        await lease.release()

    async def test_workload_quota_reserves_capacity_for_other_work(self):
        controller = AdmissionController(
            4,
            0,
            0,
            30,
            {"semantic": 2, "projection": 1},
        )
        first = await controller.acquire("semantic", connected)
        second = await controller.acquire("semantic", connected)

        with self.assertRaises(AdmissionRejected):
            await controller.acquire("semantic", connected)
        projection = await controller.acquire("projection", connected)

        self.assertEqual(controller.active, 3)
        await first.release()
        await second.release()
        await projection.release()

    async def test_bounded_waiter_acquires_released_capacity(self):
        controller = AdmissionController(1, 1, 1, 30)
        first = await controller.acquire("default", connected)
        pending = asyncio.create_task(
            controller.acquire("default", connected)
        )
        await asyncio.sleep(0.05)
        self.assertEqual(controller.waiting, 1)

        await first.release()
        second = await pending

        self.assertEqual(controller.waiting, 0)
        await second.release()

    async def test_disconnected_waiter_does_not_consume_capacity(self):
        controller = AdmissionController(1, 1, 1, 30)
        first = await controller.acquire("default", connected)

        async def disconnected() -> bool:
            return True

        with self.assertRaises(ClientDisconnected):
            await controller.acquire("default", disconnected)

        self.assertEqual(controller.active, 1)
        self.assertEqual(controller.waiting, 0)
        await first.release()


if __name__ == "__main__":
    unittest.main()

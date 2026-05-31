"""Locust load test scenarios against the /predict endpoint (RF10.3).

Generates realistic random payloads matching the PredictRequest schema
and exercises the API under sustained and spike load patterns.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


# ---------------------------------------------------------------------------
# Realistic data generators for property features
# ---------------------------------------------------------------------------

STATUSES = ["for_sale", "ready_to_build", "sold"]

CITIES = [
    "Austin", "Houston", "Dallas", "San Antonio", "Denver",
    "Phoenix", "Seattle", "Portland", "Miami", "Orlando",
    "Atlanta", "Chicago", "Boston", "New York", "Los Angeles",
    "San Francisco", "Nashville", "Charlotte", "Tampa", "Raleigh",
]

STATES = [
    "Texas", "Colorado", "Arizona", "Washington", "Oregon",
    "Florida", "Georgia", "Illinois", "Massachusetts", "New York",
    "California", "Tennessee", "North Carolina", "Virginia", "Ohio",
]

ZIP_CODES = [
    78701, 77001, 75201, 78201, 80201,
    85001, 98101, 97201, 33101, 32801,
    30301, 60601, 21010, 10001, 90001,
]


def generate_property_payload() -> dict:
    """Generate a realistic random property payload for /predict."""
    return {
        "brokered_by": round(random.uniform(1.0, 500.0), 1),
        "status": random.choice(STATUSES),
        "bed": float(random.randint(1, 6)),
        "bath": float(random.randint(1, 4)),
        "acre_lot": round(random.uniform(0.01, 5.0), 3),
        "street": round(random.uniform(1.0, 9999.0), 1),
        "city": random.choice(CITIES),
        "state": random.choice(STATES),
        "zip_code": float(random.choice(ZIP_CODES)),
        "house_size": round(random.uniform(500.0, 5000.0), 1),
        "prev_sold_year": float(random.randint(1990, 2023)),
    }


# ---------------------------------------------------------------------------
# Locust user classes
# ---------------------------------------------------------------------------


class PredictUser(HttpUser):
    """Simulates a standard user making prediction requests.

    Wait time between requests: 1-3 seconds (sustained load pattern).
    """

    wait_time = between(1, 3)

    @task(10)
    def predict_property(self) -> None:
        """POST /predict with a random property payload."""
        payload = generate_property_payload()
        self.client.post(
            "/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/predict",
        )

    @task(1)
    def health_check(self) -> None:
        """GET /health to verify API availability."""
        self.client.get("/health", name="/health")


class AggressiveUser(HttpUser):
    """Simulates a high-frequency client (spike load pattern).

    Wait time between requests: 0.1-0.5 seconds.
    """

    wait_time = between(0.1, 0.5)
    weight = 1  # Lower weight: fewer aggressive users in the mix

    @task
    def predict_rapid_fire(self) -> None:
        """POST /predict with rapid successive requests."""
        payload = generate_property_payload()
        self.client.post(
            "/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="/predict",
        )

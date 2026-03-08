from payos import PayOS

payos = PayOS(
    client_id="87ef075f-9c12-44d1-b483-8e0dab0a374d",
    api_key="04efac8d-b0d7-4f6a-8e5d-ccf5cab60bb8",
    checksum_key="fccc1b99c91a2fc49f953f1bb3fc220142a597f2e99c9f7b1ed09f3028997b29"
)

data = {
    "orderCode": 233456,
    "amount": 2000,
    "description": "TEST",
    "cancelUrl": "http://localhost",
    "returnUrl": "http://localhost"
}

link = payos.payment_requests.create(data)

print(link.checkout_url)
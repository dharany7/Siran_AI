"""
test_auth.py -- End-to-end authentication test.

Tests:
  1. POST /auth/signup  -- creates a driver account
  2. POST /auth/signup  -- re-signup with same phone -> expect 409 Conflict
  3. POST /auth/login   -- verify password, get JWT
  4. POST /dispatch     -- call with Bearer token (expect 400 "sim not running",
                          which proves auth middleware passed)
  5. POST /dispatch     -- call WITHOUT token (expect 401 Unauthorized)
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8080"
SEP  = "=" * 66

# Unique phone per run so repeated runs don't collide
TEST_PHONE    = f"+9198765{int(time.time()) % 100000:05d}"
TEST_PASSWORD = "siren@secure123"
TEST_PLATE    = f"TN09AX{int(time.time()) % 10000:04d}"
print(f"Using phone={TEST_PHONE}  plate={TEST_PLATE}")
print()


def post(path, body, token=None):
    url  = BASE + path
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code


# -- 1. Signup -----------------------------------------------------------------
print(SEP)
print("TEST 1 -- POST /auth/signup (new account)")
body, status = post("/auth/signup", {
    "phone_number":    TEST_PHONE,
    "password":        TEST_PASSWORD,
    "license_number":  "DL-TN-20240001",
    "ambulance_plate": TEST_PLATE,
})
print(f"  HTTP {status}")
print(f"  {json.dumps(body, indent=2)}")
assert status == 201, f"Expected 201, got {status}"

# -- 2. Duplicate signup -------------------------------------------------------
print(SEP)
print("TEST 2 - POST /auth/signup (duplicate phone -> expect 409)")
body2, status2 = post("/auth/signup", {
    "phone_number":    TEST_PHONE,
    "password":        "anything",
    "license_number":  "DL-DUPLICATE",
    "ambulance_plate": "TN09AX5000",
})
print(f"  HTTP {status2}")
print(f"  {json.dumps(body2, indent=2)}")
assert status2 == 409, f"Expected 409, got {status2}"

# -- 3. Login ------------------------------------------------------------------
print(SEP)
print("TEST 3 -- POST /auth/login (get JWT)")
login_body, login_status = post("/auth/login", {
    "phone_number": TEST_PHONE,
    "password":     TEST_PASSWORD,
})
print(f"  HTTP {login_status}")
print(f"  token_type   : {login_body.get('token_type')}")
token = login_body.get("access_token", "")
print(f"  access_token : {token}")
assert login_status == 200, f"Expected 200, got {login_status}"
assert token.startswith("eyJ"), "Token does not look like a JWT"

# -- 4. Protected dispatch WITH token -----------------------------------------
print(SEP)
print("TEST 4 -- POST /dispatch WITH Bearer token (expect 400 sim-not-running)")
d_body, d_status = post("/dispatch", {"start": "1", "destination": "3"}, token=token)
print(f"  HTTP {d_status}")
print(f"  {json.dumps(d_body, indent=2)}")
assert d_status == 400, (
    f"Expected 400 (sim not running), got {d_status}.\n"
    "If 401 -> auth dependency failed. If 200 -> sim was running (that's OK too)."
)
print("  [OK] Auth passed -- dispatch endpoint was reached (sim not running as expected)")

# -- 5. Protected dispatch WITHOUT token ---------------------------------------
print(SEP)
print("TEST 5 -- POST /dispatch WITHOUT token (expect 401)")
noauth_body, noauth_status = post("/dispatch", {"start": "1", "destination": "3"})
print(f"  HTTP {noauth_status}")
print(f"  {json.dumps(noauth_body, indent=2)}")
assert noauth_status == 401, f"Expected 401, got {noauth_status}"
print("  [OK] Unauthenticated request correctly rejected")

# -- Summary -------------------------------------------------------------------
print(SEP)
print("  ALL 5 TESTS PASSED")
print(SEP)

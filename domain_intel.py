
import requests
import tldextract
from urllib.parse import urlsplit, quote

BOOTSTRAP = "https://data.iana.org/rdap/dns.json"

def get_registered_domain(hostname):
    result = tldextract.extract(hostname)
    return result.top_domain_under_public_suffix

def lookup_domain(hostname):
    domain = get_registered_domain(hostname)

    if not domain:
        return {"error": "No registered domain found"}

    bootstrap = requests.get(BOOTSTRAP, timeout=20)
    bootstrap.raise_for_status()

    tld = domain.split(".")[-1]
    server = None

    for tlds, servers in bootstrap.json()["services"]:
        if tld in tlds and servers:
            server = servers[0]
            break

    if not server:
        return {"error": "No RDAP registry found"}

    parsed = urlsplit(server)

    if parsed.scheme != "https" or not parsed.hostname:
        return {"error": "Invalid RDAP registry endpoint"}

    # Initial prototype: no automatic redirects
    url = server.rstrip("/") + "/domain/" + quote(domain, safe="")

    response = requests.get(
        url,
        timeout=15,
        allow_redirects=False,
        headers={"Accept": "application/rdap+json"}
    )

    if response.status_code != 200:
        return {"error": f"RDAP returned HTTP {response.status_code}"}

    data = response.json()

    events = {
        e.get("eventAction"): e.get("eventDate")
        for e in data.get("events", [])
    }

    return {
        "domain": data.get("ldhName", domain),
        "registration_date": events.get("registration"),
        "expiration_date": events.get("expiration"),
        "last_changed": events.get("last changed"),
        "status": data.get("status", [])
    }

if __name__ == "__main__":
    print(lookup_domain("www.example.com"))

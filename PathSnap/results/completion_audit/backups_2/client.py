import requests
from urllib.parse import urlparse
class ApiClient:
    def __init__(self,base_url):
        self.url=base_url.rstrip("/")
        parsed=urlparse(self.url)
        if parsed.scheme not in ("http","https") or not parsed.hostname: raise ValueError("Enter a full HTTP API address")
    def trips(self):
        response=requests.get(self.url+"/trips",timeout=10); response.raise_for_status(); return response.json()
    def health(self):
        response=requests.get(self.url+"/health",timeout=10); response.raise_for_status(); return response.json()
    def predict(self,trip_id):
        response=requests.post(self.url+"/predict",json={"trip_id":trip_id},timeout=180)
        response.raise_for_status(); return response.json()


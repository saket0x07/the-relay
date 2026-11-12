import json
import httpx

def test_query():
    query = "What is the dimensionality d_model in Attention Is All You Need?"
    print(f"Submitting query: '{query}'\n")

    with httpx.Client(timeout=40.0) as client:
        with client.stream("POST", "http://127.0.0.1:8000/api/ask", json={"query": query}) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    event = json.loads(payload)
                    e_type = event.get("type")
                    if e_type == "token":
                        print(event.get("delta", ""), end="", flush=True)
                    elif e_type == "metadata":
                        print(f"--> Admitted: req_id={event.get('request_id')}")
                        print("--> Streaming Answer: ", end="", flush=True)
                    elif e_type == "done":
                        print(f"\n\n--> Complete [Status: {event.get('status')}, Level: {event.get('degradation_level')}]")
                        print(f"--> Spans: {event.get('spans')}")

if __name__ == "__main__":
    test_query()


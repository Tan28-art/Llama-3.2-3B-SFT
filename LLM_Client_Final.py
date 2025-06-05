# llm_client_final.py 
import requests
import argparse
import json

def query_api(user_query: str, host: str = "127.0.0.1", port: int = 8004):
    """ Sends the user query to the FastAPI server and returns the response string or None on error. """
    url = f"http://{host}:{port}/generate"
    payload = {"user_query": user_query}
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180) # Add timeout
        response.raise_for_status()
        response_data = response.json()
        # Check for server-side errors indicated in the response
        if "error" in response_data:
             print(f"\n--- Server Error ---")
             print(f"{response_data['error']}")
             print("--------------------")
             return None
        return response_data.get("response", "[API Error: No 'response' key found]")
    except requests.exceptions.Timeout:
         print(f"\n--- API Request Error: Request timed out after 180 seconds ---")
         return None
    except requests.exceptions.RequestException as e:
        print(f"\n--- API Request Error ---\n{e}\n------------------------"); return None
    except json.JSONDecodeError: print(f"\n--- API Response Error ---\n{response.text}\n------------------------"); return None

def main():
    parser = argparse.ArgumentParser(description="Chat client for the LLM API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8004, help="Port number")
    args = parser.parse_args()
    print("--- LLM Chat Client ---"); print(f"Connecting to server at http://{args.host}:{args.port}"); print("Type 'exit' or 'quit' to end."); print("-" * 25)
    while True:
        try: user_input = input("You: ")
        except EOFError: print("\nExiting..."); break
        if user_input.lower().strip() in ("exit", "quit"): print("Exiting chat. Goodbye!"); break
        if not user_input.strip(): continue

        # Query the API
        api_response = query_api(user_input, host=args.host, port=args.port)

        if api_response is not None:
            cleaned_response = api_response
            undesired_prefix = "... Instruction:" # Check if it generated prompt structure
            if undesired_prefix in cleaned_response:
                 parts = cleaned_response.split("### Response:")
                 if len(parts) > 1:
                      cleaned_response = parts[-1].strip()
                 else: 
                      cleaned_response = cleaned_response.split(undesired_prefix)[-1].strip()

            # Print the cleaned response
            print(f"Assistant: {cleaned_response}")


if __name__ == "__main__":
    main()
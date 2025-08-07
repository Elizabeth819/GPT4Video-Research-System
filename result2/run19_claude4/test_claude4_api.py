#!/usr/bin/env python3
"""
Test script for Claude 4 API connection
Tests the connection to globalai.vip with the provided API key
"""

import http.client
import json
import ssl

def test_claude4_api():
    """Test the Claude 4 API connection"""
    
    # API configuration
    host = "globalai.vip"
    endpoint = "/v1/messages"
    api_key = "sk-MsLn3MaO80cZdf6VTO9hPI3sPQNqzgtso5DfDnDXzqvl30rf"
    model = "claude-sonnet-4-20250514"
    
    # Test message
    test_message = "Hello, please respond with 'Claude 4 API working' to confirm you are Claude 4."
    
    # Request payload
    payload = {
        "model": model,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": test_message
            }
        ]
    }
    
    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'anthropic-version': '2023-06-01'
    }
    
    try:
        print(f"Testing Claude 4 API connection...")
        print(f"Host: {host}")
        print(f"Model: {model}")
        print(f"Test message: {test_message}")
        print("-" * 50)
        
        # Create HTTPS connection
        context = ssl.create_default_context()
        conn = http.client.HTTPSConnection(host, context=context)
        
        # Send request
        conn.request("POST", endpoint, json.dumps(payload), headers)
        
        # Get response
        response = conn.getresponse()
        response_data = response.read().decode('utf-8')
        
        print(f"Response Status: {response.status} {response.reason}")
        print(f"Response Headers: {dict(response.getheaders())}")
        print("-" * 50)
        
        if response.status == 200:
            try:
                json_response = json.loads(response_data)
                print("✅ SUCCESS: API connection working!")
                print(f"Response: {json.dumps(json_response, indent=2)}")
                
                # Extract the actual response content
                if 'content' in json_response and len(json_response['content']) > 0:
                    content = json_response['content'][0].get('text', '')
                    print(f"\nClaude 4 Response: {content}")
                    
                    if 'Claude 4 API working' in content:
                        print("✅ Claude 4 confirmed working!")
                    else:
                        print("⚠️  Response received but didn't contain expected confirmation")
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Raw response: {response_data}")
        else:
            print(f"❌ API request failed with status {response.status}")
            print(f"Response: {response_data}")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
    
    return response.status == 200

if __name__ == "__main__":
    success = test_claude4_api()
    if success:
        print("\n🎉 Claude 4 API test completed successfully!")
    else:
        print("\n💥 Claude 4 API test failed!")
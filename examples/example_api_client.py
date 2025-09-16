#!/usr/bin/env python3
"""
OpticXT API Client Example

This script demonstrates how to interact with the OpticXT inference API
using Python requests library.
"""

import requests
import json
import time
from pathlib import Path


class OpticXTClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/inference"

    def text_inference(self, text, task_context=None):
        """Send text-only inference request"""
        data = {"text": text}
        if task_context:
            data["task_context"] = task_context

        response = requests.post(self.endpoint, files=data)
        return response.json()

    def image_inference(self, text, image_path, task_context=None):
        """Send image inference request"""
        data = {"text": text}
        if task_context:
            data["task_context"] = task_context

        with open(image_path, "rb") as f:
            files = {"image": f}
            response = requests.post(self.endpoint, data=data, files=files)

        return response.json()

    def status_check(self):
        """Check current robot status"""
        response = requests.post(self.endpoint)
        return response.json()

    def is_server_running(self):
        """Check if the API server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            try:
                # Try the inference endpoint as fallback
                response = requests.post(
                    self.endpoint, files={"text": "ping"}, timeout=5
                )
                return response.status_code in [
                    200,
                    400,
                ]  # 400 is also valid (bad request but server running)
            except requests.exceptions.RequestException:
                return False


def main():
    print("OpticXT API Client Example")
    print("=" * 30)

    client = OpticXTClient()

    # Check if server is running
    if not client.is_server_running():
        print("❌ OpticXT API server is not running!")
        print("Start it with: cargo run --release -- --api-server")
        return

    print("✅ API server is running")
    print()

    # Example 1: Simple text inference
    print("1. Text-only inference:")
    try:
        result = client.text_inference("What can you see in the camera feed right now?")
        print(f"Response: {result['text'][:100]}...")
        print(
            f"Tokens: {result['tokens_generated']}, Time: {result['processing_time_ms']}ms"
        )
    except Exception as e:
        print(f"Error: {e}")

    print()

    # Example 2: Robot command
    print("2. Robot task command:")
    try:
        result = client.text_inference(
            "Move forward slowly and describe what you observe",
            task_context="Indoor navigation test",
        )
        print(f"Response: {result['text'][:100]}...")
        print(f"Status: {result['status']}")
        if result.get("current_task"):
            print(f"Current task: {result['current_task']}")
    except Exception as e:
        print(f"Error: {e}")

    print()

    # Example 3: Status check
    print("3. Status check:")
    try:
        result = client.status_check()
        print(f"Response: {result['text'][:100]}...")
        print(f"Status: {result['status']}")
    except Exception as e:
        print(f"Error: {e}")

    print()

    # Example 4: Image analysis (if test image exists)
    test_image = Path("test_image.jpg")
    if test_image.exists():
        print("4. Image analysis:")
        try:
            result = client.image_inference(
                "Analyze this image and describe what you see in detail",
                str(test_image),
            )
            print(f"Response: {result['text'][:150]}...")
            print(f"Processing time: {result['processing_time_ms']}ms")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("4. Image analysis: Skipped (no test_image.jpg found)")

    print()
    print("API client example completed!")


if __name__ == "__main__":
    main()

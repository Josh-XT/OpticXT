#!/bin/bash

# OpticXT API Test Script
# This script demonstrates how to use the OpticXT inference API

API_BASE="http://localhost:8080"
ENDPOINT="/v1/inference"

echo "Testing OpticXT API endpoints..."
echo "================================"

# Test 1: Simple text inference
echo "Test 1: Text-only inference"
curl -X POST "${API_BASE}${ENDPOINT}" \
  -F "text=Hello, what can you see right now?" \
  -H "Accept: application/json" | jq .
echo -e "\n"

# Test 2: Robot command
echo "Test 2: Robot task command"
curl -X POST "${API_BASE}${ENDPOINT}" \
  -F "text=Move forward slowly and describe what you see" \
  -F "task_context=Indoor navigation test" \
  -H "Accept: application/json" | jq .
echo -e "\n"

# Test 3: Status check (if task is running)
echo "Test 3: Current status check"
curl -X POST "${API_BASE}${ENDPOINT}" \
  -H "Accept: application/json" | jq .
echo -e "\n"

# Test 4: Image analysis (if test image exists)
if [ -f "test_image.jpg" ]; then
    echo "Test 4: Image analysis"
    curl -X POST "${API_BASE}${ENDPOINT}" \
      -F "text=Analyze this image and describe what you see" \
      -F "image=@test_image.jpg" \
      -H "Accept: application/json" | jq .
    echo -e "\n"
else
    echo "Test 4: Skipped (no test_image.jpg found)"
fi

echo "API tests completed!"

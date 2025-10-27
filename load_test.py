# Change load test to sequential (not concurrent)
import requests
import time
from PIL import Image
import numpy as np

# Create test image
img = Image.fromarray(np.random.randint(0,255,(32,32,3),dtype=np.uint8))
img.save('test_load.png')

# Sequential test
print("Running load test...")
start = time.time()
success = 0

for i in range(100):
    try:
        with open('test_load.png', 'rb') as f:
            response = requests.post(
                'http://localhost:8000/predict',
                files={'file': f}
            )
        if response.status_code == 200:
            success += 1
            if i == 0:
                print(f"First response: {response.json()['cached']}")
            if i == 1:
                print(f"Second response (should be cached): {response.json()['cached']}")
    except:
        pass

duration = time.time() - start
print(f"\nCompleted: {success}/100 requests")
print(f"Duration: {duration:.2f}s")
print(f"Throughput: {success/duration:.2f} req/s")

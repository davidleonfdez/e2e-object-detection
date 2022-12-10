from pathlib import Path
import numpy as np
import requests
import time
from torchserve.scripts.run_local_docker import run_with_docker_py as run_torchserve


def test_handlers():
    container = None
    model_name = 'object_detector'
    root_tests_path = Path(__file__).parent.parent
    model_store_path = root_tests_path/'inputs'/'model_store'
    input_path = root_tests_path/'inputs'/'cyclist_img.jpg'

    try:
        with open(input_path, 'rb') as f:
            img_bytes = f.read()

        container = run_torchserve(str(model_store_path))
        
        # Give TorchServe time to start
        time.sleep(30)

        resp = requests.post(f"http://localhost:8080/predictions/{model_name}", data=img_bytes)
        result = resp.json() 

        # The expected results have been calculated beforehand for the example model and image
        expected_coords = np.array([642.0, 304.0, 741.0, 631.0])
        assert len(result) == 1
        assert np.allclose(np.array(result[0]['coords']), expected_coords)
        assert 0.81 < result[0]['conf'] < 0.82
    finally:
        if container: container.stop()

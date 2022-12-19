from pathlib import Path
import pytest
import random
import string
import subprocess
import tempfile


@pytest.mark.parametrize("model_filename", ["cyclist_detector.torchscript.pt", "cyclist_detector.onnx"])
def test_generate_mar(model_filename):
    root_dir = Path(__file__).parent.parent.parent
    script_path = root_dir/'objdetserver'/'scripts'/'generate_mar.py'
    out_modelname = ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=8))
    input_model_path = Path(__file__).parent.parent/'inputs'/model_filename

    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([
            "python", str(script_path), "--model-name", out_modelname, "--out-path", tmpdir, str(input_model_path)
        ])

        out_filepath = Path(tmpdir)/f"{out_modelname}.mar"
        assert out_filepath.exists()

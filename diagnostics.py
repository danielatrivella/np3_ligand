# from https://github.com/CiSong10/MinkowskiEngine/tree/cuda12-installation
import sys
import os
import platform
import subprocess

def parse_nvidia_smi():
    sp = subprocess.Popen(
        ["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_dict = dict()
    for item in sp.communicate()[0].decode("utf-8").split("\n"):
        if item.count(":") == 1:
            key, val = [i.strip() for i in item.split(":")]
            out_dict[key] = val
    return out_dict


def print_diagnostics():
    print("==========System==========")
    print(platform.platform())
    os.system("cat /etc/lsb-release")
    print(sys.version)

    print("==========NVIDIA-SMI==========")
    os.system("which nvidia-smi")
    for k, v in parse_nvidia_smi().items():
        if "version" in k.lower():
            print(k, v)

    print("==========NVCC==========")
    os.system("which nvcc")
    os.system("nvcc --version")

    print("==========CC==========")
    CC = "c++"
    if "CC" in os.environ or "CXX" in os.environ:
        # distutils only checks CC not CXX
        if "CXX" in os.environ:
            os.environ["CC"] = os.environ["CXX"]
            CC = os.environ["CXX"]
        else:
            CC = os.environ["CC"]
        print(f"CC={CC}")
    os.system(f"which {CC}")
    os.system(f"{CC} --version")

    print("\n==========Python==========")
    print('Python version:', sys.version)

    print("==========Pytorch==========")
    try:
        import torch

        print('Torch version:', torch.__version__)
        print('Torch CUDA version:', torch.version.cuda)
        print('Torch CUDA available:', torch.cuda.is_available())
    except ImportError or ModuleNotFoundError:
        print("torch not installed")

    print("==========MinkowskiEngine==========")
    try:
        import MinkowskiEngine as ME
        print(ME.__version__)
        print(f"MinkowskiEngine compiled with CUDA Support: {ME.is_cuda_available()}")
        print(f"NVCC version MinkowskiEngine is compiled: {ME.cuda_version()}")
        print(f"CUDART version MinkowskiEngine is compiled: {ME.cudart_version()}")
    except ImportError or ModuleNotFoundError or AttributeError:
        print("MinkowskiEngine not installed")


if __name__ == "__main__":
    print_diagnostics()

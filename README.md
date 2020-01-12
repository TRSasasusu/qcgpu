# QCGPU (CUDA version)

Forked from [QCGPU](https://github.com/libtangle/qcgpu), which supports OpenCL.

Open Source, High Performance & Hardware Accelerated, Quantum Computer
Simulator. Read the [research paper](https://arxiv.org/abs/1805.00988).

**This branch settings set CUDA as default** ([state.py](https://github.com/TRSasasusu/qcgpu/blob/feature/cuda-bydefault/qcgpu/state.py#L48) has `use_cuda=True`).

**Features:**

  - Written with OpenCL and CUDA. Accelerated your simulations with GPUs and other
    accelerators, while still running cross device and cross platform.
  - Simulation of arbitrary quantum circuits
  - Includes example algorithm implementations
  - Support for arbitrary gate creation/application, with many built in.

## Installing

Install pycuda by:

```bash
$ pip install pycuda
```

Then, install this package:

```bash
$ git clone https://github.com/TRSasasusu/qcgpu
$ git checkout feature/cuda-bydefault
$ python ./setup.py bdist_wheel
$ pip install dist/*.whl
```

You can use this CUDA version QCGPU with Qiskit by [Qiskit QCGPU Provider (CUDA version)](https://github.com/TRSasasusu/qiskit-qcgpu-provider/tree/feature/cuda).

For more information read the full [installing docs](https://qcgpu.github.io/qcgpu/install.html).

import os
import random
import numpy as np
import pycuda.autoinit
from pycuda import driver, gpuarray
from pycuda.compiler import SourceModule
from pycuda.reduction import ReductionKernel
from collections import defaultdict

mod = SourceModule('''
#include <pycuda-complex.hpp>

/*
 * Returns the nth number where a given digit
 * is cleared in the binary representation of the number
 */
__device__ int nth_cleared(int n, int target)
{
    int mask = (1 << target) - 1;
    int not_mask = ~mask;

    return (n & mask) | ((n & not_mask) << 1);
}

///////////////////////////////////////////////
// KERNELS
///////////////////////////////////////////////

/*
 * Applies a single qubit gate to the register.
 * The gate matrix must be given in the form:
 *
 *  A B
 *  C D
 */
__global__ void apply_gate(
    pycuda::complex<float> *amplitudes,
    int target,
    pycuda::complex<float> A,
    pycuda::complex<float> B,
    pycuda::complex<float> C,
    pycuda::complex<float> D)
{
    int const global_id = blockDim.x * blockIdx.x + threadIdx.x;

    int const zero_state = nth_cleared(global_id, target);

    int const one_state = zero_state | (1 << target);

    pycuda::complex<float> const zero_amp = amplitudes[zero_state];
    pycuda::complex<float> const one_amp = amplitudes[one_state];

    amplitudes[zero_state] = A * zero_amp + B * one_amp;
    amplitudes[one_state] = D * one_amp + C * zero_amp;
}

/*
 * Applies a controlled single qubit gate to the register.
 */
__global__ void apply_controlled_gate(
    pycuda::complex<float> *amplitudes,
    int control,
    int target,
    pycuda::complex<float> A,
    pycuda::complex<float> B,
    pycuda::complex<float> C,
    pycuda::complex<float> D)
{
    int const global_id = blockDim.x * blockIdx.x + threadIdx.x;
    int const zero_state = nth_cleared(global_id, target);
    int const one_state = zero_state | (1 << target); // Set the target bit

    int const control_val_zero = (((1 << control) & zero_state) > 0) ? 1 : 0;
    int const control_val_one = (((1 << control) & one_state) > 0) ? 1 : 0;

    pycuda::complex<float> const zero_amp = amplitudes[zero_state];
    pycuda::complex<float> const one_amp = amplitudes[one_state];

    if (control_val_zero == 1)
    {
        amplitudes[zero_state] = A * zero_amp + B * one_amp;
    }

    if (control_val_one == 1)
    {
        amplitudes[one_state] = D * one_amp + C * zero_amp;
    }
}

/*
 * Applies a controlled-controlled single qubit gate to the register.
 */
__global__ void apply_controlled_controlled_gate(
    pycuda::complex<float> *amplitudes,
    int control,
    int control_2,
    int target,
    pycuda::complex<float> A,
    pycuda::complex<float> B,
    pycuda::complex<float> C,
    pycuda::complex<float> D)
{
    int const global_id = blockDim.x * blockIdx.x + threadIdx.x;
    int const zero_state = nth_cleared(global_id, target);
    int const one_state = zero_state | (1 << target); // Set the target bit

    int const control_val_zero = (((1 << control) & zero_state) > 0) ? 1 : 0;
    int const control_val_one = (((1 << control) & one_state) > 0) ? 1 : 0;
    int const control_val_two_zero = (((1 << control_2) & zero_state) > 0) ? 1 : 0;
    int const control_val_two_one = (((1 << control_2) & one_state) > 0) ? 1 : 0;

    pycuda::complex<float> const zero_amp = amplitudes[zero_state];
    pycuda::complex<float> const one_amp = amplitudes[one_state];

    if (control_val_zero == 1 && control_val_two_zero == 1)
    {
        amplitudes[zero_state] = A * zero_amp + B * one_amp;
    }

    if (control_val_one == 1 && control_val_two_one == 1)
    {
        amplitudes[one_state] = D * one_amp + C * zero_amp;
    }
}

/**
 * Get a single amplitude
 */
__global__ void get_single_amplitude(
    pycuda::complex<float> *const amplitudes,
    pycuda::complex<float> *out,
    int i)
{
    out[0] = amplitudes[i];
}

/**
 * Calculates The Probabilities Of A State Vector
 */
__global__ void calculate_probabilities(
    pycuda::complex<float> *const amplitudes,
    float *probabilities)
{
    int const state = blockDim.x * blockIdx.x + threadIdx.x;
    pycuda::complex<float> amp = amplitudes[state];

    probabilities[state] = pycuda::abs(amp * amp);
}

/**
 * Collapses a qubit in the register
 */
__global__ void collapse(
    pycuda::complex<float> *amplitudes,
    int const target,
    int const outcome, 
    float const norm)
{
    int const state = blockDim.x * blockIdx.x + threadIdx.x;

    if (((state >> target) & 1) == outcome) {
        amplitudes[state] = amplitudes[state] * pycuda::complex<float>(norm, 0.0);
    }
    else
    {
        amplitudes[state] = pycuda::complex<float>(0.0, 0.0);
    }
}
''')

# Setup the Cuda functions here to not prompt every execution
class _Function:
    def __init__(self):
        self.apply_gate = mod.get_function('apply_gate')
        self.apply_controlled_gate = mod.get_function('apply_controlled_gate')
        self.apply_controlled_controlled_gate = mod.get_function('apply_controlled_controlled_gate')
        self.get_single_amplitude = mod.get_function('get_single_amplitude')
        self.calculate_probabilities = mod.get_function('calculate_probabilities')
        self.collapse = mod.get_function('collapse')
_function = None


class CudaBackend:
    """
    A class for the Cuda backend to the simulator.

    This class shouldn't be used directly, as many of the
    methods don't have the same input checking as the State
    class.
    """

    # @profile
    def __init__(self, num_qubits, dtype=np.complex64):
        if not _function:
            create_function()
        
        """
        Initialize a new Cuda Backend

        Takes an argument of the number of qubits to use
        in the register, and returns the backend.
        """
        self.num_qubits = num_qubits
        self.dtype = dtype

        # Buffer for the state vector
        self.buffer = gpuarray.to_gpu(np.eye(1, 2**num_qubits, dtype=dtype))

    def calc_block_grid_size(self):
        max_threads_per_block = driver.Device(0).max_threads_per_block
        whole_size = 2 ** self.num_qubits
        if whole_size < max_threads_per_block:
            return (whole_size, 1, 1), (1, 1)
        else:
            return (max_threads_per_block, 1, 1), (whole_size // max_threads_per_block, 1)

    def apply_gate(self, gate, target):
        """Applies a gate to the quantum register"""
        block, grid = self.calc_block_grid_size()
        _function.apply_gate(
            self.buffer.gpudata,
            np.int32(target),
            self.dtype(gate.a),
            self.dtype(gate.b),
            self.dtype(gate.c),
            self.dtype(gate.d),
            block=(block[0] // 2, 1, 1),
            grid=grid
        )

    def apply_controlled_gate(self, gate, control, target):
        """Applies a controlled gate to the quantum register"""

        block, grid = self.calc_block_grid_size()
        _function.apply_controlled_gate(
            self.buffer.gpudata,
            np.int32(control),
            np.int32(target),
            self.dtype(gate.a),
            self.dtype(gate.b),
            self.dtype(gate.c),
            self.dtype(gate.d),
            block=(block[0] // 2, 1, 1),
            grid=grid
        )
    
    def apply_controlled_controlled_gate(self, gate, control1, control2, target):
        """Applies a controlled controlled gate (such as a toffoli gate) to the quantum register"""

        block, grid = self.calc_block_grid_size()
        _function.apply_controlled_controlled_gate(
            self.buffer.gpudata,
            np.int32(control1),
            np.int32(control2),
            np.int32(target),
            self.dtype(gate.a),
            self.dtype(gate.b),
            self.dtype(gate.c),
            self.dtype(gate.d),
            block=(block[0] // 2, 1, 1),
            grid=grid
        )

    def seed(self, val):
        random.seed(val)
        
    def measure(self, samples=1):
        """Measure the state of a register"""
        # This is a really horrible method that needs a rewrite - the memory
        # is attrocious

        probabilities = self.probabilities()
        # print(probabilities)
        # print(np.sum(self.amplitudes()))
        choices = np.random.choice(
            np.arange(0, 2**self.num_qubits), 
            samples, 
            p=probabilities
        )
        
        results = defaultdict(int)
        for i in choices:
            results[np.binary_repr(i, width=self.num_qubits)] += 1
        
        return dict(results)

    def measure_first(self, num, samples):
        probabilities = self.probabilities()
        # print(probabilities)
        # print(np.sum(self.amplitudes()))
        choices = np.random.choice(
            np.arange(0, 2**self.num_qubits), 
            samples, 
            p=probabilities
        )
        
        results = defaultdict(int)
        for i in choices:
            key = np.binary_repr(i, width=self.num_qubits)[-num:]
            results[key] += 1
        
        return dict(results)
       

    def qubit_probability(self, target):
        """Get the probability of a single qubit begin measured as '0'"""

        preamble = """
        #include <pycuda-complex.hpp>

        float probability(int target, int i, pycuda::complex<float> amp) {
            if ((i & (1 << target )) != 0) {
                return 0;
            }
            // return 6.0;
            float abs = pycuda::abs(amp);
            return abs * abs;
        }
        """


        kernel = ReductionKernel(
            np.float, 
            neutral = "0",
            reduce_expr="a + b",
            map_expr="probability(target, i, amps[i])",
            arguments="pycuda::complex<float> *amps, int target",
            preamble=preamble
        )

        return kernel(self.buffer, target).get()
        
    def reset(self, target):
        probability_of_0 = self.qubit_probability(target)
        norm = 1 / np.sqrt(probability_of_0)

        block, grid = self.calc_block_grid_size()
        _function.collapse(
            self.buffer.gpudata,
            np.int32(target),
            np.int32(0),
            np.float32(norm),
            block=block,
            grid=grid
        )

    def measure_collapse(self, target):
        probability_of_0 = self.qubit_probability(target)
        random_number = random.random()

        if random_number <= probability_of_0:
            outcome = '0'
            norm = 1 / np.sqrt(probability_of_0)
        else:
            outcome = '1'
            norm = 1 / np.sqrt(1 - probability_of_0)

        block, grid = self.calc_block_grid_size()
        _function.collapse(
            self.buffer.gpudata,
            np.int32(target),
            np.int32(outcome),
            np.float32(norm),
            block=block,
            grid=grid
        )
        return outcome

    def measure_qubit(self, target, samples):
        probability_of_0 = self.qubit_probability(target)

        choices = np.random.choice(
            [0, 1], 
            samples, 
            p=[probability_of_0, 1-probability_of_0]
        )
        
        results = defaultdict(int)
        for i in choices:
            results[np.binary_repr(i, width=1)] += 1
        
        return dict(results)

    def single_amplitude(self, i):
        """Gets a single probability amplitude"""
        out = np.empty(1, dtype=np.complex64)

        _function.get_single_amplitude(
            self.buffer.gpudata,
            driver.Out(out),
            np.int32(i),
            block=(1, 0, 0)
        )

        return out[0]

    def amplitudes(self):
        """Gets the probability amplitudes"""
        return self.buffer.get()
    
    def probabilities(self):
        """Gets the squared absolute value of each of the amplitudes"""
        out = np.zeros(2**self.num_qubits, dtype=np.float32)

        block, grid = self.calc_block_grid_size()
        _function.calculate_probabilities(
            self.buffer.gpudata,
            driver.Out(out),
            block=block,
            grid=grid
        )

        return out
        
    def release(self):
        self.buffer.gpudata.free()
    
def create_function():
    global _function
    _function = _Function()

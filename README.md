# Matrix Multiply
Several common methods of matrix multiplication are implemented on CPU and Nvidia GPU using C++11 and CUDA. The performance benefits of each optimization method were simply tested.

## CPU
- naive
- reordering
- tiling
- strassen
- coppersmith-winograd

## Nvidia GPU
- cublas
- naive
- kahan
- shared_memory

# Compile
## Environment
- OS: Linux
- Cmake Version: >= 3.8
- GCC Version: >= 4.8
- CUDA Version: 11.4 (best)
- CUDA Driver Version: 470.129.06 (best)

## Clone
```
git clone https://github.com/Bruce-Lee-LY/matrix_multiply.git
```

## Build
```
cd matrix_multiply
./build.sh -t Release -b OFF
./build.sh -t Debug -b ON
```

# Run Sample
```
./run_sample.sh
```

# Performance
- OS: Ubuntu 20.04.4
- CPU: i5-9400F
- GPU: NVIDIA GeForce GTX 1080 Ti
- CUDA Version: 11.4
- CUDA Driver Version: 470.129.06
- Matrix (float): A (512 * 512) * B (512 * 512) = C (512 * 512)

## CPU
|Method|Cost / ms|
|:-:|:-:|
|naive|1238.647|
|reordering|984.445|
|tiling|1000.095|
|strassen|57429.407|
|coppersmith-winograd|77668.238|

## Nvidia GPU
|Method|Cost / ms|
|:-:|:-:|
|cublas|0.100|
|naive|0.613|
|kahan|0.616|
|shared_memory|0.153|

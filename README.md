# LieIPM

This repo contains the implementation of the Lie Group Interior Point Method for trajectory optimization of rigid bodies modeled on $SE(3)$ and $SO(3)\times\mathbb{R}^3$. 

## MATLAB

Run the following two entry files directly:

- `matlab/example/drone/main.m`
- `matlab/example/SE3_vs_SO3/main.m`

## C++

Install `LieIPM-cpp` with:

```bash
cd LieIPM-cpp
mkdir -p build
cd build
cmake ..
make install
```

## test-cpp

Build and run the test with:

```bash
cd test-cpp
mkdir -p build
cd build
cmake ..
make -j
cd ..
bash ./run_all
```
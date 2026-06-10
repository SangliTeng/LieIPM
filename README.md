# LieIPM: a Line-Search Interior Point Method on Matrix Lie Groups

C++ implementation of a **backtracking line-search Interior Point Method** for trajectory optimization of systems evolving on **matrix Lie groups**.

This solver uses:
- [Eigen](https://eigen.tuxfamily.org) for linear algebra
- [MUMPS](https://mumps-solver.org) for sparse matrix factorization

---

## MATLAB

Run the following two entry files directly:

- `matlab/example/drone/main.m`
- `matlab/example/SE3_vs_SO3/main.m`

---

## C++

### Dependencies

- [Eigen3](https://eigen.tuxfamily.org) (`libeigen3-dev`)
- [MUMPS](https://mumps-solver.org) >= 5.6.2 (see note below)
- [OpenBLAS](https://www.openblas.net) (`libopenblas-dev`)

### Install LieIPM-cpp

`LieIPM-cpp` is a **header-only** library. Installing it copies the headers and generates a CMake config so downstream projects can use `find_package(LieIPM)`.

```bash
cd LieIPM-cpp
mkdir -p build && cd build
cmake ..
sudo cmake --build . --target install
```

---

## test-cpp

### Dependencies

In addition to the C++ dependencies above, `test-cpp` requires:

- [matio](https://github.com/tbeu/matio) for reading `.mat` data files (`libmatio-dev`)

> **Note on MUMPS:** Ubuntu 22.04 and older ship MUMPS < 5.6.2. Install from source if needed:
> ```bash
> git clone https://github.com/scivision/mumps.git
> cd mumps
> cmake -B build -DMUMPS_parallel=OFF -DCMAKE_BUILD_TYPE=Release
> cmake --build build
> sudo cmake --install build --prefix /usr/local
> ```

### Build and run

```bash
cd test-cpp
mkdir -p build && cd build
cmake ..
make -j
cd ..
bash ./run_all
```

---

## Citation

Please cite our work as follows.

```bibtex
@INPROCEEDINGS{TengS-RSS-25,
	AUTHOR    = {Sangli Teng AND Tzu-Yuan Lin AND William A. Clark AND Ram Vasudevan AND Maani Ghaffari},
	TITLE     = {{Riemannian Direct Trajectory Optimization of Rigid Bodies on Matrix Lie Groups}},
	BOOKTITLE = {Proceedings of Robotics: Science and Systems},
	YEAR      = {2025},
	ADDRESS   = {LosAngeles, CA, USA},
	MONTH     = {June},
	DOI       = {10.15607/RSS.2025.XXI.120}
}
```

```bibtex
@misc{teng2026lieipmliegroupinterior,
	  title={LieIPM: Lie Group Interior Point Method for Direct Trajectory Optimization of Rigid Bodies},
	  author={Sangli Teng and Ruiqi Zhang and Tzu-Yuan Lin and William A Clark and Mark Mueller and Ram Vasudevan and Maani Ghaffari and Koushil Sreenath},
	  year={2026},
	  eprint={2606.10579},
	  archivePrefix={arXiv},
	  primaryClass={cs.RO},
	  url={https://arxiv.org/abs/2606.10579},
}
```
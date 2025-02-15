# FED: Fast and Efficient Dataset Deduplication with GPU Acceleration

FED is a tool designed for efficient deduplication of data, leveraging modern computational frameworks.

---

## Prerequisites

Before using FED, ensure the following are installed on your system:

- ****CUDA****: For GPU acceleration.
- ****MPI****: For parallel processing.
- ****CMake****: For build automation.

If you use Conda, you can install the prerequisites as follows:

```bash
conda install -c conda-forge mpi4py cmake
conda install -c conda-forge cudatoolkit-dev
```

### System Configuration

The table below summarizes the hardware and software configurations used in this project.
(For more details, please refer to Table 1 in the paper.)

| Component         | Specification                       |
|------------------ |-------------------------------------|
| **CPU**           | 1 × AMD EPYC 7502 32-Core Processor |
| **Memory**        | 8 × 64GB DDR4 DIMM                  |
| **GPU**           | 4 × NVIDIA Tesla V100               |
| **OS**            | Ubuntu 20.04.6 (kernel 5.4.0-100)   |
| **Compiler**      | nvcc 12.4, GCC 9.4                  |
| **GPU Driver**    | 520.61.05                           |
| **MPI Version**   | 4.1                                 |
| **CUDA Version**  | 12.4                                |


---

## Installation

Follow these steps to build FED:

```bash
mkdir build
cd build
cmake ..
make -j
```

---

## Input, Output, and Execution

### Input and  Hyperparameters
- The input directory should contain multiple `.jsonl` files.
- For large JSONL files, it is recommended to split the data into chunks of size `MAX_LINE` and store them separately before proceeding with the deduplication process.
  - The current code assumes a setting where each JSONL file contains MAX_LINE(=30,000) documents.
  - You can adjust the value of `MAX_LINE` based on the available memory, and update the adjusted value in `src/param.h`.
- This code assumes that the JSONL files contain only the `text` field.  
  - If other fields are present, a parsing process will be required. The parsing logic can be found in `src/util.cpp`, but it is currently commented out in this version.
- Other parameters can be modified in src/param.h.
- The default settings are configured for processing RealNews dataset in our environment.

### Output
- After final duplicate removal, the cleaned dataset is stored in JSONL format within the output directory. 

**Note**: The time taken to save the final output file is excluded from the reported times in the paper. This exclusion ensures a fair comparison with the baseline methods.

### Execution

- ####  Single Process Execution
To run the tool in single-process mode:
```bash
./main <input_directory> <output_directory>
```

- #### Using MPI for Parallel Execution in Single-node
To execute our experiment, we utilized **MPI** for parallel processing. Below are the commands for both mpirun and Slurm, which were used to run the code:

Using mpirun
```bash
mpirun -np <num_processes> ./main <input_directory> <output_directory> 
```

Using Slurm
```bash
srun --ntasks=<ntasks>  --cpus-per-task=<cpus-per-task>   --gres=gpu:<num_gpus> --cpu-bind=cores --mpi=pmix --partition=<slurm partition>  ./main <input_directory> <output_directory>
```

**Note**: In our experiment, we observed optimal performance when using mpirun with 4 processes per GPU.
**Note**: While Slurm commands are provided for reference, all experiments in this study were conducted using mpirun.

- #### Multi-node Execution
To run the tool on multiple nodes with GPU resources:

```bash
mpirun -np <total_num_processes> -hostfile <hostfile> ./main <input_directory> <output_directory> 
```

```bash
srun --ntasks-per-node=<tasks_per_node> --nodes=<num_nodes> —cpus-per-task=<cpus-per-task> --gres=gpu:<num_gpus> --mpi=pmix --partition=<partition_name> ./main <input_directory> <output_directory>
```

**Note**: Our experiment was tested up to 4 nodes. In a multi-node environment, the performance was optimal when the total number of processes was kept under 16. For example, with 4 nodes, we used 4 processes per node.


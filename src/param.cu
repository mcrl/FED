#include <iostream>
#include <cstdlib>  // getenv
#include <sys/sysinfo.h>
#include "mpi.h"

#include "param.h"

using namespace std;

#define MAX_BUCKET 100000 // MAX_BUCKET * MAX_BUCKET < GPU memory
#define NUM_KEY 10240 //MAX_BUCKET * NUM_KEY >> total line (=MAX_LINE * file_num)


static int num_key=NUM_KEY;;
static int max_bucket=MAX_BUCKET;
static int c=C;
static int file_offload=true;

void set_param(const int num_file, int num_hash) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    //get node info
    const char* local_size_str = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
    if (!local_size_str) return;

    int local_size = std::stoi(local_size_str);
    
    //get gpu info
    size_t free_mem = 0;
    size_t total_mem = 0;

    cudaError_t cuda_status = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_status != cudaSuccess) {return;}

    // Get the number of available GPU devices
    int deviceCount = 0;
    cuda_status = cudaGetDeviceCount(&deviceCount);
    if (cuda_status != cudaSuccess) {return;}

     // Compute the maximum bucket size based on available GPU memory
    max_bucket=sqrt(free_mem*0.9/ ((local_size-1)/deviceCount+1));

    // Check CPU memory and decide to use file offload
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        perror("sysinfo");
        printf("sysinfo error\n");
    }

    long long tmp_index = (long long)num_file*MAX_LINE*num_hash;

    long cpu_free_mem = info.freeram * info.mem_unit;
    long cpu_required = sizeof(unsigned int) * num_file*MAX_LINE*num_hash*local_size;
    
    // if(!rank) {
    //     printf("CPU Free memory: %.2f GB\n", cpu_free_mem / 1024.0 / 1024.0 / 1024.0);
    //     printf("  minimum required memory: %.2f GB\n", cpu_required / 1024.0 / 1024.0 / 1024.0);
    // }

    // If the available CPU free memory is sufficient and the index of the hash result is smaller than INT_MAX, proceed without offloading.
    if(cpu_free_mem > 10 * cpu_required && tmp_index < INT_MAX ) {
        file_offload=false;
        if(!rank) printf("  File offload: False\n");
    } else {
        file_offload=true;
        if(!rank) printf("  File offload: True\n");
    }
    
    MPI_Bcast(&file_offload, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Calculate the number of keys based on file size and bucket size
    num_key = max(2*num_file*MAX_LINE / max_bucket, int(std::pow(num_file*MAX_LINE, 1.0/2)*2));

    // The number of keys to process simultaneously. It is adjusted based on CPU memory, with a default setting of 1024.
    c=C;
    
    if (rank == 0) {
        std::cout  << "\n( The number of processes per node : " << local_size << ", The number of gpus per node: " << deviceCount << ")\n";
        std::cout  << "( max_bucket : " << max_bucket <<  ", num_key : " << num_key << ")\n" << std::endl;
    }
}

void get_param(int &_num_key, int &_max_bucket, int &_c, int &_file_offload) {
    _num_key=num_key;
    _max_bucket=max_bucket;
    _c=c;
    _file_offload=file_offload;
}
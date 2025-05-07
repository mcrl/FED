// Adjust the parameters in the following section
#define NUM_HASH 128  // The number of hash functions
#define MAX_LINE 30000 //Max line per line 
#define BUCKET 16 // The number of bands
#define SHINGLE_LEN 5 // n-gram size

///////////////////////////////////////////////
#define MAX_BUCKET 100000 // MAX_BUCKET * MAX_BUCKET < GPU memory
#define NUM_KEY 10240 //MAX_BUCKET * NUM_KEY >> total line (=MAX_LINE * file_num)
#define C 2048  // Total file size / NUM_KEY * C * (num_process per node) < CPU memory
void set_param(const int num_file, int num_hash);
void get_param(int &_num_key, int &_max_bucket, int &_c, int &_file_offload);  
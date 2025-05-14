#include <cstring>

double get_total_file_read_time_lsh();
double get_total_file_write_time_lsh();
double get_total_computation_time_lsh();
double g2c();
double c2g();
void print_cmp_time_lsh();

void init_lsh_cuda(int _num_hash, int _len_shingle, int _b, int seed, double _th, int num_file);

void lsh_cuda(const std::string& filepath, const std::string& outputPath, int &_num_line, int idx, int num_file);
void AllgatherFileSize(int size, int files_per_process, int *file_size);
void AllgatherHashResult(int rank, int size, int files_per_process, int start_index); 
void compare_lsh_cuda(const vector<string> &file_list, const std::string& outputPath, int num_file, int *file_size, int rank, int size);

void merge_union(int rank, int size);
void generate_file_init(const std::string& outputPath);
void delete_hash_result(const std::string& outputPath);

void generate_file(const std::string& filepath, int file_idx, const std::string& outputPath);

void finalize_lsh();

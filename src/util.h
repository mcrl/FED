#include <cstring>
#include <vector>

#define MAX_LINE 30000
using namespace std;

bool getFileList(vector<string> &file_list, const std::string& folderPath);
void readFile(const std::string& filepath, int &num_line, int * &_bias, char* &_buf);
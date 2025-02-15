#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstring>
// #include "nlohmann/json.hpp"
// using json = nlohmann::json;

using namespace std;
namespace fs = std::filesystem;

// Function to calculate the size of a file
int filesize(const std::string& filename)
{
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return (int)in.tellg(); 
}

// // Parse the 'text' field from a JSONL line
// // Parsing process can be skipped when working with datasets containing only the 'text' field. 
// // We recommend using data with only the 'text' field during the deduplication phase.
// std::string parseTextFromJsonl(const std::string& line) {
//     auto j=json::parse(line);
//     return j["text"];
// }

// // Iterate through lines in a JSONL file and process the 'text' field
// void iterateLines(const std::string& filepath) {
//     std::ifstream file(filepath);
//     if (!file.is_open()) {
//         std::cerr << "Cannot open file: " << filepath << std::endl;
//         return;
//     }
//     std::string line;
//     while (std::getline(file, line)) {
//         std::string text = parseTextFromJsonl(line);
//     }
//     file.close();
// }

// Retrieve a list of all files in the given folder
bool getFileList(vector<string> &file_list, const std::string& folderPath) {
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "Cannot find folder: " << folderPath << std::endl;
        return false;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        file_list.push_back(entry.path());
    }
    
    return true;
    // for(auto &e : file_list) {
    //     iterateLines(e);
    // }
}

// Define maximum number of lines to handle
#define MAX_LINE 30000

// Read the content of a JSONL file into a buffer and store line offsets in an array
void readFile(const std::string& filepath, int &num_line, int * &_bias, char* &_buf) {
    int total_size=filesize(filepath); // Get the total size of the file

    // Allocate memory for the buffer and bias array
    char *buf=(char*)malloc(sizeof(char)*total_size);
    int *bias=(int*)malloc(sizeof(int)*MAX_LINE);
    int idx,sum;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filepath << std::endl;
        return;
    }

    string line;
    idx=0, sum=0, bias[0]=0;
    while (std::getline(file, line)) {
        strcpy(buf+sum, line.c_str());
        sum+=line.size();
        idx++;
        bias[idx]=sum;
    }
        
    num_line = idx; // Total number of lines read
    _bias = bias;   // Assign the bias array to the output pointer
    _buf = buf;     // Assign the buffer to the output pointer
    file.close();

}
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include "httplib.h"

int find_nth_occurence(const std::string& str, char ch, int n);
void save_url_as_file(const std::string& url, const std::string& file_name);
std::ifstream open_url_cached(const std::string url);
#include "util.h"

int find_nth_occurence(const std::string& str, char ch, int n) {
    int count = 0;
    auto it = std::find_if(str.begin(), str.end(), [&](char c) {
        if (c == ch) count++;
        return count == n;
    });
    return (it != str.end()) ? std::distance(str.begin(), it) : -1;
}

void save_url_as_file(const std::string& url, const std::string& file_name) {
    int splitter = find_nth_occurence(url, '/', 3);
    auto resource_domain = url.substr(0, splitter);
    auto resource_path = url.substr(splitter);
    // fetch resource from url since it did not exist
    httplib::Client cli(resource_domain);
    auto res = cli.Get(resource_path);
    if (!res) {
        std::cerr << "Failed fetching \"" << resource_path << "\" from " << resource_domain << std::endl;
        throw std::runtime_error("Error: Could not fetch from url.");
    }

    std::ofstream out_file(file_name, std::ios::binary);
    if (!out_file.is_open())
        throw std::runtime_error("Error: could not open outfile.");

    out_file.write(res->body.c_str(), res->body.size());
    out_file.close();
    std::cout << "Successfully saved \"" << url << "\" to \"" << file_name << "\"" << std::endl;
}

std::ifstream open_url_cached(const std::string url) {
    std::string file_name = url.substr(url.find_last_of('/') + 1);
    std::ifstream file_stream(file_name);
    if (file_stream.is_open()) {
        std::cout << "Successfully opened file \"" << file_name << "\"" << std::endl;
    } else {
        // attempt to fetch resource from url if we could not open it
        save_url_as_file(url, file_name);
        file_stream.open(file_name);
        if (!file_stream.is_open())
            throw std::runtime_error("Error: Could not open fetched resource.");
    }
    return file_stream;
}
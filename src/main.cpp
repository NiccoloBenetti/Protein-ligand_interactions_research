#include <cstdlib>
#include <iostream>
#include <cstdio> 

// input(const char*, char**) : reads the file specified by filename and saves its content into fileContent
void input(const char *filename, char **fileContent) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        std::cout << "Can't open the file " << filename << std::endl;
        return;
    }

    // Gets the size of the file:
    fseek(file, 0, SEEK_END); 
    long fileSize = ftell(file); 
    fseek(file, 0, SEEK_SET); 

    *fileContent = (char *)malloc(fileSize + 1); 
    if (*fileContent == NULL) {
        std::cout << "Malloc error" << std::endl;
        fclose(file);
        return;
    }

    fread(*fileContent, 1, fileSize, file); 
    (*fileContent)[fileSize] = '\0'; 

    fclose(file); 
}

int main(/*[[maybe_unused]] int argc, [[maybe_unused]] char *argv[]*/) {
    const char *filename = "path_to_file.txt"; // Path to the file you want to read
    char *fileContent = NULL;

    input(filename, &fileContent);

    return EXIT_SUCCESS;
}

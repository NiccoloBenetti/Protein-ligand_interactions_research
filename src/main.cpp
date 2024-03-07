#include <cstdlib>
#include <iostream>
#include <cstdio> 

// input(const char*, char**) : reads the file specified by filename and saves its content into fileContent
void input(char *filename, char **fileContent) {
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

int main(int argc, char *argv[]) {  // First argument: PDB file, then a non fixed number of Mol2 files
    //const char *filename = "path_to_file.txt"; // Path to the file you want to read
    char *fileContent = NULL;

    // Prints the files passed from line (argc, argv)
    if(argc >= 2){
        printf("Ci sono %d file passati:\n", argc - 1);
        std::cout << "1-" << "Protein: " << argv[1] << std::endl;
        for(int i = 2; i < argc; i++) {
            std::cout << i << "-Ligand: " << argv[i] << std::endl;
        }
    }

    //input(argv[1], &fileContent);

    return EXIT_SUCCESS;
}

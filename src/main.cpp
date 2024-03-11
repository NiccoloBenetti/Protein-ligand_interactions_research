#include <cstdlib>
#include <iostream>
#include <cstdio> 
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/Descriptors/MolDescriptors.h>

// input(char**, int, RDKit::RWMol**) : takes the command line arguments (files names and number or arguments) 
// and does the parsing for each file saving a pointer to a RWMol in the last parameter (an array of RWMol passed by ref) 
void input(char **argv, int argc, RDKit::RWMol **molArray) {
    FILE *file;
    char *fileContent = NULL;
    
    *molArray = (RDKit::RWMol*)malloc(sizeof(RDKit::RWMol) * (argc - 1));

    for(int i = 1; i < argc; i++){
        file = fopen(argv[i], "rb");
        if (!file) {
            std::cout << "Can't open the file " << argv[i] << std::endl;
        }
        else{
            // Gets the size of the file:
            fseek(file, 0, SEEK_END); 
            long fileSize = ftell(file); 
            fseek(file, 0, SEEK_SET); 

            fileContent = (char *)malloc(fileSize + 1); 
            if (fileContent == NULL) {
                std::cout << "Malloc error" << std::endl;
                fclose(file);
                return;
            }

            fread(fileContent, 1, fileSize, file); 
            (fileContent)[fileSize] = '\0'; 

            fclose(file);

            RDKit::RWMol* mol;

            if(i == 1){  // if file is a PDB
                //molArray[i-1] = RDKit::PDBBlockToMol(fileContent, true, false);  ||TODO: understand why this approach doesn not work
                mol = RDKit::PDBBlockToMol(fileContent, true, false);
                *(molArray) = mol;
            }
            else{
                //molArray[i-1] = RDKit::Mol2BlockToMol(fileContent, true, false);
                mol = RDKit::Mol2BlockToMol(fileContent, true, false);
                *(molArray + sizeof(RDKit::RWMol) * i) = mol;
            }

            


            free(fileContent);
        }
    }
}

int main(int argc, char *argv[]) {  // First argument: PDB file, then a non fixed number of Mol2 files

    RDKit::RWMol* molArray; // Array where the RWMol pointers will be saved after calling input(...)

    // Prints the files passed from line (argc, argv)
    if(argc >= 2){
        printf("Ci sono %d file passati:\n", argc - 1);
        std::cout << "1-" << "Protein: " << argv[1] << std::endl;
        for(int i = 2; i < argc; i++) {
            std::cout << i << "-Ligand: " << argv[i] << std::endl;
        }
    }

    input(argv, argc, &molArray);

    free(molArray);

    return EXIT_SUCCESS;
}

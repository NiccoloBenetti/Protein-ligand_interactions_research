#include <cstdlib>
#include <iostream>
#include <cstdio> 
#include <string>
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/Descriptors/MolDescriptors.h>


struct SMARTSPattern {
    std::string name;
    int numAtoms;
    std::string smartsString;
};

SMARTSPattern smartsPatterns[] = {
    {"hydrophobic", 1, "[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]"},
    {"hydrogen_donor-H", 2, "[$([O,S;+0]),$([N;v3,v4&+1]),n+0]-[H]"},
    {"hydrogen_acceptor", 1, "[#7&!$([nX3])&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4&+1]),O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O),o+0,F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])]"},
    {"halogen_donor-halogen", 2, "[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]"},
    {"halogen_acceptor-any", 2, "[#7,#8,P,S,Se,Te,a;!+{1-}][*]"},
    {"anion", 1, "[-{1-},$(O=[C,S,P]-[O-])]"},
    {"cation", 1, "[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]"},
    {"aromatic_ring", 5, "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1"},
    {"aromatic_ring", 6, "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1"},
    {"metal", 1, "[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]"},
    {"chelated", 1, "[O,#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4]),-{1-};!+{1-}]"}
};

void printMolOverview(RDKit::ROMol mol, bool smiles) {
    // Numero di atomi
    std::cout << "Numero di atomi: " << mol.getNumAtoms() << std::endl;

    // Numero di legami
    std::cout << "Numero di legami: " << mol.getNumBonds() << std::endl;

    /*
    // Formula molecolare
    std::string formula = RDKit::Descriptors::calcMolFormula(mol);
    std::cout << "Formula molecolare: " << formula << std::endl;

    // Peso molecolare
    double mw = RDKit::Descriptors::calcExactMW(mol);
    std::cout << "Peso molecolare: " << mw << std::endl;
    */

    // Rappresentazione SMILES
    if(smiles){
        std::string smiles = RDKit::MolToSmiles(mol);
        std::cout << "SMILES: " << smiles << std::endl;
    }
}

// input(char**, int, std::vector<RDKit::ROMol> &) : takes the command line arguments (files names and number or arguments) 
// and does the parsing for each file saving a ROMol in the last parameter (a vector of ROMol passed by ref) 
void input(char **argv, int argc, std::vector<RDKit::ROMol> &molVector) {
    FILE *file;
    char *fileContent = NULL;

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

            RDKit::ROMol* mol;

            if(i == 1){  // if file is a PDB
                mol = RDKit::PDBBlockToMol(fileContent, true, false);
                molVector.push_back(*mol);
            }
            else{
                mol = RDKit::Mol2BlockToMol(fileContent, true, false);
                molVector.push_back(*mol);
            }

            printMolOverview(*mol, false);

            free(fileContent);
        }
    }
}

int main(int argc, char *argv[]) {  // First argument: PDB file, then a non fixed number of Mol2 files

    std::vector<RDKit::ROMol> molVector; // Vector of all the molecules (the first element is always a protein, the other are ligands)

    // Prints the files passed from line (argc, argv)
    if(argc >= 2){
        printf("Ci sono %d file passati:\n", argc - 1);
        std::cout << "1-" << "Protein: " << argv[1] << std::endl;
        for(int i = 2; i < argc; i++) {
            std::cout << i << "-Ligand: " << argv[i] << std::endl;
        }
    }

    input(argv, argc, molVector);

    return EXIT_SUCCESS;
}

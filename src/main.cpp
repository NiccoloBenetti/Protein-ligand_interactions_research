#include <cstdlib>
#include <iostream>
#include <cstdio> 
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <GraphMol/GraphMol.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/Substruct/SubstructMatch.h>

enum class Pattern {
    Hydrophobic,
    Hydrogen_donor-H,
    Hydrogen_acceptor,
    Halogen_donor-halogen,
    Halogen_acceptor-any,
    Anion,
    Cation,
    Aromatic_ring5,
    Aromatic_ring6,
    Metal,
    Chelated,
}


struct SMARTSPattern {
    std::string name;
    int numAtoms;
    std::string smartsString;
};
struct FoundPatterns{
    std::map<Pattern, std::vector<RDKit::MatchVectType>> patternMatches; // Maps every pattern with vector of all it's found istances that are rappresented ad pairs <athom in the pattern, athom in the mol>.
};
struct PossibleInteraction{
    std::string name;
    std::vector<std::string> interactonPatterns; 
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

const int smartsPatternsCount = sizeof(smartsPatterns) / sizeof(SMARTSPattern);

PossibleInteraction possibleInteractions[] = {
    {"Hydrophobic interaction", {"hydrophobic", "hydrophobic"}},
    {"Hydrogen bond", {"hydrogen_donor-H", "hydrogen_acceptor"}},
    {"Halogen bond", {"halogen_donor-halogen", "halogen_acceptor-any"}},
    {"Ionic interaction (cation ... anion)", {"cation", "anion"}},
    {"Ionic interaction (cation ... aromatic_ring)", {"cation", "aromatic_ring"}},
    {"Pi stacking", {"aromatic_ring", "aromatic_ring"}},
    {"Metal coordination", {"metal", "chelated"}}
};

const int possibleInteractionsCount = sizeof(possibleInteractions) / sizeof(PossibleInteraction);

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
/*
bool contains(std::vector<std::string> vec, std::string str){
    for(int i = 0; i < vec.size(); i++){
        if(vec.at(i) == str)
            return true;
    }
    return false;
}

bool isInteractionPossible(MatchStruct firstPattern, MatchStruct secondPattern, PossibleInteraction interactions[], int interactionsCount){
    for(int i = 0; i < interactionsCount; i++){
        if(contains(interactions[i].interactonPatterns, firstPattern.pattern.name) && contains(interactions[i].interactonPatterns, secondPattern.pattern.name)){ //TODO : È SBAGLIATA PERCHE PER OGNI ELEMENTO TROVATO DOVREBBE TOGLIERLO DALLA LISTA DA CUI SI CERCA DOPO
            std::cout << firstPattern.pattern.name << " " << secondPattern.pattern.name << std::endl;
            return true;
        }
    }

    return false;
}

void identifyInteractions(std::vector<MatchStruct> proteinPatterns, std::vector<MatchStruct> ligandPatterns){
    for(int i = 0; i < proteinPatterns.size(); i++){         // | This double for compares every element in proteinPatterns with 
        for(int j = 0; j < ligandPatterns.size(); j++){      // | every element in ligandPatterns
            if(isInteractionPossible(proteinPatterns.at(i), ligandPatterns.at(j), possibleInteractions, possibleInteractionsCount)){
                for(int k = 0; k < proteinPatterns.at(i).matches.size(); k++){      // | Found a compatible pair of MatchStruct, one of the protein and one of the ligand, this double for compares every element
                    for(int s = 0; s < ligandPatterns.at(j).matches.size(); s++){   // | of a MatchStruct.matches with every element of the second MatchStruct.matches
                        //retInteraction.function(proteinPatterns.at(i).matches.at(k), ligandPatterns.at(j).matches.at(s));
                    }
                }
            }
        }
    }
}*/
void findHydrophobicInteraction(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){
    //Check that there is at list one Hydrophobic pattern found on both protein and ligand
    if ((proteinPatterns.patternMatches.find(Pattern::Hydrophobic) != proteinPatterns.patternMatches.end()) && (ligandPatterns.patternMatches.find(Pattern::Hydrophobic) != ligandPatterns.patternMatches.end())){
        
    }
}
void findHydrogenBond(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findHalogenBond(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findIonicInteraction_Ca_An(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findIonicInteraction_Ca_Ar(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findPiStacking(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findmetalCoordination(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}

void identifyInteractions(const RDKit::ROMol& protein, const RDKit::ROMol& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){
    //every function will need to serch all the interactions of that tipe and for every one found call the output function that adds them to the CSV file
    
    findHydrophobicInteraction(protein, ligand, proteinPatterns, ligandPatterns);
    findHydrogenBond(protein, ligand, proteinPatterns, ligandPatterns);
    findHalogenBond(protein, ligand, proteinPatterns, ligandPatterns);
    findIonicInteraction_Ca_An(protein, ligand, proteinPatterns, ligandPatterns);
    findIonicInteraction_Ca_Ar(protein, ligand, proteinPatterns, ligandPatterns);
    findPiStacking(protein, ligand, proteinPatterns, ligandPatterns);
    findMetalCoordination(protein, ligand, proteinPatterns, ligandPatterns);
    
}

// for eatch pattern of the Pattern enum looks if it is in the mol and saves all the matches in the MatchVectType field of the map inside FoundPatterns.
void identifySubstructs(RDKit::ROMol& mol, SMARTSPattern patterns[], int patternsCount, FoundPatterns &foundPatterns){
    for(int i = 0; i < patternsCount; i++){
        std::vector<RDKit::MatchVectType> tmpMatchesVector;
        RDKit::ROMol* patternMol = RDKit::SmartsToMol(patterns[i].smartsString);
        bool foundMatch = RDKit::SubstructMatch(mol, *patternMol, tmpMatchesVector);

        if(foundMatch && !tmpMatchesVector.empty()){
            //the number of patterns and their index must be the same inside the Pattern Enum and smartsPatterns
            foundPatterns.patternMatches[static_cast<Pattern>(i)] = tmpMatchesVector;
        }
        delete patternMol;
    }
}

void printFoundPatterns(std::vector<MatchStruct> foundPatterns){
    std::cout << "Found patterns [" << foundPatterns.size() << "]: "<< std::endl;

    for(int i = 0; i < foundPatterns.size(); i++){
        std::cout << " ------ " << foundPatterns.at(i).pattern.name << " ------ " << std::endl;

        for(int j = 0; j < foundPatterns.at(i).matches.size(); j++){
            std::cout << "    " << j+1 << std::endl;

            for(int k = 0; k < foundPatterns.at(i).matches.at(j).size(); k++){
                std::cout << "        " << "First A: " << foundPatterns.at(i).matches.at(j).at(k).first << " Second A: " << foundPatterns.at(i).matches.at(j).at(k).second << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << std::endl;
    }
}
//takes input all the values as parameters and prints on the CSV file passed by reference NB.might be necessary to escape the strings if there can be "," in them
void output(std::string ligandName, std::string proteinAtomId, std::string proteinPatterns, float proteinX, float proteinY, float proteinZ, std::string ligandAtomId, std::string ligandPattern, float ligandX, float ligandY, float ligandZ, std::string interactionType, float interactionDistance, std::ofstream &outputFile){
    if (outputFile.is_open()){
        outputFile << ligandName << ","
                   << proteinAtomId << ","
                   << proteinPatterns << ","
                   << proteinX << ","
                   << proteinY << ","
                   << proteinZ << ","
                   << ligandAtomId << ","
                   << ligandPattern << ","
                   << ligandX << ","
                   << ligandY << ","
                   << ligandZ << ","
                   << interactionType << ","
                   << interactionDistance << "\n";
    }
    else {
        std::cerr << "File was not open correctly for writing." << std::endl;
    }
}

int main(int argc, char *argv[]) {  // First argument: PDB file, then a non fixed number of Mol2 files

    std::vector<RDKit::ROMol> molVector; // Vector of all the molecules (the first element is always a protein, the other are ligands)

    FoundPatterns proteinPatterns;  //Declares a FoundPattern struct where to save all the pattern found in the protein
    FoundPatterns ligandPatterns;   //Declares a FoundPattern struct where to save all the pattern found in the ligand, the same will be used for all ligand passed in input.

    //the CSV file is created and inicialized with the HEADER line in the main
    std::ofstream outputFile("interactions.csv",std::ios::out);
    if (outputFile.is_open()){
        outputFile << "LIGAND_NAME,PROTEIN_ATOM_ID,PROTEIN_PATTERN,PROTEIN_X,PROTEIN_Y,PROTEIN_Z,LIGAND_ATOM_ID,LIGAND_PATTERN,LIGAND_X,LIGAND_Y,LIGAND_Z,INTERACTION_TYPE,INTERACTION_DISTANCE" <<std::endl;
        outputFile.close();
        std::cout << "File interactions.csv succesfuly created." <<std::endl;
    }
    else{
        std::cerr << "Error while creating CSV file." << std::endl;
    }

    /*To print on CSV file with output function use:
    outputFile.open("interactions.csv", std::ios::app);
    output(ligandName, proteinAtomId, proteinPatterns, proteinX, proteinY, proteinZ,
       ligandAtomId, ligandPattern, ligandX, ligandY, ligandZ,
       interactionType, interactionDistance, outputFile);
    outputFile.close();
    */

    // Prints the files passed from line (argc, argv)
    if(argc >= 2){
        printf("Ci sono %d file passati:\n", argc - 1);
        std::cout << "1-" << "Protein: " << argv[1] << std::endl;
        for(int i = 2; i < argc; i++) {
            std::cout << i << "-Ligand: " << argv[i] << std::endl;
        }
    }

    input(argv, argc, molVector);

    identifySubstructs(molVector.at(0), smartsPatterns, smartsPatternsCount, proteinPatterns); // Identifies all the itances of patterns inside the protein
    //printFoundPatterns(proteinPatterns);

    for(int i = 1; i < argc - 1; i++){ // For every ligand
        identifySubstructs(molVector.at(i), smartsPatterns, smartsPatternsCount, ligandPatterns); // Identifies all the itances of patterns inside the ligand
        identifyInteractions(proteinPatterns, ligandPatterns); //Identifies all the interactions between protein and ligand and adds the to the CSV file
        ligandPatterns.clear();
    } 

    return EXIT_SUCCESS;
}

#include <cstdlib>
#include <iostream>
#include <cstdio> 
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <memory>
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
    Hydrogen_donor_H,
    Hydrogen_acceptor,
    Halogen_donor_halogen,
    Halogen_acceptor_any,
    Anion,
    Cation,
    Aromatic_ring5,
    Aromatic_ring6,
    Metal,
    Chelated,
};

std::string PatternToString(Pattern pattern) {
    switch(pattern) {
        case Pattern::Hydrophobic: return "Hydrophobic"; 
        case Pattern::Hydrogen_donor_H: return "Hydrogen_donor_H"; 
        case Pattern::Hydrogen_acceptor: return "Hydrogen_acceptor"; 
        case Pattern::Halogen_donor_halogen: return "Halogen_donor_halogen"; 
        case Pattern::Halogen_acceptor_any: return "Halogen_acceptor_any"; 
        case Pattern::Anion: return "Anion"; 
        case Pattern::Cation: return "Cation"; 
        case Pattern::Aromatic_ring5: return "Aromatic_ring5"; 
        case Pattern::Aromatic_ring6: return "Aromatic_ring6";
        case Pattern::Metal: return "Metal"; 
        case Pattern::Chelated: return "Chelated";
        default:    return "Unknown";
    }
}

struct SMARTSPattern {
    Pattern pattern;
    int numAtoms;
    std::string smartsString;
};
struct FoundPatterns {
    std::map<Pattern, std::vector<RDKit::MatchVectType>> patternMatches; // Maps every pattern with vector of all it's found istances that are rappresented ad pairs <athom in the pattern, athom in the mol>.
};

struct Molecule {   //This struct is used to save ich mol with it's name
    std::string name;
    std::unique_ptr<RDKit::ROMol> mol;

    Molecule(const std::string& molName, RDKit::ROMol* molPtr)  // Constructor that populates the name and mol attributes (it neads a pointer to a ROMol object)
        : name(molName), mol(molPtr) {}

    // Disables copy to ensure the ROMol object can not be accidentaly copied
    Molecule(const Molecule&) = delete;
    Molecule& operator=(const Molecule&) = delete;

    // The following is optional but is better tu put it to avoid problems with the compiler, it enables the possibility to move controll of the object to others
    Molecule(Molecule&&) noexcept = default;
    Molecule& operator=(Molecule&&) noexcept = default;
};

SMARTSPattern smartsPatterns[] = {
    {Pattern::Hydrophobic , 1, "[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]"},
    {Pattern::Hydrogen_donor_H, 2, "[$([O,S;+0]),$([N;v3,v4&+1]),n+0]-[H]"},
    {Pattern::Hydrogen_acceptor, 1, "[#7&!$([nX3])&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4&+1]),O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O),o+0,F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])]"},
    {Pattern::Halogen_donor_halogen, 2, "[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]"},
    {Pattern::Halogen_acceptor_any, 2, "[#7,#8,P,S,Se,Te,a;!+{1-}][*]"},
    {Pattern::Anion, 1, "[-{1-},$(O=[C,S,P]-[O-])]"},
    {Pattern::Cation, 1, "[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]"},
    {Pattern::Aromatic_ring5, 5, "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1"},
    {Pattern::Aromatic_ring6, 6, "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1"},
    {Pattern::Metal, 1, "[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]"},
    {Pattern::Chelated, 1, "[O,#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4]),-{1-};!+{1-}]"}
};

const int smartsPatternsCount = sizeof(smartsPatterns) / sizeof(SMARTSPattern);

// PossibleInteraction possibleInteractions[] = {
//     {"Hydrophobic interaction", {"hydrophobic", "hydrophobic"}},
//     {"Hydrogen bond", {"hydrogen_donor-H", "hydrogen_acceptor"}},
//     {"Halogen bond", {"halogen_donor-halogen", "halogen_acceptor-any"}},
//     {"Ionic interaction (cation ... anion)", {"cation", "anion"}},
//     {"Ionic interaction (cation ... aromatic_ring)", {"cation", "aromatic_ring"}},
//     {"Pi stacking", {"aromatic_ring", "aromatic_ring"}},
//     {"Metal coordination", {"metal", "chelated"}}
// };

//const int possibleInteractionsCount = sizeof(possibleInteractions) / sizeof(PossibleInteraction);

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

// The name of the files containing the molecules has a .pdb or .mol2 extension at the end that isn't needed nor wonted so this function get's rid of it 
std::string removeFileExtension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

// input(char**, int, std::vector<Molecule> &) : takes the command line arguments (files names and number or arguments) 
// and does the parsing for each file saving a ROMol and the name of that molecule in the last parameter (a vector of struct Molecule passed by ref) 
void input(char **argv, int argc, std::vector<Molecule> &molVector) {
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

            std::unique_ptr<RDKit::ROMol> mol;

            if(i == 1){  // if file is a .pdb
                mol.reset(RDKit::PDBBlockToMol(fileContent, true, false));
            }
            else{   //if file is a .mol2
                mol.reset(RDKit::Mol2BlockToMol(fileContent, true, false));
            }

            if(mol) {
                molVector.emplace_back(removeFileExtension(argv[i]), mol.release());
            }

            printMolOverview(*(molVector.back().mol), false);

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

float calculateDistance(RDGeom::Point3D &p_a, RDGeom::Point3D &p_b){  //calculates euclidian distance between 2 points located in a 3D space
    return (p_a - p_b).length();
}
//Having three points located in a 3D space, imagine them forming a triangle: this function calculates the angle on the vertex p_a 
float calculateAngle(RDGeom::Point3D &p_a, RDGeom::Point3D &p_b, RDGeom::Point3D &p_c){
    float ab = calculateDistance(p_a, p_b);
    float bc = calculateDistance(p_b, p_c);
    float ac = calculateDistance(p_a, p_c);

    return sin((pow(ab, 2) + pow(ac, 2) - pow(bc, 2)) / (2*ab*ac));
}

// Hydrophobic and Metal interactions both need only a distance check to be evaluated, this function handles the task for both findHydrophobicInteractions() and findMetalInteractions() and handles also the output on CSV
void basicBondHandler(const Molecule &protein, const Molecule &ligand, std::map<Pattern, std::vector<RDKit::MatchVectType>>::const_iterator& tmpProt, std::map<Pattern, std::vector<RDKit::MatchVectType>>::const_iterator& tmpLig, float distRequired, std::string protPat, std::string ligPat, std::string intType){
    RDGeom::Point3D& posProt;    //are needed to easly manage x,y,z cordinates that will be feeded to the output funcion
    RDGeom::Point3D& posLig;
    int protAIndx;  //will contain the atom index for the protein in order to calculate distances
    int ligAIndx;   //same for the ligand
    float distance;

    for (const auto& matchVectProt : tmpProt->second){  //for every block of the vector containing Hydrophobic matcher in proteinPatterns.patterMatches
        for(const auto& matchVectLig : tmpLig->second){ //for every block of the vector containing Hydrophobic matcher in ligandPatterns.patternMatches
            for(const auto& pairsProt : matchVectProt){ //for every pair <atom in the pattern, atom in the prot>
                
                protAIndx = pairsProt.second;
                
                for(const auto& pairsLig : matchVectLig){   //for every pair <atom in the pattern, atom in the mol>
                    ligAIndx = pairsLig.second;
                    distance = calculateDistance(protein, protAIndx, ligand, ligAIndx, posProt, posLig);
        
                    if (distance <= distRequired){
                        output(ligand.name, /*Protein Atom ID*/, protPat, posProt.x, posProt.y, posProt.z, /*Ligand Atom ID*/, ligPat, posLig.x, posLig.y, posLig.z, intType, distance);   //call output funcion
                    }
                }
            }
        }
    }
}

void findHydrophobicInteraction(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){
    auto tmpProt = proteinPatterns.patternMatches.find(Pattern::Hydrophobic);
    auto tmpLig = ligandPatterns.patternMatches.find(Pattern::Hydrophobic);
    float distRequired = 4.5;

    //Check that there is at list one Hydrophobic pattern found on both protein and ligand
    if ((tmpProt != proteinPatterns.patternMatches.end()) && (tmpLig != ligandPatterns.patternMatches.end())){
        basicBondHandler(protein, ligand, tmpProt, tmpLig, distRequired, "Hydrophobic", "Hydrophobic", "Hydrophobic");
    }
}

void findHydrogenBond(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){
    auto tmpProt = proteinPatterns.patternMatches.find(Pattern::Hydrogen_donor_H);
    auto tmpLig = ligandPatterns.patternMatches.find(Pattern::Hydrogen_acceptor);
    float distance;
    float distRequired = 3.5;
    float minAngleRequired = 130;
    float maxAngleRequired = 180;

    if ((tmpProt != proteinPatterns.patternMatches.end()) && (tmpLig != ligandPatterns.patternMatches.end())){
        const RDKit::Conformer& confProt = protein.mol.getConformer();  //TODO: move it somewhere else cause they are used by all the interaction's functions
        const RDKit::Conformer& confLig = ligand.mol.getConformer();    

        RDGeom::Point3D& pos_donor, pos_hydrogen, pos_acceptor;

        for(const auto& matchVectProt : tmpProt->second){
            int donorId = matchVectProt.at(0).second;
            int hydrogenId = matchVectProt.at(1).second;

            pos_donor = confProt.getAtomPos(donorId);
            pos_hydrogen = confProt.getAtomPos(hydrogenId);

            for(const auto& matchVectLig : tmpLig->second){
                int acceptorId = matchVectLig.at(0).second;

                pos_acceptor = confLig.getAtomPos(acceptorId);

                if(calculateDistance(pos_donor, pos_acceptor) <= distRequired && (calculateAngle(pos_hydrogen, pos_donor, pos_acceptor) <= maxAngleRequired || calculateAngle(pos_hydrogen, pos_donor, pos_acceptor) <= minAngleRequired)){
                    output(ligand.name, /*Protein Atom ID*/, "Hydrogen donor", pos_donor.x, pos_donor.y, pos_donor.z, /*Ligand Atom ID*/, "Hydrogen acceptor", pos_acceptor.x, pos_acceptor.y, pos_acceptor.z, "Hydrogen Bond", distance);   //call output funcion
                }
        
            }
        }
    }
}

void findHalogenBond(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findIonicInteraction_Ca_An(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findIonicInteraction_Ca_Ar(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}
void findPiStacking(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){}

void findMetalCoordination(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){
    auto tmpProt = proteinPatterns.patternMatches.find(Pattern::Metal);
    auto tmpLig = ligandPatterns.patternMatches.find(Pattern::Chelated);
    float distRequired = 2.8;

    //Check that there is at list one Metal pattern found on the protein and one Chelated pattern on the ligand
    if ((tmpProt != proteinPatterns.patternMatches.end()) && (tmpLig != ligandPatterns.patternMatches.end())){
        basicBondHandler(protein, ligand, tmpProt, tmpLig, distRequired, "Metal", "Chelated", "Metal");
    }

    tmpProt = proteinPatterns.patternMatches.find(Pattern::Chelated);
    tmpLig = ligandPatterns.patternMatches.find(Pattern::Metal);

    //Check that there is at list one Chelated pattern found on the protein and one Metal pattern on the ligand
    if ((tmpProt != proteinPatterns.patternMatches.end()) && (tmpLig != ligandPatterns.patternMatches.end())){
        basicBondHandler(protein, ligand, tmpProt, tmpLig, distRequired, "Chelated", "Metal", "Metal");
    }

}

void identifyInteractions(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns){
    //every function will need to serch all the interactions of that type and for every one found call the output function that adds them to the CSV file
    
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
        //TODO: maybe its a good idea to also clean the tmpMatchesVector
    }
}

void printFoundPatterns(FoundPatterns foundPatterns){
    std::cout << "Found patterns [" << foundPatterns.patternMatches.size() << "]: "<< std::endl;

    for(const auto& patternMatch: foundPatterns.patternMatches){
        std::cout << " ------ " << PatternToString(patternMatch.first) << " ------ " << std::endl;

        for(int j = 0; j < patternMatch.second.size(); j++){
            std::cout << "    " << j+1 << std::endl;

            for(int k = 0; k < patternMatch.second.at(j).size(); k++){
                std::cout << "        " << "First A: " << patternMatch.second.at(j).at(k).first << " Second A: " << patternMatch.second.at(j).at(k).second << std::endl;
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

    std::vector<Molecule> molVector; // Vector of all the molecules with their name, (the first element is always a protein, the other are ligands)

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

    identifySubstructs(molVector.at(0).mol.get(), smartsPatterns, smartsPatternsCount, proteinPatterns); // Identifies all the itances of patterns inside the protein
    //printFoundPatterns(proteinPatterns);

    for(int i = 1; i < argc - 1; i++){ // For every ligand
        identifySubstructs(molVector.at(i).mol.get(), smartsPatterns, smartsPatternsCount, ligandPatterns); // Identifies all the itances of patterns inside the ligand
        identifyInteractions(molVector.at(0), molVector.at(i), proteinPatterns, ligandPatterns); //Identifies all the interactions between protein and ligand and adds the to the CSV file
        ligandPatterns.clear();
    } 

    return EXIT_SUCCESS;
}

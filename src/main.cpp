#include <cstdlib>
#include <iostream>
#include <cstdio> 
#include <string>
#include <fstream>
#include <map>
#include <vector>
#include <memory>
#include <cmath>
#include <GraphMol/Conformer.h>
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

bool isAngleInRange(float angle, float minAngle, float maxAngle){
    return (angle >= minAngle && angle <= maxAngle) ? true : false;
}

void findHydrophobicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const bool protA_ligB, std::ofstream &outputFile){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Hydrophobic);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Hydrophobic);

    //Check that there is at list one Hydrophobic pattern found on both protein and ligand if yes serches and prints the bonds
    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        unsigned int indx_molA;  //will contain the atom index for the protein in order to calculate distances
        unsigned int indx_molB;   //same for the ligand
        const RDKit::Conformer& confA = molA.mol.getConformer();  //Conformer is a class that represents the 2D or 3D conformation of a molecule
        const RDKit::Conformer& confB = molB.mol.getConformer();    //we get an istance of the 3D conformation of both protein and ligand
        RDGeom::Point3D& p_a;    //are needed to easly manage x,y,z cordinates that will be feeded to the output funcion
        RDGeom::Point3D& p_b;
        float distRequired = 4.5;
        float distance;

        for (const auto& matchVectA : tmpA->second){  //for every block of the vector containing Hydrophobic matcher in proteinPatterns.patterMatches
                indx_molA = matchVectA.second;
                p_a = confA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){ //for every block of the vector containing Hydrophobic matcher in ligandPatterns.patternMatches
                indx_molB = matchVectB.second;
                p_b = confB.getAtomPos(indx_molB);
                distance = calculateDistance(p_a, p_b);

                if (distance <= distRequired){
                    output(molA.name, molB.name, /*Protein Atom ID*/, "Hydrophobic", posProt.x, posProt.y, posProt.z, /*Ligand Atom ID*/, "Hydrophobic", posLig.x, posLig.y, posLig.z, "Hydrophobic", distance, protA_ligB, outputFile);   //call output funcion
                }
            }
        }
    }
}

void findHydrogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns){
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Hydrogen_donor_H);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Hydrogen_acceptor);
    float distance;
    float distance_required = 3.5;
    float minAngle_required = 130;
    float maxAngle_required = 180;

    if ((molA_pattern != molA_patterns.patternMatches.end()) && (molB_pattern != molB_patterns.patternMatches.end())){
        const RDKit::Conformer& conformer_molA = molA.mol.getConformer();  //TODO: move it somewhere else cause they are used by all the interaction's functions
        const RDKit::Conformer& conformer_molB = molB.mol.getConformer();    

        RDGeom::Point3D& pos_donor, pos_hydrogen, pos_acceptor;

        for(const auto& matchVect_molA : molA_pattern->second){
            int id_donor = matchVect_molA.at(0).second;
            int id_hydrogen = matchVect_molA.at(1).second;

            pos_donor = conformer_molA.getAtomPos(id_donor);
            pos_hydrogen = conformer_molA.getAtomPos(id_hydrogen);

            for(const auto& matchVect_molB : molB_pattern->second){
                int id_acceptor = matchVect_molB.at(0).second;

                pos_acceptor = conformer_molB.getAtomPos(id_acceptor);

                distance = calculateDistance(pos_donor, pos_acceptor);
                float angle = calculateAngle(pos_hydrogen, pos_donor, pos_acceptor);

                if(distance <= distance_required && isAngleInRange(angle, minAngle_required, maxAngle_required)){
                    output(molB.name, /*Protein Atom ID*/, "Hydrogen donor", pos_donor.x, pos_donor.y, pos_donor.z, /*Ligand Atom ID*/, "Hydrogen acceptor", pos_acceptor.x, pos_acceptor.y, pos_acceptor.z, "Hydrogen Bond", distance);   //call output funcion
                }
        
            }
        }
    }
}

void findHalogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const bool protA_ligB, std::ofstream &outputFile){
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Halogen_donor_halogen);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Halogen_acceptor_any);
    float distance;
    float distance_required = 3.5;
    float minAngle_required_first = 130;
    float maxAngle_required_first = 180;
    float minAngle_required_second = 80;
    float maxAngle_required_second = 140;

    if ((molA_pattern != molA_patterns.patternMatches.end()) && (molB_pattern != molB_patterns.patternMatches.end())){
        const RDKit::Conformer& conformer_molA = molA.mol.getConformer();  //TODO: move it somewhere else cause they are used by all the interaction's functions
        const RDKit::Conformer& conformer_molB = molB.mol.getConformer();    

        RDGeom::Point3D& pos_donor, pos_halogen, pos_acceptor, pos_any;

        for(const auto& matchVect_molA : molA_pattern->second){
            int id_donor = matchVect_molA.at(0).second;
            int id_halogen = matchVect_molA.at(1).second;

            pos_donor = conformer_molA.getAtomPos(id_donor);
            pos_halogen = conformer_molA.getAtomPos(id_halogen);

            for(const auto& matchVect_molB : molB_pattern->second){
                int id_acceptor = matchVect_molB.at(0).second;
                int id_any = matchVect_molB.at(1).second;

                pos_acceptor = conformer_molB.getAtomPos(id_acceptor);
                pos_any = conformer_molB.getAtomPos(id_any);

                distance = calculateDistance(pos_donor, pos_acceptor);
                float firstAngle = calculateAngle(pos_halogen, pos_donor, pos_acceptor);
                float secondAngle = calculateAngle(pos_acceptor, pos_halogen, pos_any);

                if(distance <= distance_required && isAngleInRange(firstAngle, minAngle_required_first, maxAngle_required_first) && isAngleInRange(secondAngle, minAngle_required_second, maxAngle_required_second)){
                    output(molB.name, /*Protein Atom ID*/, "Halogen donor", pos_donor.x, pos_donor.y, pos_donor.z, /*Ligand Atom ID*/, "Halogen acceptor", pos_acceptor.x, pos_acceptor.y, pos_acceptor.z, "Halogen Bond", distance);   //call output funcion
                }
        
            }
        }
    }
}

void findIonicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const bool protA_ligB, std::ofstream &outputFile){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Cation);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Anion);
    unsigned int indx_molA;  //will contain the atom index for the protein in order to calculate distances
    unsigned int indx_molB;   //same for the ligand
    const RDKit::Conformer& confA = molA.mol.getConformer();  //Conformer is a class that represents the 2D or 3D conformation of a molecule
    const RDKit::Conformer& confB = molB.mol.getConformer();    //we get an istance of the 3D conformation of both protein and ligand
    RDGeom::Point3D& p_a;    //are needed to easly manage x,y,z cordinates that will be feeded to the output funcion
    RDGeom::Point3D& p_b;
    float distRequired = 4.5;
    float distance;

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        for (const auto& matchVectA : tmpA->second){  //for every block of the vector containing Hydrophobic matcher in proteinPatterns.patterMatches
                indx_molA = matchVectA.second;
                p_a = confA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){ //for every block of the vector containing Hydrophobic matcher in ligandPatterns.patternMatches
                indx_molB = matchVectB.second;
                p_b = confB.getAtomPos(indx_molB);
                distance = calculateDistance(p_a, p_b);

                if (distance <= distRequired){
                    output(molA.name, molB.name, /*Protein Atom ID*/, "Cation", posProt.x, posProt.y, posProt.z, /*Ligand Atom ID*/, "Anion", posLig.x, posLig.y, posLig.z, "Ionic", distance, protA_ligB, outputFile);   //call output funcion
                }
            }
        }
    }

    //NEEDS TO BE COMPLEATED

}
void findPiStacking(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const bool protA_ligB, std::ofstream &outputFile){}

void findMetalCoordination(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const bool protA_ligB, std::ofstream &outputFile){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Metal);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Chelated);

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        unsigned int indx_molA;  //will contain the atom index for the protein in order to calculate distances
        unsigned int indx_molB;   //same for the ligand
        const RDKit::Conformer& confA = molA.mol.getConformer();  //Conformer is a class that represents the 2D or 3D conformation of a molecule
        const RDKit::Conformer& confB = molB.mol.getConformer();    //we get an istance of the 3D conformation of both protein and ligand
        RDGeom::Point3D& p_a;    //are needed to easly manage x,y,z cordinates that will be feeded to the output funcion
        RDGeom::Point3D& p_b;
        float distRequired = 2.8;
        float distance;

        for (const auto& matchVectA : tmpA->second){  //for every block of the vector containing Hydrophobic matcher in proteinPatterns.patterMatches
                indx_molA = matchVectA.second;
                p_a = confA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){ //for every block of the vector containing Hydrophobic matcher in ligandPatterns.patternMatches
                indx_molB = matchVectB.second;
                p_b = confB.getAtomPos(indx_molB);
                distance = calculateDistance(p_a, p_b);

                if (distance <= distRequired){
                    output(molA.name, molB.name, /*Protein Atom ID*/, "Metal", posProt.x, posProt.y, posProt.z, /*Ligand Atom ID*/, "Chelated", posLig.x, posLig.y, posLig.z, "Metal", distance, protA_ligB, outputFile);   //call output funcion
                }
            }
        }
    }

}

void identifyInteractions(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns, std::ofstream &outputFile){
    //every function will need to serch all the interactions of that type and for every one found call the output function that adds them to the CSV file
    
    findHydrophobicInteraction(protein, ligand, proteinPatterns, ligandPatterns, true, outputFile);
    findHydrogenBond(protein, ligand, proteinPatterns, ligandPatterns);
    findHalogenBond(protein, ligand, proteinPatterns, ligandPatterns);
    findIonicInteraction(protein, ligand, proteinPatterns, ligandPatterns, true, outputFile);
    findIonicInteraction(ligand, protein, ligandPatterns, proteinPatterns, false, outputFile);
    findPiStacking(protein, ligand, proteinPatterns, ligandPatterns);
    findMetalCoordination(protein, ligand, proteinPatterns, ligandPatterns, true, outputFile);
    findMetalCoordination(ligand, protein, ligandPatterns, proteinPatterns, false, outputFile);
    
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
void output(std::string name_molA, std::string name_molB, std::string atom_id_molA, std::string pattern_molA, float x_molA, float y_molA, float z_molA, std::string atom_id_molB, std::string pattern_molB, float x_molB, float y_molB, float z_molB, std::string interactionType, float interactionDistance, const bool protA_ligB, std::ofstream &outputFile){
    if (outputFile.is_open()){
        if(protA_ligB){
            outputFile << name_molB << ","
                       << atom_id_molA << ","
                       << pattern_molA << ","
                       << x_molA << ","
                       << y_molA << ","
                       << z_molA << ","
                       << atom_id_molB << ","
                       << pattern_molB << ","
                       << x_molB << ","
                       << y_molB << ","
                       << z_molB << ","
                       << interactionType << ","
                       << interactionDistance << "\n";
        }
        else{
            outputFile << name_molA << ","
                       << atom_id_molB << ","
                       << pattern_molB << ","
                       << x_molB << ","
                       << y_molB << ","
                       << z_molB << ","
                       << atom_id_molA << ","
                       << pattern_molA << ","
                       << x_molA << ","
                       << y_molA << ","
                       << z_molA << ","
                       << interactionType << ","
                       << interactionDistance << "\n";
        }
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

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
#include <GraphMol/MonomerInfo.h>
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
    Aromatic_ring,
    Metal,
    Chelated,
};

// --------------------------------------------------- OUTPUT'S FILE MANAGEMENT ------------------------------------------------------

std::ofstream outputFile;

void initializeFile(const char* fileName) {
    outputFile.open(fileName, std::ios::out);
    if (outputFile.is_open()) {
        outputFile << "LIGAND_NAME,PROTEIN_ATOM_ID,PROTEIN_PATTERN,PROTEIN_X,PROTEIN_Y,PROTEIN_Z,LIGAND_ATOM_ID,LIGAND_PATTERN,LIGAND_X,LIGAND_Y,LIGAND_Z,INTERACTION_TYPE,INTERACTION_DISTANCE" << std::endl;
        //std::cout << "File " << fileName << " successfully created." << std::endl;
    } else {
        std::cerr << "Error while creating CSV file." << std::endl;
    }
}

void closeFile() {
    if (outputFile.is_open()) {
        outputFile.close();
    }
}

//takes input all the values as parameters and prints on the CSV file passed by reference NB.might be necessary to escape the strings if there can be "," in them
void output(std::string name_molA, std::string name_molB, std::string atom_id_molA, std::string pattern_molA, float x_molA, float y_molA, float z_molA, std::string atom_id_molB, std::string pattern_molB, float x_molB, float y_molB, float z_molB, std::string interactionType, float interactionDistance, const bool protA_ligB){
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

// ----------------------------------------------------------- STRUCTS -------------------------------------------------------------------------

struct SMARTSPattern {
    Pattern pattern;
    std::string smartsString;
};
struct FoundPatterns {
    std::map<Pattern, std::vector<RDKit::MatchVectType>> patternMatches; // Maps every pattern with vector of all it's found istances that are rappresented ad pairs <athom in the pattern, athom in the mol>.
};
struct Molecule {   //This struct is used to save each mol with it's name
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

// ---------------------------------------------------- OTHER UTILITIES -----------------------------------------------------------------------

SMARTSPattern smartsPatterns[] = {
    {Pattern::Hydrophobic , "[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]"},
    {Pattern::Hydrogen_donor_H, "[$([O,S;+0]),$([N;v3,v4&+1]),n+0]-[H]"},
    {Pattern::Hydrogen_acceptor, "[#7&!$([nX3])&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4&+1]),O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O),o+0,F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])]"},
    {Pattern::Halogen_donor_halogen, "[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]"},
    {Pattern::Halogen_acceptor_any, "[#7,#8,P,S,Se,Te,a;!+{1-}][*]"},
    {Pattern::Anion, "[-{1-},$(O=[C,S,P]-[O-])]"},
    {Pattern::Cation, "[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]"},
    {Pattern::Aromatic_ring, "[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1"},
    {Pattern::Aromatic_ring, "[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1"},
    {Pattern::Metal, "[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]"},
    {Pattern::Chelated, "[O,#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4]),-{1-};!+{1-}]"}
};

const int smartsPatternsCount = sizeof(smartsPatterns) / sizeof(SMARTSPattern);

std::string PatternToString(Pattern pattern) {
    switch(pattern) {
        case Pattern::Hydrophobic: return "Hydrophobic"; 
        case Pattern::Hydrogen_donor_H: return "Hydrogen_donor_H"; 
        case Pattern::Hydrogen_acceptor: return "Hydrogen_acceptor"; 
        case Pattern::Halogen_donor_halogen: return "Halogen_donor_halogen"; 
        case Pattern::Halogen_acceptor_any: return "Halogen_acceptor_any"; 
        case Pattern::Anion: return "Anion"; 
        case Pattern::Cation: return "Cation"; 
        case Pattern::Aromatic_ring: return "Aromatic_ring"; 
        case Pattern::Metal: return "Metal"; 
        case Pattern::Chelated: return "Chelated";
        default:    return "Unknown";
    }
}

void printFoundPatterns(FoundPatterns foundPatterns){
    std::cout << "Found patterns [" << foundPatterns.patternMatches.size() << "]: "<< std::endl;

    for(const auto& patternMatch: foundPatterns.patternMatches){
        std::cout << " ------ " << PatternToString(patternMatch.first) << " ------ " << std::endl;

        for(size_t j = 0; j < patternMatch.second.size(); j++){
           std::cout << "    " << j+1 << std::endl;

           for(size_t k = 0; k < patternMatch.second.at(j).size(); k++){
               std::cout << "        " << "First A: " << patternMatch.second.at(j).at(k).first << " Second A: " << patternMatch.second.at(j).at(k).second << std::endl;
           }
        }

        //std::cout << std::endl;
        std::cout << std::endl;
    }
}

void printMolOverview(RDKit::ROMol mol, bool smiles) {
    // Numero di atomi std::cout << "Numero di atomi: " << mol.getNumAtoms() << std::endl;
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

// Creates the PROTEIN_ATOM_ID and LIGAND_ATOM_ID attributes for the CSV file
 void getProtLigAtomID(const Molecule& molA, const Molecule& molB, unsigned int indx_molA, unsigned int indx_molB, std::string &atom_id_prot, std::string &atom_id_lig, const bool protA_ligB){

    if(protA_ligB){ // If molA contains the protein and molB the ligand
        //Creation of PROTEIN_ATOM_ID
        const RDKit::Atom *atomA = molA.mol->getAtomWithIdx(indx_molA);
        if(atomA->getMonomerInfo() && atomA->getMonomerInfo()->getMonomerType() == RDKit::AtomMonomerInfo::PDBRESIDUE){ //Checks that there is MonomerInfo in this atom NB. the second condition is for additional safty but can be removed
            const RDKit::AtomPDBResidueInfo *pdbInfo = static_cast<const RDKit::AtomPDBResidueInfo*>(atomA->getMonomerInfo());  //since there is no AtomPDBResidueInfo getter available a cast is needed
            atom_id_prot = pdbInfo->getChainId() + "." + pdbInfo->getResidueName() + std::to_string(pdbInfo->getResidueNumber()) + "." + pdbInfo->getName();    // Combines the desired values for the protein atom in a string
        }else{
            atom_id_prot = "Error: " + std::to_string(indx_molA) + "(" + atomA->getSymbol() + ")" + " no correct MonomerInfo"; // prints Error and some basic info to identify the atom 
            std::cout<< "Error: " + std::to_string(indx_molA) + "(" + atomA->getSymbol() + ")" + " has no correct MonomerInfo.";
        }
        //Cration of LIGAND_ATOM_ID
        const RDKit::Atom *atomB = molB.mol->getAtomWithIdx(indx_molB);
        atom_id_lig = std::to_string(indx_molB) + "(" + atomB->getSymbol() + ")";   //Combines the desired values for the ligand atom in a string
    }
    else{  // If molA contains the ligand and molB the protein
        const RDKit::Atom *atomB = molB.mol->getAtomWithIdx(indx_molB);
        if(atomB->getMonomerInfo() && atomB->getMonomerInfo()->getMonomerType() == RDKit::AtomMonomerInfo::PDBRESIDUE){
            const RDKit::AtomPDBResidueInfo *pdbInfo = static_cast<const RDKit::AtomPDBResidueInfo*>(atomB->getMonomerInfo());
            atom_id_prot = pdbInfo->getChainId() + "." + pdbInfo->getResidueName() + std::to_string(pdbInfo->getResidueNumber()) + "." + pdbInfo->getName();
        }else{
            atom_id_prot = "Error: " + std::to_string(indx_molB) + "(" + atomB->getSymbol() + ")" + " no correct MonomerInfo";
            std::cout<< "Error: " + std::to_string(indx_molB) + "(" + atomB->getSymbol() + ")" + " has no correct MonomerInfo.";
        }
        const RDKit::Atom *atomA = molA.mol->getAtomWithIdx(indx_molA);
        atom_id_lig = std::to_string(indx_molA) + "(" + atomA->getSymbol() + ")";
    }
}

// ----------------------------------------------------- GEOMETRIC FUNCTIONS --------------------------------------------------------------------

//TODO: mi sa che le due funzioni dopo ci sono gia in rdkit
float dotProduct(const RDGeom::Point3D &vect_a, const RDGeom::Point3D &vect_b) { //calculates the dot product of a vector
    return vect_a.x * vect_b.x + vect_a.y * vect_b.y + vect_a.z * vect_b.z;
}

float norm(const RDGeom::Point3D &vect) { //calculates the norm of a vector
    return sqrt(vect.x * vect.x + vect.y * vect.y + vect.z * vect.z);
}

bool isVectorNull(RDGeom::Point3D &v) {
    return v.length() == 0;
}

// void lineIntersection(float m1, float m2, float q1, float q2, RDGeom::Point3D* intersection){
//     float x, y;

//     if(m1 == m2) intersection = nullptr;

//     x = (q2 - q2) / (m1 - m2);
//     y = m1*x + q1;

//     intersection->x = x;
//     intersection->y = y;
// }

float calculateRotationAngleY(RDGeom::Point3D& D) {
    return std::atan2(D.z, D.x);
}

float calculateRotationAngleX(RDGeom::Point3D& D) {
    return std::atan2(D.z, D.y);
}

// Applys a rotation to the point around the Y axis, of an angle theta
void rotateY(RDGeom::Point3D* p, float theta) {
        double xNew = cos(theta) * p->x + sin(theta) * p->z;
        double zNew = -sin(theta) * p->x + cos(theta) * p->z;
        p->x = xNew;
        p->z = zNew;
}

// Applys a rotation to the point around the X axis, of an angle theta
void rotateX(RDGeom::Point3D* p, float theta) { 
    double yNew = cos(theta) * p->y - sin(theta) * p->z;
    double zNew = sin(theta) * p->y + cos(theta) * p->z;
    p->y = yNew;
    p->z = zNew;
}

RDGeom::Point3D calculateNormalVector(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b, RDGeom::Point3D &pos_c){  // calculates the normal vector to the plane identified by the 3 points in input (assuming they are not in line)
    RDGeom::Point3D vect_ab = pos_b - pos_a;
    RDGeom::Point3D vect_ac = pos_c - pos_a;

    RDGeom::Point3D normal = vect_ab.crossProduct(vect_ac);
    normal.normalize();
    return normal;
}

float calculateDistance(RDGeom::Point2D &pos_a, RDGeom::Point2D &pos_b){  //calculates euclidian distance between 2 points located in a 2D space
    return (pos_a - pos_b).length();
}

float calculateDistance(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b){  //calculates euclidian distance between 2 points located in a 3D space
    return (pos_a - pos_b).length();
}

float calculateDistance(RDGeom::Point3D &p1, RDGeom::Point3D &p2, RDGeom::Point3D &p3, RDGeom::Point3D &point) { //calculates euclidian distance between the plane formed by the first three points and the fourth point in a 3D space
    
    RDGeom::Point3D normal = calculateNormalVector(p1, p2, p3);

    if(isVectorNull(normal)){
        return -1; //if the three points are aligned the funtions returns -1
    }

    normal.normalize();

    double D = -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z); //caluclates the D coefficient of the plane equation

    // distance formula
    double distance = std::abs(normal.x * point.x + normal.y * point.y + normal.z * point.z + D) / 
                      std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

    return distance;
}


//Having three points located in a 3D space, imagine them forming a triangle: this function calculates the angle on the vertex pos_a 
float calculateAngle(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b, RDGeom::Point3D &pos_c){
    float ab = calculateDistance(pos_a, pos_b);
    float bc = calculateDistance(pos_b, pos_c);
    float ac = calculateDistance(pos_a, pos_c);

    return sin((pow(ab, 2) + pow(ac, 2) - pow(bc, 2)) / (2*ab*ac));
}

bool isAngleInRange(float angle, float minAngle, float maxAngle){
    return (angle >= minAngle && angle <= maxAngle) ? true : false;
}

RDGeom::Point3D calculateCentroid(std::vector<RDGeom::Point3D>& pos_points_ring){   // calculates the centroid for a vector of 3D points
    RDGeom::Point3D centroid(0, 0, 0);
    
    for(const auto& point : pos_points_ring){
        centroid += point;
    }

    centroid /= static_cast<double>(pos_points_ring.size());
    return centroid;
}



float calculateVectorAngle(RDGeom::Point3D &vect_a, RDGeom::Point3D &vect_b){ //calculates the angle in degrees between two vectors (the smallest angle of the incidents infinite lines that are formed extending the vectors)
    return std::acos(abs(dotProduct(vect_a, vect_b) / ((norm(vect_a)) * (norm(vect_b))) * 180 / M_PI));
}

//TODO: questa si dovrà chiamare caluclateVectorAngle e per l'altra si trova un altro nome
float calculateActualVectorAngle(RDGeom::Point3D &vect_a, RDGeom::Point3D &vect_b){ //calculates the angle in degrees between two vectors
    return std::acos(dotProduct(vect_a, vect_b) / ((norm(vect_a)) * (norm(vect_b))) * 180 / M_PI);
}

bool isGreaterThenNinety(float value){ //takes a value, returns true if its greater or equal to 90, false if not
    return value >= 90 ? true : false;
}

// ------------------------------------------------------- INTERACTIONS --------------------------------------------------------------------------

void findHydrophobicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Hydrophobic);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Hydrophobic);

    //Check that there is at list one Hydrophobic pattern found on both protein and ligand if yes serches and prints the bonds
    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        RDGeom::Point3D pos_a, pos_b;    //are needed to easly manage x,y,z cordinates that will be feeded to the output funcion
        float distRequired = 4.5;
        float distance;
        unsigned int indx_molA;     //will contain the atom index for molA in order to calculate distances
        unsigned int indx_molB;
        std::string atom_id_molA, atom_id_molB;

        for (const auto& matchVectA : tmpA->second){  //for every block of the vector containing Hydrophobic matcher in molA_patterns.patterMatches
                indx_molA = matchVectA.at(0).second;  //gets the index number of the atom in molA that we whant to check
                pos_a = conformer_molA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){ //for every block of the vector containing Hydrophobic matcher in molB_patterns.patternMatches
                indx_molB = matchVectB.at(0).second;
                pos_b = conformer_molB.getAtomPos(indx_molB);
                distance = calculateDistance(pos_a, pos_b);

                if (distance <= distRequired){
                    getProtLigAtomID(molA, molB, indx_molA, indx_molB, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Hydrophobic\n";
                    output(molA.name, molB.name, atom_id_molA, "Hydrophobic", pos_a.x, pos_a.y, pos_a.z, atom_id_molB, "Hydrophobic", pos_b.x, pos_b.y, pos_b.z, "Hydrophobic", distance, protA_ligB);
                }
            }
        }
    }
}

void findHydrogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Hydrogen_donor_H);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Hydrogen_acceptor);
    float distance;
    float distance_required = 3.5;
    float minAngle_required = 130;
    float maxAngle_required = 180;

    if ((molA_pattern != molA_patterns.patternMatches.end()) && (molB_pattern != molB_patterns.patternMatches.end())){ // if there are the researched patterns in both the molucles  

        std::string atom_id_molA, atom_id_molB;
        RDGeom::Point3D pos_donor, pos_hydrogen, pos_acceptor; 

        for(const auto& matchVect_molA : molA_pattern->second){ // for each Hydrogen_donor-H pattern in molA
            int id_donor = matchVect_molA.at(0).second; // gets the donor id
            int id_hydrogen = matchVect_molA.at(1).second; //gets the hydrogen id

            pos_donor = conformer_molA.getAtomPos(id_donor); // gets the 3D positioning of the donor
            pos_hydrogen = conformer_molA.getAtomPos(id_hydrogen); // gets the 3D positioning of the hydrogen

            for(const auto& matchVect_molB : molB_pattern->second){ //for each Hydrogen_acceptor pattern in molB
                int id_acceptor = matchVect_molB.at(0).second; // gets the acceptor id
                pos_acceptor = conformer_molB.getAtomPos(id_acceptor); // gets the 3D positioning of the acceptor

                distance = calculateDistance(pos_donor, pos_acceptor); //finds the distance between donor and acceptor
                float angle = calculateAngle(pos_hydrogen, pos_donor, pos_acceptor); //finds the angle between the donor-hydrogen atoms and the hydrogen-acceptor atoms

                if(distance <= distance_required && isAngleInRange(angle, minAngle_required, maxAngle_required)){
                    getProtLigAtomID(molA, molB, id_hydrogen, id_acceptor, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Hydrogen bond\n";
                    output(molA.name, molB.name, atom_id_molA, "Hydrogen donor", pos_hydrogen.x, pos_hydrogen.y, pos_hydrogen.z, atom_id_molB, "Hydrogen acceptor", pos_acceptor.x, pos_acceptor.y, pos_acceptor.z, "Hydrogen Bond", distance, protA_ligB);
                }
            }
        }
    }
}

void findHalogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Halogen_donor_halogen);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Halogen_acceptor_any);
    float distance;
    float distance_required = 3.5;
    float minAngle_required_first = 130;
    float maxAngle_required_first = 180;
    float minAngle_required_second = 80;
    float maxAngle_required_second = 140;

    if ((molA_pattern != molA_patterns.patternMatches.end()) && (molB_pattern != molB_patterns.patternMatches.end())){ // if there are the researched patterns in both the molucles

        std::string atom_id_molA, atom_id_molB;
        RDGeom::Point3D pos_donor, pos_halogen, pos_acceptor, pos_any;

        for(const auto& matchVect_molA : molA_pattern->second){ // for each Halogen_donor-halogen pattern in molA
            int id_donor = matchVect_molA.at(0).second; // gets the donor id
            int id_halogen = matchVect_molA.at(1).second; //gets the halogen id

            pos_donor = conformer_molA.getAtomPos(id_donor); // gets the 3D positioning of the donor
            pos_halogen = conformer_molA.getAtomPos(id_halogen); // gets the 3D positioning of the halogen

            for(const auto& matchVect_molB : molB_pattern->second){ // for each Halogen_donor-halogen pattern in molB
                int id_acceptor = matchVect_molB.at(0).second; // gets the acceptor id
                int id_any = matchVect_molB.at(1).second; // gets the any id

                pos_acceptor = conformer_molB.getAtomPos(id_acceptor); // gets the 3D positioning of the acceptor
                pos_any = conformer_molB.getAtomPos(id_any); // gets the 3D positioning of the any

                distance = calculateDistance(pos_donor, pos_acceptor); //finds the distance between donor and acceptor
                float firstAngle = calculateAngle(pos_halogen, pos_donor, pos_acceptor); //finds the angle between the donor-halogen atoms and the halogen-acceptor atoms
                float secondAngle = calculateAngle(pos_acceptor, pos_halogen, pos_any); //the angle between the halogen-acceptor atoms and the acceptor-any atoms

                if(distance <= distance_required && isAngleInRange(firstAngle, minAngle_required_first, maxAngle_required_first) && isAngleInRange(secondAngle, minAngle_required_second, maxAngle_required_second)){
                    getProtLigAtomID(molA, molB, id_halogen, id_acceptor, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Halogen bond\n";
                    output(molA.name, molB.name, atom_id_molA, "Halogen donor", pos_halogen.x, pos_halogen.y, pos_halogen.z, atom_id_molB, "Halogen acceptor", pos_acceptor.x, pos_acceptor.y, pos_acceptor.z, "Halogen Bond", distance, protA_ligB);
                }
        
            }
        }
    }
}

void findIonicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Cation);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Anion);
    unsigned int indx_molA;
    unsigned int indx_molB;
    RDGeom::Point3D pos_a, pos_b;
    float distRequired = 4.5;
    float distance;
    std::string atom_id_molA, atom_id_molB;

    // Find cation-anion interaction
    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        for (const auto& matchVectA : tmpA->second){
                indx_molA = matchVectA.at(0).second;
                pos_a = conformer_molA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){
                indx_molB = matchVectB.at(0).second;
                pos_b = conformer_molB.getAtomPos(indx_molB);
                distance = calculateDistance(pos_a, pos_b);

                if (distance <= distRequired){
                    getProtLigAtomID(molA, molB, indx_molA, indx_molB, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Ionic\n";
                    output(molA.name, molB.name, atom_id_molA, "Cation", pos_a.x, pos_a.y, pos_a.z, atom_id_molB, "Anion", pos_b.x, pos_b.y, pos_b.z, "Ionic", distance, protA_ligB);
                }
            }
        }
    }
    
    // Find cation-aromatic_ring interaction
    tmpB = molB_patterns.patternMatches.find(Pattern::Aromatic_ring);
    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        float angle;
        float minAngle_required = 30;
        float maxAngle_required = 150;
        RDGeom::Point3D centroid, normal, pos_c;
        std::vector<RDGeom::Point3D> pos_points_ring;
        for (const auto& matchVectA : tmpA->second){    // Iterats on the Cations patterns
                indx_molA = matchVectA.at(0).second;
                pos_a = conformer_molA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){ // Iterats on the Aromatic ring patterns
                pos_points_ring.clear();
                for(const auto& pairs_molB : matchVectB){   //for every pair <atom in the pattern, atom in the mol>
                    indx_molB = pairs_molB.second;  // currently is not necessary but it could become when we clarify how AtomIDs shoud work
                    pos_b = conformer_molB.getAtomPos(indx_molB);
                    pos_points_ring.push_back(pos_b);   // fils the vector containing the positions in 3D space of the ring atoms
                }
                centroid = calculateCentroid(pos_points_ring);
                distance = calculateDistance(pos_a, centroid);

                if (distance <= distRequired){
                    normal = calculateNormalVector(pos_points_ring.at(0), pos_points_ring.at(1), pos_points_ring.at(2));    //finds the normal vector to the plane defined by the aromatic ring atoms
                    pos_c = normal + centroid; // it' a point on the line normal to the ring and passing throw the centroid
                    angle = calculateAngle(centroid, pos_c, pos_a); // calculates the angle that must be <30 for the Ionic bond requirements
                    if((!isAngleInRange(angle, minAngle_required, maxAngle_required)) || angle == 30 || angle == 150){  //pos_c and pos_a can be on different sides of the aromatic ring plane
                        getProtLigAtomID(molA, molB, indx_molA, indx_molB, atom_id_molA, atom_id_molB, protA_ligB);
                        std::cout << "Ionic\n";
                        output(molA.name, molB.name, atom_id_molA, "Cation", pos_a.x, pos_a.y, pos_a.z, atom_id_molB, "Aromatic_ring", centroid.x, centroid.y, centroid.z, "Ionic", distance, protA_ligB);  // For aromatic ring the name of the last atom in the vector conteining pair <atom of the pattern, atom of the molecule> and the position of the centroid are printed.
                    }
                }
            }
        }
    }
}



void findMetalCoordination(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Metal);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Chelated);

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        RDGeom::Point3D pos_a, pos_b; 
        float distRequired = 2.8;
        float distance;
        unsigned int indx_molA;
        unsigned int indx_molB;
        std::string atom_id_molA, atom_id_molB;

        for (const auto& matchVectA : tmpA->second){
                indx_molA = matchVectA.at(0).second;
                pos_a = conformer_molA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){
                indx_molB = matchVectB.at(0).second;
                pos_b = conformer_molB.getAtomPos(indx_molB);
                distance = calculateDistance(pos_a, pos_b);

                if (distance <= distRequired){
                    getProtLigAtomID(molA, molB, indx_molA, indx_molB, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Metal\n";
                    output(molA.name, molB.name, atom_id_molA, "Metal", pos_a.x, pos_a.y, pos_a.z, atom_id_molB, "Chelated", pos_b.x, pos_b.y, pos_b.z, "Metal", distance, protA_ligB);
                }
            }
        }
    }

}


void identifyInteractions(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns, const RDKit::Conformer& proteinConformer, const RDKit::Conformer& ligandConformer){
    //every function will need to serch all the interactions of that type and for every one found call the output function that adds them to the CSV file
    //considering some interactions can be formed both ways (cation-anion ; anion-cation) we call the find function two times  
    
    findHydrophobicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);

    findHydrogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findHydrogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);

    findHalogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findHalogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);

    findIonicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findIonicInteraction(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);

    findMetalCoordination(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findMetalCoordination(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);
}

// for eatch pattern of the Pattern enum looks if it is in the mol and saves all the matches in the MatchVectType field of the map inside FoundPatterns.
void identifySubstructs(Molecule& molecule, FoundPatterns &foundPatterns){
    for(auto smartsPattern : smartsPatterns){
        std::vector<RDKit::MatchVectType> tmpMatchesVector;
        RDKit::ROMol* patternMol = RDKit::SmartsToMol(smartsPattern.smartsString);
        if (!patternMol) {
            std::cerr << "Failed to convert SMARTS to molecule for pattern: " << smartsPattern.smartsString << std::endl;
            continue;  // Skip this iteration if the molecule could not be created.
}
        bool foundMatch = RDKit::SubstructMatch(*(molecule.mol), *patternMol, tmpMatchesVector);

        if(foundMatch && !tmpMatchesVector.empty()){
            //the number of patterns and their index must be the same inside the Pattern Enum and smartsPatterns
            if(smartsPattern.pattern == Pattern::Aromatic_ring && foundPatterns.patternMatches.find(Pattern::Aromatic_ring) != foundPatterns.patternMatches.end()){ //if others aromatic rings where already found
                foundPatterns.patternMatches[Pattern::Aromatic_ring].insert(foundPatterns.patternMatches[Pattern::Aromatic_ring].end(), tmpMatchesVector.begin(), tmpMatchesVector.end()); //append tmpMatchesVector to the end of the already found aromatic rings
            }
            // else foundPatterns.patternMatches[static_cast<Pattern>(i)] = tmpMatchesVector;
            else foundPatterns.patternMatches[smartsPattern.pattern] = tmpMatchesVector;
        }
        delete patternMol;
        //TODO: maybe its a good idea to also clean the tmpMatchesVector
    }
}

// ------------------------------------------------------- MAIN and INPUT ----------------------------------------------------------------------------------------

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

            //printMolOverview(*(molVector.back().mol), false);

            free(fileContent);
        }
    }
}

int main(int argc, char *argv[]) {  // First argument: PDB file, then a non fixed number of Mol2 files

    std::vector<Molecule> molVector; // Vector of all the molecules with their name, (the first element is always a protein, the other are ligands)

    FoundPatterns proteinPatterns;  //Declares a FoundPattern struct where to save all the pattern found in the protein
    FoundPatterns ligandPatterns;   //Declares a FoundPattern struct where to save all the pattern found in the ligand, the same will be used for all ligand passed in input.

    //the CSV file is created and inicialized with the HEADER line in the main
    initializeFile("interactions.csv");


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

    identifySubstructs(molVector.at(0), proteinPatterns); // Identifies all the istances of patterns inside the protein
    printFoundPatterns(proteinPatterns);
    
    const RDKit::Conformer& proteinConformer = molVector.at(0).mol->getConformer(); //Conformer is a class that represents the 2D or 3D conformation of a molecule

    for(int i = 1; i < argc - 1; i++){ // For every ligand
        identifySubstructs(molVector.at(i), ligandPatterns); // Identifies all the istances of patterns inside the ligand
        printFoundPatterns(ligandPatterns);
        
        const RDKit::Conformer& ligandConformer = molVector.at(i).mol->getConformer();  
        
        identifyInteractions(molVector.at(0), molVector.at(i), proteinPatterns, ligandPatterns, proteinConformer, ligandConformer); //Identifies all the interactions between protein and ligand and adds the to the CSV file
        ligandPatterns.patternMatches.clear();
    } 

    return EXIT_SUCCESS;
}

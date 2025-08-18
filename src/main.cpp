/**
* @file main.cpp
* @brief Entry point for the GPU-accelerated application.
* This file implements the complete logic of the application with CUDA acceleration. It:
* - Loads molecular structures from PDB, Mol2 (and other supported formats) using RDKit
* - Applies SMARTS pattern matching to identify interaction-relevant atoms and groups
* - Offloads distance, angle, and geometric property calculations to CUDA kernels for parallel execution
* - Detects physical interactions between molecules (e.g., hydrophobic, hydrogen bonds, halogen bonds, ionic, π-stacking, metal coordination)
* - Outputs the results to a structured CSV file
* This is the main entry point, coordinating CPU logic with GPU kernels to achieve hardware-accelerated performance.
*/
 
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helpers.cuh"
#include "main.hpp"
#include "nvtx_tags.hpp"
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
#include <GraphMol/MolOps.h>
#include "nvtx_tags.hpp"
#include <GraphMol/RWMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>        

/**
 * @brief A static variable that keeps track of the count of interactions.
 * 
 * This variable is used to store the total number of interactions that have 
 * occurred during the execution of the program.
 */
static std::size_t g_interaction_count = 0;

extern int num_streams = NUM_STREAMS;
extern int blockDimX = BLOCKSIZEX;
extern int blockDimY = BLOCKSIZEY;

/**
 * @enum Pattern
 * @brief Enumeration of atom or group types used for intermolecular interaction matching.
 * 
 * These pattern types correspond to chemical features or functional groups
 * defined via SMARTS patterns. Each entry maps to a specific interaction role.
 */
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

/** @defgroup OutputFileManagement Output File Management
 *  @brief Functions and variables for managing the output CSV file.
 *  @{
 */
/// Output file stream used to write the interaction results in CSV format.
std::ofstream outputFile;

/**
 * @brief Initializes the output CSV file.
 * 
 * Opens a file with the given name and writes the CSV header line.
 * 
 * @param fileName Name of the output CSV file.
 */
void initializeFile(const char* fileName) {
    outputFile.open(fileName, std::ios::out);
    if (outputFile.is_open()) {
        outputFile << "LIGAND_NAME,PROTEIN_ATOM_ID,PROTEIN_PATTERN,PROTEIN_X,PROTEIN_Y,PROTEIN_Z,LIGAND_ATOM_ID,LIGAND_PATTERN,LIGAND_X,LIGAND_Y,LIGAND_Z,INTERACTION_TYPE,INTERACTION_DISTANCE" << std::endl;
        //std::cout << "File " << fileName << " successfully created." << std::endl;
    } else {
        std::cerr << "Error while creating CSV file." << std::endl;
    }
}

/**
 * @brief Closes the output file stream.
 * 
 * Ensures the file is properly closed if it was open.
 */
void closeFile() {
    if (outputFile.is_open()) {
        outputFile.close();
    }
}

/**
 * @brief Writes a single interaction entry to the CSV output file.
 * 
 * This function records the details of a detected interaction between two atoms,
 * including their identifiers, 3D coordinates, assigned SMARTS patterns, interaction type,
 * and interaction distance. It also handles the ordering depending on which molecule
 * is the protein and which is the ligand.
 * 
 * @param name_molA Name of molecule A
 * @param name_molB Name of molecule B
 * @param atom_id_molA Atom ID in molecule A
 * @param pattern_molA SMARTS pattern assigned to the atom in molecule A
 * @param x_molA X coordinate of atom in molecule A
 * @param y_molA Y coordinate of atom in molecule A
 * @param z_molA Z coordinate of atom in molecule A
 * @param atom_id_molB Atom ID in molecule B
 * @param pattern_molB SMARTS pattern assigned to the atom in molecule B
 * @param x_molB X coordinate of atom in molecule B
 * @param y_molB Y coordinate of atom in molecule B
 * @param z_molB Z coordinate of atom in molecule B
 * @param interactionType Type of detected interaction (e.g. Hydrogen bond, Ionic)
 * @param interactionDistance Measured distance between the interacting atoms
 * @param protA_ligB Boolean flag indicating the role of molecules (true = A is protein, B is ligand)
 */
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

    ++g_interaction_count;
}
/** @} */

// ----------------------------------------------------------- STRUCTS -------------------------------------------------------------------------

/**
 * @struct SMARTSPattern
 * @brief Associates a chemical interaction pattern with its corresponding SMARTS string.
 */
struct SMARTSPattern {
    Pattern pattern;
    std::string smartsString;
};

/**
 * @struct FoundPatterns
 * @brief Stores the substructure matches found in a molecule for each defined pattern.
 * 
 * Each pattern is mapped to a vector of MatchVectType, where each match is a pair
 * of indices representing the mapping from atoms in the pattern to atoms in the molecule.
 */
struct FoundPatterns {
    std::map<Pattern, std::vector<RDKit::MatchVectType>> patternMatches; 
};

/**
 * @struct Molecule
 * @brief Holds a molecule and its associated name.
 * 
 * This structure manages the ownership of an RDKit ROMol object via a unique pointer
 * and ensures it is not copied accidentally.
 */
struct Molecule {   //This struct is used to save each mol with it's name
    std::string name;
    std::unique_ptr<RDKit::ROMol> mol;

    /**
     * @brief Constructs a Molecule object from a name and a raw RDKit ROMol pointer.
     * 
     * @param molName Name of the molecule
     * @param molPtr Pointer to an RDKit ROMol object (ownership is transferred)
     */
    Molecule(const std::string& molName, RDKit::ROMol* molPtr)
        : name(molName), mol(molPtr) {}

    /// @brief Disable copy constructor
    Molecule(const Molecule&) = delete;
    Molecule& operator=(const Molecule&) = delete;

    /// @brief Enable move constructor (defaulted)
    Molecule(Molecule&&) noexcept = default;
    Molecule& operator=(Molecule&&) noexcept = default;
};

// ---------------------------------------------------- OTHER UTILITIES -----------------------------------------------------------------------

/**
 * @brief List of predefined SMARTS patterns associated with chemical interaction roles.
 * 
 * Each entry maps a Pattern enum to a SMARTS string used for substructure matching.
 * Note: the Aromatic_ring pattern appears twice (for 5- and 6-membered rings).
 */
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

/**
 * @brief Number of SMARTS patterns defined in the smartsPatterns array.
 */
const int smartsPatternsCount = sizeof(smartsPatterns) / sizeof(SMARTSPattern);

/**
 * @brief Converts a Pattern enum value to its string representation.
 * 
 * @param pattern The pattern enum to convert.
 * @return std::string The corresponding string name, or "Unknown" if not matched.
 */
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

/**
 * @brief Prints all SMARTS pattern matches found in a molecule.
 * 
 * For each pattern in the FoundPatterns structure, this function prints:
 * - The name of the pattern
 * - The number of matches
 * - The mapping between the pattern atom indices and the corresponding atom indices in the molecule
 * 
 * @param foundPatterns A structure containing all matched SMARTS patterns in the molecule.
 */
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

/**
 * @brief Prints a basic overview of a molecule.
 * 
 * This function outputs structural information about the molecule, such as:
 * - Number of bonds
 * - Optionally, its SMILES representation
 * 
 * (Note: molecular formula and weight are available but currently commented out.)
 * 
 * @param mol The molecule to analyze (passed by value).
 * @param smiles If true, prints the SMILES string of the molecule.
 */
void printMolOverview(RDKit::ROMol mol, bool smiles) {
    std::cout << "Numero di legami: " << mol.getNumBonds() << std::endl;
    // Rappresentazione SMILES
    if(smiles){
        std::string smiles = RDKit::MolToSmiles(mol);
        std::cout << "SMILES: " << smiles << std::endl;
    }
}

/**
 * @brief Removes the file extension from a filename string.
 * 
 * This function strips common file extensions (e.g., `.pdb`, `.mol2`)
 * from the input filename. If no dot is found, the original string is returned.
 * 
 * @param filename The name of the file including its extension.
 * @return std::string The filename without the extension.
 */ 
std::string removeFileExtension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

/**
 * @brief Builds the PROTEIN_ATOM_ID and LIGAND_ATOM_ID strings for CSV output.
 * 
 * Depending on whether molA or molB represents the protein, this function:
 * - Extracts PDB-specific identifiers (chain ID, residue name, residue number, atom name) for the protein atom
 * - Constructs a simple identifier with atom index and symbol for the ligand atom
 * 
 * The PDB information is accessed via AtomPDBResidueInfo if available.
 * 
 * @param molA First molecule (protein or ligand)
 * @param molB Second molecule (ligand or protein)
 * @param indx_molA Index of the atom in molA
 * @param indx_molB Index of the atom in molB
 * @param atom_id_prot [out] Formatted identifier string for the protein atom
 * @param atom_id_lig [out] Formatted identifier string for the ligand atom
 * @param protA_ligB If true, molA is the protein and molB is the ligand; otherwise reversed
 */
void getProtLigAtomID(const Molecule& molA, const Molecule& molB, unsigned int indx_molA, unsigned int indx_molB, std::string &atom_id_prot, std::string &atom_id_lig, const bool protA_ligB){

    if(protA_ligB){ // If molA contains the protein and molB the ligand
        //Creation of PROTEIN_ATOM_ID
        const RDKit::Atom *atomA = molA.mol->getAtomWithIdx(indx_molA);
        if(atomA->getMonomerInfo() && atomA->getMonomerInfo()->getMonomerType() == RDKit::AtomMonomerInfo::PDBRESIDUE){ //Checks that there is MonomerInfo in this atom
            const RDKit::AtomPDBResidueInfo *pdbInfo = static_cast<const RDKit::AtomPDBResidueInfo*>(atomA->getMonomerInfo());  //since there is no AtomPDBResidueInfo getter available we cast
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

/** @defgroup GeometricFunctions Geometric Functions
 *  @brief Functions for geometric calculations in 2D and 3D space.
 *  @{
 */
/**
 * @brief Calculates the dot product of two 3D vectors.
 * 
 * @param vect_a First vector
 * @param vect_b Second vector
 * @return float The dot product of the two vectors.
 */
float dotProduct(const RDGeom::Point3D &vect_a, const RDGeom::Point3D &vect_b) {
    return vect_a.x * vect_b.x + vect_a.y * vect_b.y + vect_a.z * vect_b.z;
}

/**
 * @brief Calculates the norm of a 3D vector.
 * 
 * @param vect The vector to calculate the norm for.
 * @return float The length of the vector.
 */
float norm(const RDGeom::Point3D &vect) {
    return sqrt(vect.x * vect.x + vect.y * vect.y + vect.z * vect.z);
}

/**
 * @brief Checks if a vector is null (length is zero).
 * 
 * @param v The vector to check.
 * @return bool True if the vector is null, false otherwise.
 */
bool isVectorNull(RDGeom::Point3D &v) {
    return v.length() == 0;
}

/**
 * @brief Calculates the rotation angle around the Y-axis for a given 3D vector.
 * 
 * 
 * @param D A 3D vector represented as RDGeom::Point3D.
 * @return float Rotation angle in radians.
 */
float calculateRotationAngleY(RDGeom::Point3D& D) {
    return std::atan2(D.z, D.x);
}

/**
 * @brief Calculates the rotation angle around the X-axis for a given 3D vector.
 * 
 * @param D A 3D vector represented as RDGeom::Point3D.
 * @return float Rotation angle in radians.
 */
float calculateRotationAngleX(RDGeom::Point3D& D) {
    return std::atan2(D.z, D.y);
}

/**
 * @brief Applies a rotation to a 3D point around the Y-axis.
 * 
 * @param p Pointer to the 3D point to rotate.
 * @param theta Rotation angle in radians.
 */
void rotateY(RDGeom::Point3D* p, float theta) {
        double xNew = cos(theta) * p->x + sin(theta) * p->z;
        double zNew = -sin(theta) * p->x + cos(theta) * p->z;
        p->x = xNew;
        p->z = zNew;
}

/**
 * @brief Applies a rotation to a 3D point around the X-axis.
 * 
 * @param p Pointer to the 3D point to rotate.
 * @param theta Rotation angle in radians.
 */
void rotateX(RDGeom::Point3D* p, float theta) { 
    double yNew = cos(theta) * p->y - sin(theta) * p->z;
    double zNew = sin(theta) * p->y + cos(theta) * p->z;
    p->y = yNew;
    p->z = zNew;
}

/**
 * @brief Checks if two segments in 2D space intersect.
 * 
 * This function checks if two line segments defined by points a1, b1 and a2, b2 intersect.
 * It assumes that the segments are coplanar.
 * 
 * @param a1 First point of the first segment
 * @param b1 Second point of the first segment
 * @param a2 First point of the second segment
 * @param b2 Second point of the second segment
 * @return bool True if the segments intersect, false otherwise.
 */
bool doSegmentsIntersect(RDGeom::Point3D &a1, RDGeom::Point3D &b1, RDGeom::Point3D &a2, RDGeom::Point3D &b2){ 
    RDGeom::Point3D a1a2 = a1 - a2;
    RDGeom::Point3D b1b2 = b1 - b2;
    RDGeom::Point3D a1b1 = a1 - b1;

    double a =  a1a2.x, b = -b1b2.x, c = a1a2.y, d = -b1b2.y;//fill the coeficients in the matrix rapresenting the equations
    double determinant = a * d - b * c;

    if(fabs(determinant) < 1e-10) return false; //checks if the determinant = 0 it means that there are no solutions the segments are or parallel or the same segment

    double t = (d * a1b1.x -b * a1b1.y) / determinant; //solves the equation using Cramer's rule
    double s = (-c * a1b1.x + a * a1b1.y) / determinant;

    return (t >= 0 && t<= 1 && s >= 0 && s <= 1);
}

/**
 * @brief Calculates the normal vector to the plane defined by three points in 3D space.
 * 
 * This function assumes that the three points are not collinear.
 * 
 * @param pos_a First point
 * @param pos_b Second point
 * @param pos_c Third point
 * @return RDGeom::Point3D The normal vector to the plane defined by the three points.
 */
RDGeom::Point3D calculateNormalVector(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b, RDGeom::Point3D &pos_c){  // calculates the normal vector to the plane identified by the 3 points in input (assuming they are not in line)
    RDGeom::Point3D vect_ab = pos_b - pos_a;
    RDGeom::Point3D vect_ac = pos_c - pos_a;

    RDGeom::Point3D normal = vect_ab.crossProduct(vect_ac);
    normal.normalize();
    return normal;
}

/**
 * @brief Calculates the Euclidean distance between two points in 2D space.
 * 
 * @param pos_a First point
 * @param pos_b Second point
 * @return float The distance between the two points.
 */
float calculateDistance(RDGeom::Point2D &pos_a, RDGeom::Point2D &pos_b){
    return (pos_a - pos_b).length();
}

/**
 * @brief Calculates the Euclidean distance between two points in 3D space.
 * 
 * @param pos_a First point
 * @param pos_b Second point
 * @return float The distance between the two points.
 */
float calculateDistance(const RDGeom::Point3D &pos_a, const RDGeom::Point3D &pos_b){
    float x_diff = pos_a.x - pos_b.x; 
    float y_diff = pos_a.y - pos_b.y; 
    float z_diff = pos_a.z - pos_b.z;  

    return std::sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
}

/**
 * @brief Calculates the distance from a point to the plane defined by three other points in 3D space.
 * 
 * This function returns -1 if the three points are collinear.
 * 
 * @param p1 First point defining the plane
 * @param p2 Second point defining the plane
 * @param p3 Third point defining the plane
 * @param point The point to measure the distance to
 * @return float The distance from the point to the plane, or -1 if collinear.
 */
float calculateDistance(RDGeom::Point3D &p1, RDGeom::Point3D &p2, RDGeom::Point3D &p3, RDGeom::Point3D &point) {
    
    RDGeom::Point3D normal = calculateNormalVector(p1, p2, p3);

    if(isVectorNull(normal)){
        return -1;
    }

    normal.normalize();

    double D = -(normal.x * p1.x + normal.y * p1.y + normal.z * p1.z);

    double distance = std::abs(normal.x * point.x + normal.y * point.y + normal.z * point.z + D) / 
                    std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);

    return distance;
}

/**
 * @brief Calculates the angle in degrees between three points in 3D space.
 * 
 * This function assumes that the points are not collinear.
 * Having three points located in a 3D space, imagine them forming a triangle: this function calculates the angle in degreeson of the vertex pos_a 
 * 
 * @param pos_a First point (vertex of the angle)
 * @param pos_b Second point
 * @param pos_c Third point
 * @return float The angle in degrees at pos_a.
 */
float calculateAngle(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b, RDGeom::Point3D &pos_c){
    float ab = calculateDistance(pos_a, pos_b);
    float bc = calculateDistance(pos_b, pos_c);
    float ac = calculateDistance(pos_a, pos_c);

    return (acos((pow(ab, 2) + pow(ac, 2) - pow(bc, 2)) / (2 * ab * ac))) * (180.0 / M_PI);
}

/**
 * @brief Checks if an angle is within a specified range.
 * 
 * @param angle The angle to check
 * @param minAngle The minimum angle of the range
 * @param maxAngle The maximum angle of the range
 * @return bool True if the angle is within the range, false otherwise.
 */
bool isAngleInRange(float angle, float minAngle, float maxAngle){
    return (angle >= minAngle && angle <= maxAngle) ? true : false;
}

/**
 * @brief Calculates the centroid of a vector of 3D points.
 * 
 * @param pos_points_ring A vector of 3D points
 * @return RDGeom::Point3D The centroid of the points.
 */
RDGeom::Point3D calculateCentroid(std::vector<RDGeom::Point3D>& pos_points_ring){
    RDGeom::Point3D centroid(0, 0, 0);
    
    for(const auto& point : pos_points_ring){
        centroid += point;
    }

    centroid /= static_cast<double>(pos_points_ring.size());
    return centroid;
}


/**
 * @brief Returns the acute angle (in degrees) between two 3D vectors.
 *
 * Computes θ = arccos( | dot(a,b) / (||a||·||b||) | ), yielding an angle in [0°, 90°].
 * The absolute value makes the result invariant to vector direction (a vs. −a). The dot ratio is clamped to [-1, 1] to avoid
 * NaNs from floating point rounding. If either vector has near-zero length, 0° is returned.
 *
 * @param vect_a First vector (not required to be normalized).
 * @param vect_b Second vector (not required to be normalized).
 * @return float Acute angle in degrees between the two vectors.
 */
float calculateVectorAngle(RDGeom::Point3D &vect_a, RDGeom::Point3D &vect_b){
    float dot = dotProduct(vect_a, vect_b);
    float norms = norm(vect_a) * norm(vect_b);
    float angle = std::acos(abs(dot / norms));
    return angle * 180 / M_PI; 
}

/**
 * @brief Calculates the actual angle between two vectors in 3D space.
 * 
 * This function calculates the angle in degrees between two vectors using the dot product formula.
 * 
 * @param vect_a First vector
 * @param vect_b Second vector
 * @return float The angle in degrees between the two vectors.
 */
float calculateActualVectorAngle(RDGeom::Point3D &vect_a, RDGeom::Point3D &vect_b){
    return std::acos(dotProduct(vect_a, vect_b) / ((norm(vect_a)) * (norm(vect_b))) * 180 / M_PI);
}

/**
 * @brief Checks if a value is greater than or equal to 90.
 * 
 * @param value The value to check
 * @return bool True if the value is greater than or equal to 90, false otherwise.
 */
bool isGreaterThenNinety(float value){
    return value >= 90 ? true : false;
}
/** @} */

// ------------------------------------------------------- INTERACTIONS --------------------------------------------------------------------------




// Dichiarazione della funzione wrapper definita in kernel.cu

// --- Wrappers CUDA: assumono mappatura coalescente B->X, A->Y ---

/**
* @brief Launches the CUDA kernel to detect hydrophobic interactions.
*
* @param d_posA_x Device array of x-coordinates for atoms in molecule A
* @param d_posA_y Device array of y-coordinates for atoms in molecule A
* @param d_posA_z Device array of z-coordinates for atoms in molecule A
* @param d_posB_x Device array of x-coordinates for atoms in molecule B
* @param d_posB_y Device array of y-coordinates for atoms in molecule B
* @param d_posB_z Device array of z-coordinates for atoms in molecule B
* @param d_distances Device array to store computed atom-atom distances
* @param numA Number of atoms in molecule A
* @param numB Number of atoms in molecule B
* @param blockSizeX CUDA block size in X dimension
* @param blockSizeY CUDA block size in Y dimension
* @param stream CUDA stream for asynchronous execution
*/
extern void launchHydrophobicBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                        float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                        float* d_distances, int numA, int numB,
                                        int blockSizeX, int blockSizeY, cudaStream_t stream);


                                        /**
* @brief Launches the CUDA kernel to detect hydrogen bonds.
*
* @param d_donor_x Device array of x-coordinates for donor atoms
* @param d_donor_y Device array of y-coordinates for donor atoms
* @param d_donor_z Device array of z-coordinates for donor atoms
* @param d_hydrogen_x Device array of x-coordinates for hydrogen atoms
* @param d_hydrogen_y Device array of y-coordinates for hydrogen atoms
* @param d_hydrogen_z Device array of z-coordinates for hydrogen atoms
* @param d_acceptor_x Device array of x-coordinates for acceptor atoms
* @param d_acceptor_y Device array of y-coordinates for acceptor atoms
* @param d_acceptor_z Device array of z-coordinates for acceptor atoms
* @param d_distances Device array to store donor–acceptor distances
* @param numDonors Number of donor atoms
* @param numAcceptors Number of acceptor atoms
* @param blockSizeX CUDA block size in X dimension
* @param blockSizeY CUDA block size in Y dimension
*/
extern void launchHydrogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                     float* d_hydrogen_x, float* d_hydrogen_y, float* d_hydrogen_z,
                                     float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                     float* d_distances,
                                     int numDonors, int numAcceptors,
                                     int blockSizeX, int blockSizeY);


/**
* @brief Launches the CUDA kernel to detect halogen bonds.
*
* @param d_donor_x Device array of x-coordinates for donor atoms
* @param d_donor_y Device array of y-coordinates for donor atoms
* @param d_donor_z Device array of z-coordinates for donor atoms
* @param d_halogen_x Device array of x-coordinates for halogen atoms
* @param d_halogen_y Device array of y-coordinates for halogen atoms
* @param d_halogen_z Device array of z-coordinates for halogen atoms
* @param d_acceptor_x Device array of x-coordinates for acceptor atoms
* @param d_acceptor_y Device array of y-coordinates for acceptor atoms
* @param d_acceptor_z Device array of z-coordinates for acceptor atoms
* @param d_any_x Device array of x-coordinates for adjacent atoms
* @param d_any_y Device array of y-coordinates for adjacent atoms
* @param d_any_z Device array of z-coordinates for adjacent atoms
* @param d_distances Device array to store donor–acceptor distances
* @param numDonors Number of donor atoms
* @param numAcceptors Number of acceptor atoms
* @param blockSizeX CUDA block size in X dimension
* @param blockSizeY CUDA block size in Y dimension
* @param stream CUDA stream for asynchronous execution
*/
extern void launchHalogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                    float* d_halogen_x, float* d_halogen_y, float* d_halogen_z,
                                    float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                    float* d_any_x, float* d_any_y, float* d_any_z,
                                    float* d_distances,
                                    int numDonors, int numAcceptors,
                                    int blockSizeX, int blockSizeY, cudaStream_t stream);

/**
* @brief Launches the CUDA kernel to detect ionic interactions (cation–anion).
*
* @param d_cation_x Device array of x-coordinates for cations
* @param d_cation_y Device array of y-coordinates for cations
* @param d_cation_z Device array of z-coordinates for cations
* @param d_anion_x Device array of x-coordinates for anions
* @param d_anion_y Device array of y-coordinates for anions
* @param d_anion_z Device array of z-coordinates for anions
* @param d_distances Device array to store cation–anion distances
* @param numCations Number of cations
* @param numAnions Number of anions
* @param blockSizeX CUDA block size in X dimension
* @param blockSizeY CUDA block size in Y dimension
*/
extern void launchIonicInteractionsKernel_CationAnion(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                      float* d_anion_x, float* d_anion_y, float* d_anion_z,
                                                      float* d_distances, int numCations, int numAnions,
                                                      int blockSizeX, int blockSizeY);


/**
* @brief Launches the CUDA kernel to detect ionic interactions (cation–aromatic ring).
*
* @param d_cation_x Device array of x-coordinates for cations
* @param d_cation_y Device array of y-coordinates for cations
* @param d_cation_z Device array of z-coordinates for cations
* @param d_ring_centroid_x Device array of x-coordinates for ring centroids
* @param d_ring_centroid_y Device array of y-coordinates for ring centroids
* @param d_ring_centroid_z Device array of z-coordinates for ring centroids
* @param d_ring_normal_x Device array of x-components of ring normals
* @param d_ring_normal_y Device array of y-components of ring normals
* @param d_ring_normal_z Device array of z-components of ring normals
* @param d_distances Device array to store cation–centroid distances
* @param d_angles Device array to store cation–ring angle values
* @param numCations Number of cations
* @param numRings Number of aromatic rings
* @param blockSizeX CUDA block size in X dimension
* @param blockSizeY CUDA block size in Y dimension
*/
extern void launchIonicInteractionsKernel_CationRing(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                     float* d_ring_centroid_x, float* d_ring_centroid_y, float* d_ring_centroid_z,
                                                     float* d_ring_normal_x, float* d_ring_normal_y, float* d_ring_normal_z,
                                                     float* d_distances, float* d_angles,
                                                     int numCations, int numRings,
                                                     int blockSizeX, int blockSizeY);

                                                     
/**
* @brief Launches the CUDA kernel to detect π-stacking interactions.
*
* @param d_centroidA_x Device array of x-coordinates for centroids of rings in set A
* @param d_centroidA_y Device array of y-coordinates for centroids of rings in set A
* @param d_centroidA_z Device array of z-coordinates for centroids of rings in set A
* @param d_normalA_x Device array of x-components of ring normals in set A
* @param d_normalA_y Device array of y-components of ring normals in set A
* @param d_normalA_z Device array of z-components of ring normals in set A
* @param d_centroidB_x Device array of x-coordinates for centroids of rings in set B
* @param d_centroidB_y Device array of y-coordinates for centroids of rings in set B
* @param d_centroidB_z Device array of z-coordinates for centroids of rings in set B
* @param d_normalB_x Device array of x-components of ring normals in set B
* @param d_normalB_y Device array of y-components of ring normals in set B
* @param d_normalB_z Device array of z-components of ring normals in set B
* @param d_distances Device array to store centroid–centroid distances
* @param d_planesAngles Device array to store angles between ring planes
* @param d_normalCentroidAnglesA Device array to store normal–centroid angles for set A
* @param d_normalCentroidAnglesB Device array to store normal–centroid angles for set B
* @param numRingsA Number of aromatic rings in set A
* @param numRingsB Number of aromatic rings in set B
* @param blockSizeX CUDA block size in X dimension
* @param blockSizeY CUDA block size in Y dimension
*/
extern void launchPiStackingKernel(float* d_centroidA_x, float* d_centroidA_y, float* d_centroidA_z,
                                   float* d_normalA_x,   float* d_normalA_y,   float* d_normalA_z,
                                   float* d_centroidB_x, float* d_centroidB_y, float* d_centroidB_z,
                                   float* d_normalB_x,   float* d_normalB_y,   float* d_normalB_z,
                                   float* d_distances, float* d_planesAngles,
                                   float* d_normalCentroidAnglesA, float* d_normalCentroidAnglesB,
                                   int numRingsA, int numRingsB,
                                   int blockSizeX, int blockSizeY);


/**
* @brief Launches the CUDA kernel to detect metal coordination bonds.
*
* @param d_posA_x Device array of x-coordinates for metal atoms
* @param d_posA_y Device array of y-coordinates for metal atoms
* @param d_posA_z Device array of z-coordinates for metal atoms
* @param d_posB_x Device array of x-coordinates for chelating atoms
* @param d_posB_y Device array of y-coordinates for chelating atoms
* @param d_posB_z Device array of z-coordinates for chelating atoms
* @param d_distances Device array to store metal–chelator distances
* @param numA Number of metal atoms
* @param numB Number of chelating atoms
* @param blockSizeX CUDA block size in X dimension
* @param blockSizeY CUDA block size in Y dimension
* @param stream CUDA stream for asynchronous execution
*/
extern void launchMetalBondKernel(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                  float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                  float* d_distances, int numA, int numB,
                                  int blockSizeX, int blockSizeY, cudaStream_t stream);

   



/**
 * @brief Finds hydrophobic interactions between two molecules using CUDA acceleration.
 *
 * This function identifies hydrophobic atoms in both molecules, transfers their coordinates
 * to the GPU, computes all pairwise distances in parallel, and reports pairs within the
 * hydrophobic cutoff as interactions.
 *
 * @param molA              Molecule A (protein or ligand; used for naming and output IDs).
 * @param molB              Molecule B (protein or ligand; used for naming and output IDs).
 * @param molA_patterns     Found patterns in A; must include hydrophobic atoms.
 * @param molB_patterns     Found patterns in B; must include hydrophobic atoms.
 * @param conformer_molA    RDKit conformer of A providing atomic coordinates.
 * @param conformer_molB    RDKit conformer of B providing atomic coordinates.
 * @param protA_ligB        If true: A = protein, B = ligand; affects atom ID formatting in the output.
 * @param printInteractions If true, prints "Hydrophobic" to stdout for each interaction found.
 *
 * @return void
 */
void findHydrophobicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Hydrophobic);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Hydrophobic);

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())) {
        // Alloca memoria pinned per i vettori di A e B con CudaMallocHost
        float *posA_x, *posA_y, *posA_z;
        float *posB_x, *posB_y, *posB_z;
        float *distances_host;  // Output sulla CPU

        cudaMallocHost(&posA_x, tmpA->second.size() * sizeof(float));
        cudaMallocHost(&posA_y, tmpA->second.size() * sizeof(float));
        cudaMallocHost(&posA_z, tmpA->second.size() * sizeof(float));

        cudaMallocHost(&posB_x, tmpB->second.size() * sizeof(float));
        cudaMallocHost(&posB_y, tmpB->second.size() * sizeof(float));
        cudaMallocHost(&posB_z, tmpB->second.size() * sizeof(float));

        // Serve a tenere traccia degli indici degli atomi di A e B
        std::vector<unsigned int> idxA, idxB;

        // Alloca memoria pinned per le distanze
        cudaMallocHost(&distances_host, tmpA->second.size() * tmpB->second.size() * sizeof(float));

        // Estrae le posizioni atomiche da molA usando RDKit
        size_t idx = 0;
        for (const auto& matchVectA : tmpA->second) {
            unsigned int indx_molA = matchVectA.at(0).second;
            idxA.push_back(indx_molA);
            RDGeom::Point3D posA = conformer_molA.getAtomPos(indx_molA);
            posA_x[idx] = posA.x;
            posA_y[idx] = posA.y;
            posA_z[idx] = posA.z;
            idx++;
        }

        // Estrae le posizioni atomiche da molB usando RDKit
        idx = 0;
        for (const auto& matchVectB : tmpB->second) {
            unsigned int indx_molB = matchVectB.at(0).second;
            idxB.push_back(indx_molB);
            RDGeom::Point3D posB = conformer_molB.getAtomPos(indx_molB);
            posB_x[idx] = posB.x;
            posB_y[idx] = posB.y;
            posB_z[idx] = posB.z;
            idx++;
        }

        // Allocazione della memoria GPU
        float *d_posA_x, *d_posA_y, *d_posA_z;
        float *d_posB_x, *d_posB_y, *d_posB_z;
        float *d_distances;

        cudaMalloc(&d_posA_x, tmpA->second.size() * sizeof(float));
        cudaMalloc(&d_posA_y, tmpA->second.size() * sizeof(float));
        cudaMalloc(&d_posA_z, tmpA->second.size() * sizeof(float));
        cudaMalloc(&d_posB_x, tmpB->second.size() * sizeof(float));
        cudaMalloc(&d_posB_y, tmpB->second.size() * sizeof(float));
        cudaMalloc(&d_posB_z, tmpB->second.size() * sizeof(float));
        cudaMalloc(&d_distances, tmpA->second.size() * tmpB->second.size() * sizeof(float));

        // Numero di stream e chunk
        cudaStream_t streams[num_streams];
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&streams[i]);
        }

        // Trasferisci B solo una volta, al di fuori del ciclo
        cudaMemcpy(d_posB_x, posB_x, tmpB->second.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_posB_y, posB_y, tmpB->second.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_posB_z, posB_z, tmpB->second.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Definisci la dimensione di ciascun chunk
        size_t chunk_sizeA = (tmpA->second.size() + num_streams - 1) / num_streams;

        // Lancia i trasferimenti e i kernel per ogni chunk di A
        for (int stream = 0; stream < num_streams; ++stream) {
            size_t lowerA = stream * chunk_sizeA;
            size_t upperA = std::min(lowerA + chunk_sizeA, tmpA->second.size());
            size_t widthA = upperA - lowerA;

            // Trasferimento dei chunk di A (ognuno su uno stream)
            cudaMemcpyAsync(d_posA_x + lowerA, posA_x + lowerA, widthA * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
            cudaMemcpyAsync(d_posA_y + lowerA, posA_y + lowerA, widthA * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
            cudaMemcpyAsync(d_posA_z + lowerA, posA_z + lowerA, widthA * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);

            int blockSizeX = BLOCKSIZEX;
            int blockSizeY = BLOCKSIZEY;

        launchHydrophobicBondKernel(
            d_posA_x + lowerA, d_posA_y + lowerA, d_posA_z + lowerA,
            d_posB_x,           d_posB_y,           d_posB_z,
            d_distances + static_cast<size_t>(lowerA) * tmpB->second.size(),
            static_cast<int>(widthA),
            static_cast<int>(tmpB->second.size()),
            blockSizeX, blockSizeY, streams[stream]
        );

            // Copia i risultati parziali dalla GPU alla CPU per il chunk di A
            cudaMemcpyAsync(distances_host + lowerA * tmpB->second.size(), d_distances + lowerA * tmpB->second.size(),
                            widthA * tmpB->second.size() * sizeof(float), cudaMemcpyDeviceToHost, streams[stream]);
        }

        // Sincronizza gli stream
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }

        // Post-processamento sulla CPU per identificare interazioni idrofobiche
        for (size_t i = 0; i < tmpA->second.size(); ++i) {
            for (size_t j = 0; j < tmpB->second.size(); ++j) {
                if (distances_host[i * tmpB->second.size() + j] > 0) {
                    std::string atom_id_molA, atom_id_molB;
                        getProtLigAtomID(molA, molB, idxA[i], idxB[j], atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions)
                        std::cout << "Hydrophobic\n";
                    output(molA.name, molB.name, atom_id_molA, "Hydrophobic", posA_x[i], posA_y[i], posA_z[i],
                        atom_id_molB, "Hydrophobic", posB_x[j], posB_y[j], posB_z[j], "Hydrophobic", distances_host[i * tmpB->second.size() + j], protA_ligB);
                }
            }
        }

        cudaFree(d_posA_x);
        cudaFree(d_posA_y);
        cudaFree(d_posA_z);
        cudaFree(d_posB_x);
        cudaFree(d_posB_y);
        cudaFree(d_posB_z);
        cudaFree(d_distances);

        cudaFreeHost(posA_x);
        cudaFreeHost(posA_y);
        cudaFreeHost(posA_z);
        cudaFreeHost(posB_x);
        cudaFreeHost(posB_y);
        cudaFreeHost(posB_z);
        cudaFreeHost(distances_host);

        for (int i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
    }
}

/**
 * @brief Finds hydrogen bonds between two molecules using CUDA acceleration.
 *
 * This function identifies donor–H pairs in molA and acceptors in molB, transfers their
 * coordinates to the GPU, computes donor–H…acceptor geometry in parallel, and reports
 * pairs that satisfy the hydrogen-bond criteria (kernel writes a positive distance for hits).
 *
 * @param molA              Molecule A (protein or ligand; used for naming and output IDs).
 * @param molB              Molecule B (protein or ligand; used for naming and output IDs).
 * @param molA_patterns     Found patterns in A; must include Pattern::Hydrogen_donor_H (donor and its bound H).
 * @param molB_patterns     Found patterns in B; must include Pattern::Hydrogen_acceptor.
 * @param conformer_molA    RDKit conformer of A providing donor and hydrogen 3D coordinates.
 * @param conformer_molB    RDKit conformer of B providing acceptor 3D coordinates.
 * @param protA_ligB        If true: A = protein, B = ligand; affects atom ID formatting in the output.
 * @param printInteractions If true, prints "Hydrogen bond" to stdout for each interaction found.
 *
 * @return void
 */

void findHydrogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Hydrogen_donor_H);
auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Hydrogen_acceptor);

if ((molA_pattern != molA_patterns.patternMatches.end()) && (molB_pattern != molB_patterns.patternMatches.end())) {  
    std::vector<float> donor_x, donor_y, donor_z;
    std::vector<float> hydrogen_x, hydrogen_y, hydrogen_z;
    std::vector<float> acceptor_x, acceptor_y, acceptor_z;
    std::vector<unsigned int> idxA, idxB;

    // Estrazione delle coordinate da molA (donatore e idrogeno)
    for (const auto& matchVect_molA : molA_pattern->second) {
        int id_donor = matchVect_molA.at(0).second;
        int id_hydrogen = matchVect_molA.at(1).second;
        idxA.push_back(id_hydrogen);

        RDGeom::Point3D pos_donor = conformer_molA.getAtomPos(id_donor);
        RDGeom::Point3D pos_hydrogen = conformer_molA.getAtomPos(id_hydrogen);

        donor_x.push_back(pos_donor.x);
        donor_y.push_back(pos_donor.y);
        donor_z.push_back(pos_donor.z);

        hydrogen_x.push_back(pos_hydrogen.x);
        hydrogen_y.push_back(pos_hydrogen.y);
        hydrogen_z.push_back(pos_hydrogen.z);
    }

    // Estrazione delle coordinate da molB (accettore)
    for (const auto& matchVect_molB : molB_pattern->second) {
        int id_acceptor = matchVect_molB.at(0).second;
        idxB.push_back(id_acceptor);
        RDGeom::Point3D pos_acceptor = conformer_molB.getAtomPos(id_acceptor);

        acceptor_x.push_back(pos_acceptor.x);
        acceptor_y.push_back(pos_acceptor.y);
        acceptor_z.push_back(pos_acceptor.z);
    }

    // Allocazione della memoria sulla GPU per le coordinate e i risultati
    float *d_donor_x, *d_donor_y, *d_donor_z;
    float *d_hydrogen_x, *d_hydrogen_y, *d_hydrogen_z;
    float *d_acceptor_x, *d_acceptor_y, *d_acceptor_z;
    float *d_distances;

    cudaMalloc(&d_donor_x, donor_x.size() * sizeof(float));
    cudaMalloc(&d_donor_y, donor_y.size() * sizeof(float));
    cudaMalloc(&d_donor_z, donor_z.size() * sizeof(float));
    cudaMalloc(&d_hydrogen_x, hydrogen_x.size() * sizeof(float));
    cudaMalloc(&d_hydrogen_y, hydrogen_y.size() * sizeof(float));
    cudaMalloc(&d_hydrogen_z, hydrogen_z.size() * sizeof(float));
    cudaMalloc(&d_acceptor_x, acceptor_x.size() * sizeof(float));
    cudaMalloc(&d_acceptor_y, acceptor_y.size() * sizeof(float));
    cudaMalloc(&d_acceptor_z, acceptor_z.size() * sizeof(float));
    cudaMalloc(&d_distances, donor_x.size() * acceptor_x.size() * sizeof(float));

    // Copia dei dati sulla GPU
    cudaMemcpy(d_donor_x, donor_x.data(), donor_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_donor_y, donor_y.data(), donor_y.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_donor_z, donor_z.data(), donor_z.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hydrogen_x, hydrogen_x.data(), hydrogen_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hydrogen_y, hydrogen_y.data(), hydrogen_y.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hydrogen_z, hydrogen_z.data(), hydrogen_z.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceptor_x, acceptor_x.data(), acceptor_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceptor_y, acceptor_y.data(), acceptor_y.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acceptor_z, acceptor_z.data(), acceptor_z.size() * sizeof(float), cudaMemcpyHostToDevice);

    int blockSizeX = BLOCKSIZEX;
    int blockSizeY = BLOCKSIZEY;

    launchHydrogenBondKernel(
        d_donor_x, d_donor_y, d_donor_z,
        d_hydrogen_x, d_hydrogen_y, d_hydrogen_z,
        d_acceptor_x, d_acceptor_y, d_acceptor_z,
        d_distances,
        static_cast<int>(donor_x.size()),
        static_cast<int>(acceptor_x.size()),
        blockSizeX, blockSizeY
    );



    // Copia dei risultati dalla GPU alla CPU
    std::vector<float> distances(donor_x.size() * acceptor_x.size());
    cudaMemcpy(distances.data(), d_distances, distances.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Post-processamento sulla CPU
    for (size_t i = 0; i < donor_x.size(); ++i) {
        for (size_t j = 0; j < acceptor_x.size(); ++j) {
            float distance = distances[i * acceptor_x.size() + j];

            if (distance > 0) {
                std::string atom_id_molA, atom_id_molB;
                getProtLigAtomID(molA, molB, idxA[i], idxB[j], atom_id_molA, atom_id_molB, protA_ligB);
                if (printInteractions)
                    std::cout << "Hydrogen bond\n";
                output(molA.name, molB.name, atom_id_molA, "Hydrogen donor", hydrogen_x[i], hydrogen_y[i], hydrogen_z[i],
                        atom_id_molB, "Hydrogen acceptor", acceptor_x[j], acceptor_y[j], acceptor_z[j], "Hydrogen Bond", distance, protA_ligB);
            }
        }
    }

    cudaFree(d_donor_x);
    cudaFree(d_donor_y);
    cudaFree(d_donor_z);
    cudaFree(d_hydrogen_x);
    cudaFree(d_hydrogen_y);
    cudaFree(d_hydrogen_z);
    cudaFree(d_acceptor_x);
    cudaFree(d_acceptor_y);
    cudaFree(d_acceptor_z);
    cudaFree(d_distances);
    }
}

/**
 * @brief Finds halogen bonds between two molecules using CUDA acceleration.
 *
 * This function collects halogen-bond donors in molA (donor heavy atom D and halogen X) and
 * acceptors in molB (acceptor atom A and a neighboring atom “any” for angle evaluation),
 * transfers coordinates to the GPU (B kept resident, A streamed in chunks), launches
 * `launchHalogenBondKernel` to evaluate X-A geometry in parallel, and reports pairs that
 * satisfy the halogen-bond criteria (the kernel writes a positive value for hits).
 *
 * @param molA              Molecule A (protein or ligand; used for naming and output IDs).
 * @param molB              Molecule B (protein or ligand; used for naming and output IDs).
 * @param molA_patterns     Found patterns in A; must include Pattern::Halogen_donor_halogen (D and X).
 * @param molB_patterns     Found patterns in B; must include Pattern::Halogen_acceptor_any (A and its neighbor).
 * @param conformer_molA    RDKit conformer of A providing donor and halogen 3D coordinates.
 * @param conformer_molB    RDKit conformer of B providing acceptor and neighbor 3D coordinates.
 * @param protA_ligB        If true: A = protein, B = ligand; affects atom ID formatting in the output.
 * @param printInteractions If true, prints "Halogen bond" to stdout for each interaction found.
 *
 * @return void
 */

void findHalogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Halogen_donor_halogen);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Halogen_acceptor_any);

    if (molA_pattern != molA_patterns.patternMatches.end() && molB_pattern != molB_patterns.patternMatches.end()) {
        // Allora memoria pinned per i vettori con cudaMallocHost
        float *donor_x, *donor_y, *donor_z;
        float *halogen_x, *halogen_y, *halogen_z;
        float *acceptor_x, *acceptor_y, *acceptor_z;
        float *any_x, *any_y, *any_z;
        float *distances_host;
        std::vector<unsigned int> idxA, idxB;

        int numDonors = molA_pattern->second.size();
        int numAcceptors = molB_pattern->second.size();

        cudaMallocHost(&donor_x, numDonors * sizeof(float));
        cudaMallocHost(&donor_y, numDonors * sizeof(float));
        cudaMallocHost(&donor_z, numDonors * sizeof(float));
        cudaMallocHost(&halogen_x, numDonors * sizeof(float));
        cudaMallocHost(&halogen_y, numDonors * sizeof(float));
        cudaMallocHost(&halogen_z, numDonors * sizeof(float));
        cudaMallocHost(&acceptor_x, numAcceptors * sizeof(float));
        cudaMallocHost(&acceptor_y, numAcceptors * sizeof(float));
        cudaMallocHost(&acceptor_z, numAcceptors * sizeof(float));
        cudaMallocHost(&any_x, numAcceptors * sizeof(float));
        cudaMallocHost(&any_y, numAcceptors * sizeof(float));
        cudaMallocHost(&any_z, numAcceptors * sizeof(float));

        // Allocazione memoria pinned per le distanze
        cudaMallocHost(&distances_host, numDonors * numAcceptors * sizeof(float));

        // Estrae coordinate da molA (donatori e alogeni)
        for (int i = 0; i < numDonors; ++i) {
            size_t id_donor = molA_pattern->second[i].at(0).second;
            int id_halogen = molA_pattern->second[i].at(1).second;
            idxA.push_back(id_donor);

            RDGeom::Point3D pos_donor = conformer_molA.getAtomPos(id_donor);
            RDGeom::Point3D pos_halogen = conformer_molA.getAtomPos(id_halogen);

            donor_x[i] = pos_donor.x;
            donor_y[i] = pos_donor.y;
            donor_z[i] = pos_donor.z;

            halogen_x[i] = pos_halogen.x;
            halogen_y[i] = pos_halogen.y;
            halogen_z[i] = pos_halogen.z;
        }

        // Estrae coordinate da molB (accettori e atomi generici)
        for (int i = 0; i < numAcceptors; ++i) {
            int id_acceptor = molB_pattern->second[i].at(0).second;
            int id_any = molB_pattern->second[i].at(1).second;
            idxB.push_back(id_acceptor);

            RDGeom::Point3D pos_acceptor = conformer_molB.getAtomPos(id_acceptor);
            RDGeom::Point3D pos_any = conformer_molB.getAtomPos(id_any);

            acceptor_x[i] = pos_acceptor.x;
            acceptor_y[i] = pos_acceptor.y;
            acceptor_z[i] = pos_acceptor.z;

            any_x[i] = pos_any.x;
            any_y[i] = pos_any.y;
            any_z[i] = pos_any.z;
        }

        // Allocazione memoria GPU
        float *d_donor_x, *d_donor_y, *d_donor_z;
        float *d_halogen_x, *d_halogen_y, *d_halogen_z;
        float *d_acceptor_x, *d_acceptor_y, *d_acceptor_z;
        float *d_any_x, *d_any_y, *d_any_z;
        float *d_distances;

        cudaMalloc(&d_donor_x, numDonors * sizeof(float));
        cudaMalloc(&d_donor_y, numDonors * sizeof(float));
        cudaMalloc(&d_donor_z, numDonors * sizeof(float));
        cudaMalloc(&d_halogen_x, numDonors * sizeof(float));
        cudaMalloc(&d_halogen_y, numDonors * sizeof(float));
        cudaMalloc(&d_halogen_z, numDonors * sizeof(float));
        cudaMalloc(&d_acceptor_x, numAcceptors * sizeof(float));
        cudaMalloc(&d_acceptor_y, numAcceptors * sizeof(float));
        cudaMalloc(&d_acceptor_z, numAcceptors * sizeof(float));
        cudaMalloc(&d_any_x, numAcceptors * sizeof(float));
        cudaMalloc(&d_any_y, numAcceptors * sizeof(float));
        cudaMalloc(&d_any_z, numAcceptors * sizeof(float));
        cudaMalloc(&d_distances, numDonors * numAcceptors * sizeof(float));

        // Trasferisce B (accettori e any) solo una volta
        cudaMemcpy(d_acceptor_x, acceptor_x, numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acceptor_y, acceptor_y, numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acceptor_z, acceptor_z, numAcceptors * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_any_x, any_x, numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_any_y, any_y, numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_any_z, any_z, numAcceptors * sizeof(float), cudaMemcpyHostToDevice);

        cudaStream_t streams[num_streams];
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&streams[i]);
        }

        size_t chunk_size = (numDonors + num_streams - 1) / num_streams;

        // Lancio trasferimenti e kernel per ogni chunk di donatori
        for (int stream = 0; stream < num_streams; ++stream) {
            size_t lower = stream * chunk_size;
            size_t upper = std::min(lower + chunk_size, static_cast<size_t>(numDonors));
            size_t width = upper - lower;

            // Trasferimento dei chunk di A (ognuno su uno stream)
            cudaMemcpyAsync(d_donor_x + lower, donor_x + lower, width * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
            cudaMemcpyAsync(d_donor_y + lower, donor_y + lower, width * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
            cudaMemcpyAsync(d_donor_z + lower, donor_z + lower, width * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);

            cudaMemcpyAsync(d_halogen_x + lower, halogen_x + lower, width * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
            cudaMemcpyAsync(d_halogen_y + lower, halogen_y + lower, width * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);
            cudaMemcpyAsync(d_halogen_z + lower, halogen_z + lower, width * sizeof(float), cudaMemcpyHostToDevice, streams[stream]);

            int blockSizeX = BLOCKSIZEX;
            int blockSizeY = BLOCKSIZEY;

            launchHalogenBondKernel(
                d_donor_x + lower,  d_donor_y + lower,  d_donor_z + lower,
                d_halogen_x + lower,d_halogen_y + lower,d_halogen_z + lower,
                d_acceptor_x, d_acceptor_y, d_acceptor_z,
                d_any_x,      d_any_y,      d_any_z,
                d_distances + static_cast<size_t>(lower) * numAcceptors,
                static_cast<int>(width),
                numAcceptors,
                blockSizeX, blockSizeY, streams[stream]
            );


            // Copia i risultati parziali dalla GPU alla CPU per ogni chunk
            cudaMemcpyAsync(distances_host + lower * numAcceptors, d_distances + lower * numAcceptors, width * numAcceptors * sizeof(float), cudaMemcpyDeviceToHost, streams[stream]);
        }

        // Sincronizza gli stream
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }

        // Post-processamento sulla CPU per verificare le interazioni e stampare i risultati
        for (int i = 0; i < numDonors; ++i) {
            for (int j = 0; j < numAcceptors; ++j) {
                if (distances_host[i * numAcceptors + j] > 0) {  // Solo interazioni valide (distanze positive)
                    std::string atom_id_molA, atom_id_molB;
                    getProtLigAtomID(molA, molB, idxA[i], idxB[j], atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions)
                        std::cout << "Halogen bond\n";
                    output(molA.name, molB.name, atom_id_molA, "Halogen donor", halogen_x[i], halogen_y[i], halogen_z[i],
                           atom_id_molB, "Halogen acceptor", acceptor_x[j], acceptor_y[j], acceptor_z[j],
                           "Halogen Bond", distances_host[i * numAcceptors + j], protA_ligB);
                }
            }
        }

        cudaFree(d_donor_x);
        cudaFree(d_donor_y);
        cudaFree(d_donor_z);
        cudaFree(d_halogen_x);
        cudaFree(d_halogen_y);
        cudaFree(d_halogen_z);
        cudaFree(d_acceptor_x);
        cudaFree(d_acceptor_y);
        cudaFree(d_acceptor_z);
        cudaFree(d_any_x);
        cudaFree(d_any_y);
        cudaFree(d_any_z);
        cudaFree(d_distances);

        cudaFreeHost(donor_x);
        cudaFreeHost(donor_y);
        cudaFreeHost(donor_z);
        cudaFreeHost(halogen_x);
        cudaFreeHost(halogen_y);
        cudaFreeHost(halogen_z);
        cudaFreeHost(acceptor_x);
        cudaFreeHost(acceptor_y);
        cudaFreeHost(acceptor_z);
        cudaFreeHost(any_x);
        cudaFreeHost(any_y);
        cudaFreeHost(any_z);
        cudaFreeHost(distances_host);

        for (int i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
    }
}


/**
 * @brief Finds ionic interactions between two molecules using CUDA acceleration.
 *
 * This function extracts cations from molA and anions or aromatic rings from molB,
 * transfers their coordinates to the GPU, computes all pairwise geometries in parallel using
 * dedicated kernels (cation-anion and cation-π), and reports pairs that satisfy the ionic
 * criteria (kernels write a positive value for hits).
 *
 * @param molA              Molecule A (protein or ligand; used for naming and output IDs).
 * @param molB              Molecule B (protein or ligand; used for naming and output IDs).
 * @param molA_patterns     Found patterns in A; must include Pattern::Cation.
 * @param molB_patterns     Found patterns in B; may include Pattern::Anion and/or Pattern::Aromatic_ring.
 * @param conformer_molA    RDKit conformer of A providing 3D coordinates for cations.
 * @param conformer_molB    RDKit conformer of B providing 3D coordinates for anions and ring atoms.
 * @param protA_ligB        If true: A = protein, B = ligand; affects atom ID formatting in the output.
 * @param printInteractions If true, prints a short label for each interaction found.
 *
 * @return void
 */

void findIonicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Cation);
    auto tmpB_anion = molB_patterns.patternMatches.find(Pattern::Anion);
    auto tmpB_ring = molB_patterns.patternMatches.find(Pattern::Aromatic_ring);

    std::vector<float> cation_x, cation_y, cation_z;
    std::vector<float> anion_x, anion_y, anion_z;
    std::vector<float> ring_centroid_x, ring_centroid_y, ring_centroid_z;
    std::vector<float> ring_normal_x, ring_normal_y, ring_normal_z;
    std::vector<unsigned int> idxA, idxB, idxB_ring;

    // Estrazione coordinate cationi
    if (tmpA != molA_patterns.patternMatches.end()) {
        for (const auto& matchVectA : tmpA->second) {
            int indx_molA = matchVectA.at(0).second;
            idxA.push_back(indx_molA);
            RDGeom::Point3D pos_a = conformer_molA.getAtomPos(indx_molA);
            cation_x.push_back(pos_a.x);
            cation_y.push_back(pos_a.y);
            cation_z.push_back(pos_a.z);
        }
    }

    // Estrazione coordinate anioni
    if (tmpB_anion != molB_patterns.patternMatches.end()) {
        for (const auto& matchVectB : tmpB_anion->second) {
            int indx_molB = matchVectB.at(0).second;
            idxB.push_back(indx_molB);
            RDGeom::Point3D pos_b = conformer_molB.getAtomPos(indx_molB);
            anion_x.push_back(pos_b.x);
            anion_y.push_back(pos_b.y);
            anion_z.push_back(pos_b.z);
        }
    }

    // Estrazione coordinate per centri degli anelli aromatici e i loro vettori normali
    if (tmpB_ring != molB_patterns.patternMatches.end()) {
        for (const auto& matchVectB : tmpB_ring->second) {
            std::vector<RDGeom::Point3D> pos_points_ring;
            for (const auto& pairs_molB : matchVectB) {
                int indx_molB = pairs_molB.second;
                RDGeom::Point3D pos_b = conformer_molB.getAtomPos(indx_molB);
                pos_points_ring.push_back(pos_b);
            }
            idxB_ring.push_back(matchVectB.back().second);

            RDGeom::Point3D centroid = calculateCentroid(pos_points_ring);
            RDGeom::Point3D normal = calculateNormalVector(pos_points_ring.at(0), pos_points_ring.at(1), pos_points_ring.at(2));

            ring_centroid_x.push_back(centroid.x);
            ring_centroid_y.push_back(centroid.y);
            ring_centroid_z.push_back(centroid.z);

            ring_normal_x.push_back(normal.x);
            ring_normal_y.push_back(normal.y);
            ring_normal_z.push_back(normal.z);
        }
    }

    int numCations = cation_x.size();
    int numAnions = anion_x.size();
    int numRings = ring_centroid_x.size();

    float *d_cation_x, *d_cation_y, *d_cation_z;
    float *d_anion_x, *d_anion_y, *d_anion_z;
    float *d_ring_centroid_x, *d_ring_centroid_y, *d_ring_centroid_z;
    float *d_ring_normal_x, *d_ring_normal_y, *d_ring_normal_z;
    float *d_distances_anion, *d_distances_ring, *d_angles_ring;

    cudaMalloc(&d_cation_x, numCations * sizeof(float));
    cudaMalloc(&d_cation_y, numCations * sizeof(float));
    cudaMalloc(&d_cation_z, numCations * sizeof(float));
    cudaMalloc(&d_anion_x, numAnions * sizeof(float));
    cudaMalloc(&d_anion_y, numAnions * sizeof(float));
    cudaMalloc(&d_anion_z, numAnions * sizeof(float));
    cudaMalloc(&d_ring_centroid_x, numRings * sizeof(float));
    cudaMalloc(&d_ring_centroid_y, numRings * sizeof(float));
    cudaMalloc(&d_ring_centroid_z, numRings * sizeof(float));
    cudaMalloc(&d_ring_normal_x, numRings * sizeof(float));
    cudaMalloc(&d_ring_normal_y, numRings * sizeof(float));
    cudaMalloc(&d_ring_normal_z, numRings * sizeof(float));
    cudaMalloc(&d_distances_anion, numCations * numAnions * sizeof(float));
    cudaMalloc(&d_distances_ring, numCations * numRings * sizeof(float));
    cudaMalloc(&d_angles_ring, numCations * numRings * sizeof(float));

    // Copia dei dati sulla GPU
    cudaMemcpy(d_cation_x, cation_x.data(), numCations * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cation_y, cation_y.data(), numCations * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cation_z, cation_z.data(), numCations * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anion_x, anion_x.data(), numAnions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anion_y, anion_y.data(), numAnions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anion_z, anion_z.data(), numAnions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ring_centroid_x, ring_centroid_x.data(), numRings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ring_centroid_y, ring_centroid_y.data(), numRings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ring_centroid_z, ring_centroid_z.data(), numRings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ring_normal_x, ring_normal_x.data(), numRings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ring_normal_y, ring_normal_y.data(), numRings * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ring_normal_z, ring_normal_z.data(), numRings * sizeof(float), cudaMemcpyHostToDevice);

    int blockSizeX = BLOCKSIZEX;
    int blockSizeY = BLOCKSIZEY;

    launchIonicInteractionsKernel_CationAnion(d_cation_x, d_cation_y, d_cation_z,
                                          d_anion_x, d_anion_y, d_anion_z,
                                          d_distances_anion, numCations, numAnions,
                                          blockSizeX, blockSizeY);


    // Lancia il kernel per il calcolo delle distanze tra cationi e anelli aromatici
    launchIonicInteractionsKernel_CationRing(d_cation_x, d_cation_y, d_cation_z,
                                         d_ring_centroid_x, d_ring_centroid_y, d_ring_centroid_z,
                                         d_ring_normal_x, d_ring_normal_y, d_ring_normal_z,
                                         d_distances_ring, d_angles_ring, numCations, numRings,
                                         blockSizeX, blockSizeY);


    // Copia risultati dalla GPU alla CPU
    std::vector<float> distances_anion(numCations * numAnions);
    std::vector<float> distances_ring(numCations * numRings);
    cudaMemcpy(distances_anion.data(), d_distances_anion, numCations * numAnions * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances_ring.data(), d_distances_ring, numCations * numRings * sizeof(float), cudaMemcpyDeviceToHost);

    // Post-processamento cationi-anioni
    for (int i = 0; i < numCations; ++i) {
        for (int j = 0; j < numAnions; ++j) {
            float distance = distances_anion[i * numAnions + j];
            if (distance > 0) {
                std::string atom_id_molA, atom_id_molB;
                getProtLigAtomID(molA, molB, idxA[i], idxB[j], atom_id_molA, atom_id_molB, protA_ligB);
                if (printInteractions)
                    std::cout << "Ionic\n";
                output(molA.name, molB.name, atom_id_molA, "Cation", cation_x[i], cation_y[i], cation_z[i],
                       atom_id_molB, "Anion", anion_x[j], anion_y[j], anion_z[j], "Ionic", distance, protA_ligB);
            }
        }
    }

    // Post-processamento cationi-anelli aromatici
    for (int i = 0; i < numCations; ++i) {
        for (int j = 0; j < numRings; ++j) {
            float distance = distances_ring[i * numRings + j];
            if (distance > 0) {
                std::string atom_id_molA, atom_id_molB;
                getProtLigAtomID(molA, molB, idxA[i], idxB_ring[j], atom_id_molA, atom_id_molB, protA_ligB);
                if (printInteractions)
                    std::cout << "Ionic with aromatic ring\n";
                output(molA.name, molB.name, atom_id_molA, "Cation", cation_x[i], cation_y[i], cation_z[i],
                       atom_id_molB, "Aromatic_ring", ring_centroid_x[j], ring_centroid_y[j], ring_centroid_z[j], "Ionic", distance, protA_ligB);
            }
        }
    }

    cudaFree(d_cation_x);
    cudaFree(d_cation_y);
    cudaFree(d_cation_z);
    cudaFree(d_anion_x);
    cudaFree(d_anion_y);
    cudaFree(d_anion_z);
    cudaFree(d_ring_centroid_x);
    cudaFree(d_ring_centroid_y);
    cudaFree(d_ring_centroid_z);
    cudaFree(d_ring_normal_x);
    cudaFree(d_ring_normal_y);
    cudaFree(d_ring_normal_z);
    cudaFree(d_distances_anion);
    cudaFree(d_distances_ring);
    cudaFree(d_angles_ring);
}


/**
 * @brief Detects π-π stacking interactions (Sandwich and T-shape) between two molecules using CUDA.
 *
 * The function locates aromatic rings in both molecules, computes ring centroids and normals on
 * the CPU, transfers these to the GPU, and launches `launchPiStackingKernel` to evaluate, in
 * parallel, for every ring pair (A,B): centroid-centroid distance, inter-plane angle, and the
 * angles between each ring normal and the opposite centroid vector. Host-side post-processing
 * then classifies pairs as:
 *  - **Sandwich** if distance and angular thresholds (DISTANCE_SANDWICH, MIN/MAX_PLANES_ANGLE_SANDWICH,
 *    MIN/MAX_NORMAL_CENTROID_ANGLE_SANDWICH) are met;
 *  - **T-shape** if corresponding thresholds (DISTANCE_TSHAPE, MIN/MAX_PLANES_ANGLE_TSHAPE,
 *    MIN/MAX_NORMAL_CENTROID_ANGLE_TSHAPE) are met **and** the B centroid lies inside the polygon
 *    of ring A (ray-casting in the ring plane).
 *
 * @param molA              Molecule A (protein or ligand; used for naming and output IDs).
 * @param molB              Molecule B (protein or ligand; used for naming and output IDs).
 * @param molA_patterns     Found patterns in A; must include Pattern::Aromatic_ring.
 * @param molB_patterns     Found patterns in B; must include Pattern::Aromatic_ring.
 * @param conformer_molA    RDKit conformer of A used to extract ring atom coordinates.
 * @param conformer_molB    RDKit conformer of B used to extract ring atom coordinates.
 * @param protA_ligB        If true: A = protein, B = ligand; affects atom ID formatting in the output.
 * @param printInteractions If true, prints a label for each interaction found (“Pi Stacking - SANDWICH”
 *                          or “Pi Stacking - T-SHAPE”).
 *
 * @return void
 */

 //two planes facing each other: SANDWICH | two planes perpendicular: T-SHAPE
void findPiStacking(const Molecule& molA, const Molecule& molB,
                    const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns,
                    const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB,
                    const bool protA_ligB, const bool printInteractions)
    {
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Aromatic_ring);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Aromatic_ring);
    if ((molA_pattern == molA_patterns.patternMatches.end()) ||
        (molB_pattern == molB_patterns.patternMatches.end())) {
        return;
    }

    // ---- Prepara anelli: centroidi e normali (CPU) ----
    std::vector<float> cA_x, cA_y, cA_z, nA_x, nA_y, nA_z;
    std::vector<float> cB_x, cB_y, cB_z, nB_x, nB_y, nB_z;
    std::vector<unsigned int> ringA_id, ringB_id; // per output
    std::vector<std::vector<RDGeom::Point3D>> ringsA_points, ringsB_points; // per check T-shape

    // A
    for (const auto& matchVectA : molA_pattern->second) {
        std::vector<RDGeom::Point3D> pos_ringA;
        pos_ringA.reserve(matchVectA.size());
        for (const auto& pairs_molA : matchVectA) {
            unsigned int idx = pairs_molA.second;
            pos_ringA.push_back(conformer_molA.getAtomPos(idx));
        }
        ringsA_points.push_back(pos_ringA);
        ringA_id.push_back(matchVectA.back().second);

        RDGeom::Point3D centroidA = calculateCentroid(pos_ringA);
        RDGeom::Point3D normalA = calculateNormalVector(pos_ringA.at(0), pos_ringA.at(1), pos_ringA.at(2));

        cA_x.push_back(centroidA.x); cA_y.push_back(centroidA.y); cA_z.push_back(centroidA.z);
        nA_x.push_back(normalA.x);   nA_y.push_back(normalA.y);   nA_z.push_back(normalA.z);
    }

    // B
    for (const auto& matchVectB : molB_pattern->second) {
        std::vector<RDGeom::Point3D> pos_ringB;
        pos_ringB.reserve(matchVectB.size());
        for (const auto& pairs_molB : matchVectB) {
            unsigned int idx = pairs_molB.second;
            pos_ringB.push_back(conformer_molB.getAtomPos(idx));
        }
        ringsB_points.push_back(pos_ringB);
        ringB_id.push_back(matchVectB.back().second);

        RDGeom::Point3D centroidB = calculateCentroid(pos_ringB);
        RDGeom::Point3D normalB   = calculateNormalVector(pos_ringB.at(0), pos_ringB.at(1), pos_ringB.at(2));

        cB_x.push_back(centroidB.x); cB_y.push_back(centroidB.y); cB_z.push_back(centroidB.z);
        nB_x.push_back(normalB.x);   nB_y.push_back(normalB.y);   nB_z.push_back(normalB.z);
    }

    const int numA = static_cast<int>(cA_x.size());
    const int numB = static_cast<int>(cB_x.size());
    if (numA == 0 || numB == 0) return;

    float *d_cA_x, *d_cA_y, *d_cA_z, *d_nA_x, *d_nA_y, *d_nA_z;
    float *d_cB_x, *d_cB_y, *d_cB_z, *d_nB_x, *d_nB_y, *d_nB_z;
    float *d_dist, *d_planeAng, *d_nCentA, *d_nCentB;

    cudaMalloc(&d_cA_x, numA*sizeof(float)); cudaMalloc(&d_cA_y, numA*sizeof(float)); cudaMalloc(&d_cA_z, numA*sizeof(float));
    cudaMalloc(&d_nA_x, numA*sizeof(float)); cudaMalloc(&d_nA_y, numA*sizeof(float)); cudaMalloc(&d_nA_z, numA*sizeof(float));
    cudaMalloc(&d_cB_x, numB*sizeof(float)); cudaMalloc(&d_cB_y, numB*sizeof(float)); cudaMalloc(&d_cB_z, numB*sizeof(float));
    cudaMalloc(&d_nB_x, numB*sizeof(float)); cudaMalloc(&d_nB_y, numB*sizeof(float)); cudaMalloc(&d_nB_z, numB*sizeof(float));

    cudaMalloc(&d_dist,     numA*numB*sizeof(float));
    cudaMalloc(&d_planeAng, numA*numB*sizeof(float));
    cudaMalloc(&d_nCentA,   numA*numB*sizeof(float));
    cudaMalloc(&d_nCentB,   numA*numB*sizeof(float));

    cudaMemcpy(d_cA_x, cA_x.data(), numA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cA_y, cA_y.data(), numA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cA_z, cA_z.data(), numA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nA_x, nA_x.data(), numA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nA_y, nA_y.data(), numA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nA_z, nA_z.data(), numA*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_cB_x, cB_x.data(), numB*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cB_y, cB_y.data(), numB*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cB_z, cB_z.data(), numB*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nB_x, nB_x.data(), numB*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nB_y, nB_y.data(), numB*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nB_z, nB_z.data(), numB*sizeof(float), cudaMemcpyHostToDevice);

    const int blockSizeX = BLOCKSIZEX;
    const int blockSizeY = BLOCKSIZEY;

    launchPiStackingKernel(d_cA_x, d_cA_y, d_cA_z,
                           d_nA_x, d_nA_y, d_nA_z,
                           d_cB_x, d_cB_y, d_cB_z,
                           d_nB_x, d_nB_y, d_nB_z,
                           d_dist, d_planeAng, d_nCentA, d_nCentB,
                           numA, numB, blockSizeX, blockSizeY);

    // ---- Copy D→H ----
    std::vector<float> h_dist(numA*numB), h_plane(numA*numB), h_ncA(numA*numB), h_ncB(numA*numB);
    cudaMemcpy(h_dist.data(),  d_dist,     numA*numB*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_plane.data(), d_planeAng, numA*numB*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ncA.data(),   d_nCentA,   numA*numB*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ncB.data(),   d_nCentB,   numA*numB*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_cA_x); cudaFree(d_cA_y); cudaFree(d_cA_z);
    cudaFree(d_nA_x); cudaFree(d_nA_y); cudaFree(d_nA_z);
    cudaFree(d_cB_x); cudaFree(d_cB_y); cudaFree(d_cB_z);
    cudaFree(d_nB_x); cudaFree(d_nB_y); cudaFree(d_nB_z);
    cudaFree(d_dist); cudaFree(d_planeAng); cudaFree(d_nCentA); cudaFree(d_nCentB);

    // ---- Post-processing su CPU ----
    auto angleIn = [](float a, float mn, float mx){ return (a >= mn && a <= mx); };

    // Helper locale: proietta un punto P sul piano (Q, n) e fa ray-casting in quel piano per test di "inside-polygon".
    auto centroidInsideRing = [&](const RDGeom::Point3D& centroidToTest,
                                  const std::vector<RDGeom::Point3D>& ringPts,
                                  const RDGeom::Point3D& ringNormal) -> bool
    {
        // Proiezione sul piano dell'anello: Pproj = P - n * dot(P-Q, n)
        RDGeom::Point3D Q = ringPts.front();
        RDGeom::Point3D n = ringNormal; n.normalize();
        RDGeom::Point3D PQ = centroidToTest - Q;
        RDGeom::Point3D Pproj = centroidToTest - n * dotProduct(PQ, n);

        // Costruisci un vettore qualsiasi nel piano per il raggio
        RDGeom::Point3D ref(1.0, 0.0, 0.0);
        if (fabs(dotProduct(ref, n)) > 0.99) ref = RDGeom::Point3D(0.0, 1.0, 0.0);
        RDGeom::Point3D u = n.crossProduct(ref); u.normalize();

        // Punto lontano nel piano, da usare come "esterno"
        RDGeom::Point3D Pfar = Pproj + u * 1000.0;

        // Conta intersezioni del segmento (Pproj, Pfar) con i lati dell'anello (poligono chiuso)
        int count = 0;
        const size_t N = ringPts.size();
        for (size_t k = 0; k < N; ++k) {
            RDGeom::Point3D e1 = ringPts[k];
            RDGeom::Point3D e2 = ringPts[(k+1)%N];
            RDGeom::Point3D A = Pproj;
            RDGeom::Point3D B = Pfar;
            if (doSegmentsIntersect(A, B, e1, e2)) ++count;
        }
        return (count % 2) == 1; // dispari->inside
    };

    // Output
    unsigned int id_pointA = 0, id_pointB = 0;
    std::string atom_id_molA, atom_id_molB;

    for (int i = 0; i < numA; ++i) {
        for (int j = 0; j < numB; ++j) {
            const int idx = i * numB + j;
            const float distance     = h_dist[idx];
            const float planesAngle  = h_plane[idx];
            const float ncentA       = h_ncA[idx];
            const float ncentB       = h_ncB[idx];

            RDGeom::Point3D centroidA(cA_x[i], cA_y[i], cA_z[i]);
            RDGeom::Point3D centroidB(cB_x[j], cB_y[j], cB_z[j]);
            RDGeom::Point3D normalA  (nA_x[i], nA_y[i], nA_z[i]);
            RDGeom::Point3D normalB  (nB_x[j], nB_y[j], nB_z[j]);

            // SANDWICH
            if (distance > 0.0f && distance <= DISTANCE_SANDWICH &&
                angleIn(planesAngle, MIN_PLANES_ANGLE_SANDWICH, MAX_PLANES_ANGLE_SANDWICH) &&
                angleIn(ncentA, MIN_NORMAL_CENTROID_ANGLE_SANDWICH, MAX_NORMAL_CENTROID_ANGLE_SANDWICH) &&
                angleIn(ncentB, MIN_NORMAL_CENTROID_ANGLE_SANDWICH, MAX_NORMAL_CENTROID_ANGLE_SANDWICH))
            {
                id_pointA = ringA_id[i];
                id_pointB = ringB_id[j];
                getProtLigAtomID(molA, molB, id_pointA, id_pointB, atom_id_molA, atom_id_molB, protA_ligB);
                if (printInteractions) std::cout << "Pi Stacking - SANDWICH\n";
                output(molA.name, molB.name, atom_id_molA, "Aromatic_ring", centroidA.x, centroidA.y, centroidA.z,
                       atom_id_molB, "Aromatic_ring", centroidB.x, centroidB.y, centroidB.z,
                       "Pi Stacking", distance, protA_ligB);
            }
            // T-SHAPE
            else if (distance > 0.0f && distance <= DISTANCE_TSHAPE &&
                     angleIn(planesAngle, MIN_PLANES_ANGLE_TSHAPE, MAX_PLANES_ANGLE_TSHAPE) &&
                     angleIn(ncentA, MIN_NORMAL_CENTROID_ANGLE_TSHAPE, MAX_NORMAL_CENTROID_ANGLE_TSHAPE) &&
                     angleIn(ncentB, MIN_NORMAL_CENTROID_ANGLE_TSHAPE, MAX_NORMAL_CENTROID_ANGLE_TSHAPE))
            {
                bool inside = centroidInsideRing(centroidB, ringsA_points[i], normalA);
                if (inside) {
                    id_pointA = ringA_id[i];
                    id_pointB = ringB_id[j];
                    getProtLigAtomID(molA, molB, id_pointA, id_pointB, atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions) std::cout << "Pi Stacking - T-SHAPE\n";
                    output(molA.name, molB.name, atom_id_molA, "Aromatic_ring", centroidA.x, centroidA.y, centroidA.z,
                           atom_id_molB, "Aromatic_ring", centroidB.x, centroidB.y, centroidB.z,
                           "Pi Stacking", distance, protA_ligB);
                }
            }
        }
    }
}

/**
 * @brief Finds metal–ligand coordination interactions using CUDA acceleration.
 *
 * The function collects metal atoms from molA and chelating atoms from molB using the
 * precomputed patterns, uploads their coordinates to the GPU, computes all pairwise
 * distances in parallel via `launchMetalBondKernel`, and reports pairs that meet the
 * coordination cutoff (the kernel writes a positive value for valid hits).
 *
 * @param molA              Molecule A (protein or ligand; used for naming and output IDs).
 * @param molB              Molecule B (protein or ligand; used for naming and output IDs).
 * @param molA_patterns     Found patterns in A; must include Pattern::Metal.
 * @param molB_patterns     Found patterns in B; must include Pattern::Chelated.
 * @param conformer_molA    RDKit conformer of A providing 3D coordinates for metals.
 * @param conformer_molB    RDKit conformer of B providing 3D coordinates for chelating atoms.
 * @param protA_ligB        If true: A = protein, B = ligand; affects atom ID formatting in the output.
 * @param printInteractions If true, prints "Metal" to stdout for each interaction found.
 *
 * @return void
 */

void findMetalCoordination(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Metal);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Chelated);

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())) {
        std::vector<float> metal_x, metal_y, metal_z;
        std::vector<float> chelated_x, chelated_y, chelated_z;
        std::vector<unsigned int> idxA, idxB;


        // Estrae le coordinate per i metalli
        for (const auto& matchVectA : tmpA->second) {
            unsigned int indx_molA = matchVectA.at(0).second;
            idxA.push_back(indx_molA);
            RDGeom::Point3D pos_a = conformer_molA.getAtomPos(indx_molA);
            metal_x.push_back(pos_a.x);
            metal_y.push_back(pos_a.y);
            metal_z.push_back(pos_a.z);
        }

        // Estrae le coordinate per i chelati
        for (const auto& matchVectB : tmpB->second) {
            unsigned int indx_molB = matchVectB.at(0).second;
            idxB.push_back(indx_molB);
            RDGeom::Point3D pos_b = conformer_molB.getAtomPos(indx_molB);
            chelated_x.push_back(pos_b.x);
            chelated_y.push_back(pos_b.y);
            chelated_z.push_back(pos_b.z);
        }

        int numMetals = metal_x.size();
        int numChelated = chelated_x.size();

        // Allocazione memoria GPU
        float *d_metal_x, *d_metal_y, *d_metal_z;
        float *d_chelated_x, *d_chelated_y, *d_chelated_z;
        float *d_distances;

        cudaMalloc(&d_metal_x, numMetals * sizeof(float));
        cudaMalloc(&d_metal_y, numMetals * sizeof(float));
        cudaMalloc(&d_metal_z, numMetals * sizeof(float));
        cudaMalloc(&d_chelated_x, numChelated * sizeof(float));
        cudaMalloc(&d_chelated_y, numChelated * sizeof(float));
        cudaMalloc(&d_chelated_z, numChelated * sizeof(float));
        cudaMalloc(&d_distances, numMetals * numChelated * sizeof(float));

        // Copia dati sulla GPU
        cudaMemcpy(d_metal_x, metal_x.data(), numMetals * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_metal_y, metal_y.data(), numMetals * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_metal_z, metal_z.data(), numMetals * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chelated_x, chelated_x.data(), numChelated * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chelated_y, chelated_y.data(), numChelated * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chelated_z, chelated_z.data(), numChelated * sizeof(float), cudaMemcpyHostToDevice);

        // Lancia il kernel CUDA per calcolare le distanze
        int blockSizeX = BLOCKSIZEX;
        int blockSizeY = BLOCKSIZEY;
        launchMetalBondKernel(d_metal_x, d_metal_y, d_metal_z,
                                      d_chelated_x, d_chelated_y, d_chelated_z,
                                      d_distances, numMetals, numChelated, blockSizeX, blockSizeY, 0); //stream momentaneamente a 0

        // Copia dei risultati dalla GPU alla CPU
        std::vector<float> distances(numMetals * numChelated);
        cudaMemcpy(distances.data(), d_distances, numMetals * numChelated * sizeof(float), cudaMemcpyDeviceToHost);

        // Post-processamento sulla CPU per stampare i risultati
        for (int i = 0; i < numMetals; ++i) {
            for (int j = 0; j < numChelated; ++j) {
                if (distances[i * numChelated + j] > 0) { 
                    std::string atom_id_molA, atom_id_molB;
                    getProtLigAtomID(molA, molB, idxA[i], idxB[j], atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions)
                        std::cout << "Metal\n";
                    output(molA.name, molB.name, atom_id_molA, "Metal", metal_x[i], metal_y[i], metal_z[i],
                           atom_id_molB, "Chelated", chelated_x[j], chelated_y[j], chelated_z[j],
                           "Metal", distances[i * numChelated + j], protA_ligB);
                }
            }
        }

        cudaFree(d_metal_x);
        cudaFree(d_metal_y);
        cudaFree(d_metal_z);
        cudaFree(d_chelated_x);
        cudaFree(d_chelated_y);
        cudaFree(d_chelated_z);
        cudaFree(d_distances);
    }
}

/**
 * @brief Identifies all relevant intermolecular interactions between a protein and a ligand.
 * 
 * This function calls each specific interaction detection function:
 * - Hydrophobic
 * - Hydrogen bond (both directions)
 * - Halogen bond (both directions)
 * - Ionic (both directions)
 * - Pi-stacking
 * - Metal coordination (both directions)
 * 
 * Each interaction found is recorded via the `output()` function.
 * 
 * @param protein Molecule object representing the protein
 * @param ligand Molecule object representing the ligand
 * @param proteinPatterns SMARTS matches found in the protein
 * @param ligandPatterns SMARTS matches found in the ligand
 * @param proteinConformer 3D conformer of the protein
 * @param ligandConformer 3D conformer of the ligand
 */
void identifyInteractions(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns, const RDKit::Conformer& proteinConformer, const RDKit::Conformer& ligandConformer, const bool printInteractions){
    // every function will need to serch all the interactions of that type and for every one found call the output function that adds them to the CSV file
    // considering some interactions can be formed both ways (cation-anion ; anion-cation) we call the find function two times  
    
    NVTX_PUSH("Hydrophobic Interaction");
    findHydrophobicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
    NVTX_POP(); // Hydrophobic Interaction

    NVTX_PUSH("Hydrogen Bond");
    findHydrogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
    findHydrogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);
    NVTX_POP(); // Hydrogen Bond

    NVTX_PUSH("Halogen Bond");
    findHalogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
    findHalogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);
    NVTX_POP(); // Halogen Bond

    NVTX_PUSH("Ionic Interaction");
    findIonicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
    findIonicInteraction(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);
    NVTX_POP(); // Ionic Interaction

    NVTX_PUSH("Pi Stacking");
    findPiStacking(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
    NVTX_POP(); // Pi Stacking

    NVTX_PUSH("Metal Coordination");
    findMetalCoordination(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
    findMetalCoordination(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);
    NVTX_POP(); // Metal Coordination
}

/**
 * @brief Identifies substructures in a molecule based on predefined SMARTS patterns.
 * 
 * For each pattern defined in the global `smartsPatterns` list:
 * - Converts the SMARTS string to an RDKit molecule.
 * - Applies substructure matching to the input molecule.
 * - If matches are found, stores them in the `FoundPatterns` map.
 * 
 * 
 * @param molecule Molecule to analyze for SMARTS matches.
 * @param foundPatterns Output map storing matches for each recognized pattern.
 */
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
            else foundPatterns.patternMatches[smartsPattern.pattern] = tmpMatchesVector;
        }
        delete patternMol;
    }
}

    // ------------------------------------------------------- MAIN and INPUT ----------------------------------------------------------------------------------------

/**
 * @brief Parse an integer from a C-string, returning a default value on failure.
 *
 * Attempts to convert @p arg to an int with std::stoi; if any exception is thrown,
 * returns @p defaultVal instead.
 *
 * @param arg         Null-terminated numeric string to parse.
 * @param defaultVal  Fallback value returned when parsing fails.
 * @return Parsed integer value, or @p defaultVal if parsing fails.
 */
int parseIntOrDefault(const char* arg, int defaultVal) {
    try {
        return std::stoi(arg);
    } 
    catch (...) {
        return defaultVal;
    }
}

/**
 * @brief Parse a CUDA 2D block size from a string formatted as "XxY".
 *
 * Splits @p input on the first lowercase 'x' and converts the left/right substrings
 * with std::stoi, writing the results to @p x and @p y. If 'x' is not found,
 * @p x and @p y are left unchanged.
 *
 * @param input  String formatted as "widthxheight" (e.g., "32x8").
 * @param x      Reference receiving the parsed X dimension (overwritten on success).
 * @param y      Reference receiving the parsed Y dimension (overwritten on success).
 * @return void
 * @throws std::invalid_argument if a substring is not a valid integer.
 * @throws std::out_of_range     if a substring is numerically out of range for int.
 */
void parseBlockDim(const std::string& input, int& x, int& y) {
    size_t xpos = input.find('x');
    if (xpos != std::string::npos) {
        x = std::stoi(input.substr(0, xpos));
        y = std::stoi(input.substr(xpos + 1));
    }
}

/**
 * @brief Perform a lightweight in-place RDKit sanitization on a mutable molecule.
 *
 * Runs a subset of RDKit sanitization operations: FINDRADICALS, KEKULIZE,
 * SETAROMATICITY, SETCONJUGATION, SETHYBRIDIZATION, and SYMMRINGS. Modifies @p rw
 * in place and records the failing operation (if any) via the local @c failedOp.
 *
 * @param rw  RDKit::RWMol to be sanitized (modified in place).
 * @return void
 * @throws RDKit::MolSanitizeException if sanitization fails (errors are not caught).
 */
static inline void sanitize_light_inplace(RDKit::RWMol &rw) {
    unsigned int failedOp = RDKit::MolOps::SANITIZE_NONE;
    const unsigned int ops =
        RDKit::MolOps::SANITIZE_FINDRADICALS |
        RDKit::MolOps::SANITIZE_KEKULIZE |
        RDKit::MolOps::SANITIZE_SETAROMATICITY |
        RDKit::MolOps::SANITIZE_SETCONJUGATION |
        RDKit::MolOps::SANITIZE_SETHYBRIDIZATION |
        RDKit::MolOps::SANITIZE_SYMMRINGS;

    // firma corretta: (mol, failedOp[out], ops[in], catchErrors)
    RDKit::MolOps::sanitizeMol(rw, failedOp, ops);
}

/**
 * @brief Parses molecular structure files (.pdb and .mol2) from command line arguments.
 * 
 * This function takes a list of input file names and:
 * - Opens each file and reads its contents into memory
 * - Converts the contents into RDKit ROMol objects (PDB for the first file, Mol2 for others)
 * - Stores each resulting molecule (with its filename as name) into a vector of Molecule structs
 * 
 * Memory management is handled via `malloc/free` and `std::unique_ptr`.
 * 
 * @param argv Command-line argument array containing file paths
 * @param argc Number of arguments (including program name)
 * @param molVector Vector that will store parsed molecules as Molecule objects
 */
void input(char **argv, int argc, std::vector<Molecule> &molVector) {
    FILE *file;
    char *fileContent = nullptr;

    for (int i = 1; i < argc; i++) {
        file = fopen(argv[i], "rb");
        if (!file) {
            std::cerr << "Can't open the file " << argv[i] << std::endl;
            continue;
        }

        fseek(file, 0, SEEK_END);
        long fileSize = ftell(file);
        fseek(file, 0, SEEK_SET);

        fileContent = (char*)malloc(fileSize + 1);
        if (!fileContent) {
            std::cerr << "Malloc error\n";
            fclose(file);
            return;
        }

        fread(fileContent, 1, fileSize, file);
        fileContent[fileSize] = '\0';
        fclose(file);

        std::unique_ptr<RDKit::ROMol> mol;

        try {
            // Parse without sanitize and without removing H (per evitare eccezioni di valenza)
            if (i == 1) {
                // PROTEINA (PDB)
                mol.reset(RDKit::PDBBlockToMol(fileContent, false, false)); //(fileContent, sanitize, removeHs)
            } else {
                // LIGANDO (MOL2)
                mol.reset(RDKit::Mol2BlockToMol(fileContent, false, false)); //(fileContent, sanitize, removeHs)
            }
            if (!mol) throw std::runtime_error("RDKit returned null molecule");

            // Passa a RWMol per le operazioni MolOps
            RDKit::RWMol rw(*mol);

            // sanitize leggera (inizializza anche le RingInfo)
            try {
                sanitize_light_inplace(rw);
            } catch (const std::exception &se) {
                std::cerr << "Warning: sanitizeMol(light) failed for " << argv[i]
                        << " -> " << se.what() << "\n";
            }

            RDKit::MolOps::addHs(rw, false, true); //(rows, explicitOnly = false, addCoords = true);

            mol = std::make_unique<RDKit::ROMol>(rw);

        } catch (const std::exception &e) {
            std::cerr << "Failed to parse " << argv[i] << ": " << e.what() << "\n";
            free(fileContent);
            continue;
        }

        if (mol) {
            molVector.emplace_back(removeFileExtension(argv[i]), mol.release());
        }

        free(fileContent);
    }
}

/**
 * @brief Main function of the application.
 * 
 * This program detects intermolecular interactions between a protein (PDB) and one or more ligands (Mol2).
 * 
 * Workflow:
 * - Parses input files from command line (1st = protein, others = ligands)
 * - Initializes the CSV output file
 * - Identifies SMARTS patterns in all molecules
 * - Detects interactions for each ligand vs. protein
 * - Outputs results to CSV
 * 
 * @param argc Number of command line arguments (including program name)
 * @param argv Array of C-style strings representing file paths
 * @return int Exit code
 */
int main(int argc, char *argv[]) {  // First argument: PDB file, then a non fixed number of Mol2 files

    NVTX_PUSH("TotalProgram");

    NVTX_PUSH("Input");

    std::vector<Molecule> molVector; // Vector of all the molecules with their name, (the first element is always a protein, the other are ligands)

    FoundPatterns proteinPatterns;  //Declares a FoundPattern struct where to save all the pattern found in the protein
    FoundPatterns ligandPatterns;   //Declares a FoundPattern struct where to save all the pattern found in the ligand, the same will be used for all ligand passed in input.

    CPUTimer cpu_timer;
    CPUTimer overall_cpu_timer;

    //the CSV file is created and inicialized with the HEADER line in the main
    initializeFile("interactions.csv");

    // Prints the files passed from line (argc, argv)
    if(argc >= 2){
        printf("There are %d files passed as args:\n", argc - 1);
        std::cout << "1-Protein: " << argv[1] << std::endl;
        for(int i = 2; i < argc; i++) {
            std::cout << i << "-Ligand: " << argv[i] << std::endl;
        }
    }

    
    input(argv, argc, molVector);

    NVTX_POP(); // Input

    NVTX_PUSH("IdentifyProtSubstructs");

    identifySubstructs(molVector.at(0), proteinPatterns); // Identifies all the istances of patterns inside the protein
    printFoundPatterns(proteinPatterns);

    NVTX_POP(); // IdentifyProtSubstructs
    
    const RDKit::Conformer& proteinConformer = molVector.at(0).mol->getConformer(); //Conformer is a class that represents the 2D or 3D conformation of a molecule

    for(int i = 1; i < molVector.size(); i++){ // For every ligand

        NVTX_PUSH("IdentifyLigandSubstructs");
        identifySubstructs(molVector.at(i), ligandPatterns); // Identifies all the istances of patterns inside the ligand
        printFoundPatterns(ligandPatterns);
        NVTX_POP(); // IdentifyLigandSubstructs
        
        const RDKit::Conformer& ligandConformer = molVector.at(i).mol->getConformer();  

        NVTX_PUSH("IdentifyInteractions");
        identifyInteractions(molVector.at(0), molVector.at(i), proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true); //Identifies all the interactions between protein and ligand and adds the to the CSV file
        NVTX_POP(); // IdentifyInteractions

        ligandPatterns.patternMatches.clear();
    }

    NVTX_POP(); // TotalProgram

    std::cout << "\nInterazioni totali trovate: " << g_interaction_count << '\n';


    return EXIT_SUCCESS;
}

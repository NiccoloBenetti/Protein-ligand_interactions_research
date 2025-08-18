/**
 * @file main.cpp
 * @brief Entry point for the application.
 * 
 * This file implements the complete logic of the application. It:
 * - Loads molecular structures from PDB and Mol2 files using RDKit
 * - Applies SMARTS pattern matching to identify interaction-relevant atoms and groups
 * - Computes distances, angles and other geometric proprieties to detect physical interactions between molecules
 * - Outputs the results to a structured CSV file
 *
 * This is the main entry point, central logic, and full implementation of the tool.
 */



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



/** @defgroup HydrophobicInteraction Hydrophobic Interaction Constants
 *  @brief Thresholds used to identify hydrophobic interactions.
 *  @{
 */
#define DISTANCE_HYDROPHOBIC 4.5
/** @} */

/** @defgroup HydrogenBond Hydrogen Bond Constants
 *  @brief Thresholds for detecting hydrogen bonds.
 *  @{
 */
#define DISTANCE_HYDROGENBOND 3.5
#define MIN_ANGLE_HYDROGENBOND 130
#define MAX_ANGLE_HYDROGENBOND 180
/** @} */

/** @defgroup HalogenBond Halogen Bond Constants
 *  @brief Thresholds for halogen bond detection.
 *  @{
 */
#define DISTANCE_HALOGENBOND 3.5
#define MIN_ANGLE1_HALOGENBOND 130
#define MAX_ANGLE1_HALOGENBOND 180
#define MIN_ANGLE2_HALOGENBOND 80
#define MAX_ANGLE2_HALOGENBOND 140
/** @} */

/** @defgroup IonicInteraction Ionic Interaction Constants
 *  @brief Distance and angular thresholds for ionic interactions.
 *  @{
 */
#define DISTANCE_IONIC 4.5
#define MIN_ANGLE_IONIC 30
#define MAX_ANGLE_IONIC 150
/** @} */

/** @defgroup PiStackingSandwich Pi-Stacking Sandwich Constants
 *  @brief Thresholds for parallel π-π stacking (sandwich geometry).
 *  @{
 */
#define DISTANCE_SANDWICH 5.5
#define MIN_PLANES_ANGLE_SANDWICH 0
#define MAX_PLANES_ANGLE_SANDWICH 30
#define MIN_NORMAL_CENTROID_ANGLE_SANDWICH 0
#define MAX_NORMAL_CENTROID_ANGLE_SANDWICH 33
/** @} */

/** @defgroup PiStackingTshape Pi-Stacking T-shape Constants
 *  @brief Thresholds for perpendicular π-π interactions (T-shaped geometry).
 *  @{
 */
#define DISTANCE_TSHAPE 6.5
#define MIN_PLANES_ANGLE_TSHAPE 50
#define MAX_PLANES_ANGLE_TSHAPE 90
#define MIN_NORMAL_CENTROID_ANGLE_TSHAPE 0
#define MAX_NORMAL_CENTROID_ANGLE_TSHAPE 30
/** @} */

/** @defgroup MetalCoordination Metal Coordination Constants
 *  @brief Threshold for detecting metal-ligand interactions.
 *  @{
 */
#define DISTANCE_METAL 2.8
/** @} */


/**
 * @enum Pattern
 * @brief Enumeration of atom or group types used for intermolecular interaction matching.
 * 
 * These pattern types correspond to chemical features or functional groups
 * defined via SMARTS patterns. Each entry maps to a specific interaction role.
 */
enum class Pattern {
    Hydrophobic,             ///< Atom with hydrophobic character
    Hydrogen_donor_H,        ///< Hydrogen donor including the hydrogen atom
    Hydrogen_acceptor,       ///< Atom capable of accepting a hydrogen bond
    Halogen_donor_halogen,   ///< Halogen donor including the halogen atom
    Halogen_acceptor_any,    ///< Halogen acceptor and its bonded atom
    Anion,                   ///< Anion
    Cation,                  ///< Cation
    Aromatic_ring,           ///< A 5- or 6-membered aromatic ring
    Metal,                   ///< Metal atom involved in coordination
    Chelated                 ///< Atom capable of chelating a metal
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
    Pattern pattern;           ///< Enum value representing the pattern type
    std::string smartsString; ///< SMARTS string used for substructure matching
};

/**
 * @struct FoundPatterns
 * @brief Stores the substructure matches found in a molecule for each defined pattern.
 * 
 * Each pattern is mapped to a vector of MatchVectType, where each match is a pair
 * of indices representing the mapping from atoms in the pattern to atoms in the molecule.
 */
struct FoundPatterns {
    /// Map of patterns to the list of all matching atom indices in the molecule.
    std::map<Pattern, std::vector<RDKit::MatchVectType>> patternMatches;
};

/**
 * @struct Molecule
 * @brief Holds a molecule and its associated name.
 * 
 * This structure manages the ownership of an RDKit ROMol object via a unique pointer
 * and ensures it is not copied accidentally.
 */
struct Molecule {
    std::string name; ///< Name or identifier of the molecule
    std::unique_ptr<RDKit::ROMol> mol; ///< Pointer to the RDKit molecule structure

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

    /// @brief Disable copy assignment
    Molecule& operator=(const Molecule&) = delete;

    /// @brief Enable move constructor (defaulted)
    Molecule(Molecule&&) noexcept = default;

    /// @brief Enable move assignment (defaulted)
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
            atom_id_prot = "Error: " + std::to_string(indx_molA) + "(" + atomA->getSymbol() + ")" + " no correct MonomerInfo";
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

    double a =  a1a2.x, b = -b1b2.x, c = a1a2.y, d = -b1b2.y;
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
RDGeom::Point3D calculateNormalVector(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b, RDGeom::Point3D &pos_c){
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
float calculateVectorAngle(const RDGeom::Point3D& vect_a, const RDGeom::Point3D& vect_b){
    const double na = std::sqrt(vect_a.x*vect_a.x + vect_a.y*vect_a.y + vect_a.z*vect_a.z);
    const double nb = std::sqrt(vect_b.x*vect_b.x + vect_b.y*vect_b.y + vect_b.z*vect_b.z);
    if (na < 1e-20 || nb < 1e-20) return 0.0f; // avoids NaN on almost zero vectors
    double c = (vect_a.x*vect_b.x + vect_a.y*vect_b.y + vect_a.z*vect_b.z) / (na * nb);
    if (c > 1.0) c = 1.0;
    else if (c < -1.0) c = -1.0;
    return static_cast<float>(std::acos(std::fabs(c)) * 180.0 / M_PI);
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

/**
 * @brief Tests whether a 3D point lies inside a planar ring polygon using projection + ray casting.
 *
 * @details
 * Projects the test point and the ring vertices onto the plane defined by @p ringNormal and the
 * first vertex of @p ringPts. An orthonormal in-plane basis (u,v) is constructed by projecting the
 * first edge onto the plane (fallback to a safe reference if degenerate), then a standard 2D
 * ray-casting test is performed along +u: the number of intersections between the half-ray and the
 * polygon edges is counted using @c doSegmentsIntersect, and inside-ness is decided by odd parity
 * (count % 2 == 1). This replicates the GPU semantics for the T-shape “inside-polygon” check.
 *
 * Assumptions: @p ringPts describes a simple (non-self-intersecting) polygon whose vertices lie
 * approximately on the plane orthogonal to @p ringNormal and are ordered around the perimeter.
 * The normal need not be unit length; it is normalized internally.
 *
 * Numerical notes: small degeneracies are handled (clamping/normalization; fallback basis if the
 * first edge is nearly parallel to @p ringNormal). If @p ringPts has fewer than 3 vertices, the
 * function returns false.
 *
 *
 * @param centroidToTest The 3D point to test (e.g., centroid of ring B).
 * @param ringPts        The 3D vertices of the ring polygon (ring A), ordered around the ring.
 * @param ringNormal     The (possibly non-unit) normal of the ring plane.
 * @return true if the projected point is inside the projected polygon by odd-parity rule; false otherwise.
 */

bool centroidInsideRing(const RDGeom::Point3D& centroidToTest,
                        const std::vector<RDGeom::Point3D>& ringPts,
                        RDGeom::Point3D ringNormal)
{
    if (ringPts.size() < 3) return false;

    ringNormal.normalize();
    const RDGeom::Point3D Q = ringPts.front();

    RDGeom::Point3D u = ringPts[1] - ringPts[0];
    u -= ringNormal * dotProduct(u, ringNormal);
    if (u.length() < 1e-8) {
        RDGeom::Point3D ref(1.0, 0.0, 0.0);
        if (std::fabs(dotProduct(ref, ringNormal)) > 0.99) ref = RDGeom::Point3D(0.0, 1.0, 0.0);
        u = ringNormal.crossProduct(ref);
    }
    u.normalize();
    RDGeom::Point3D v = ringNormal.crossProduct(u);
    v.normalize();

    auto proj2D = [&](const RDGeom::Point3D& P) -> RDGeom::Point3D {
        RDGeom::Point3D rel = P - Q;
        return RDGeom::Point3D(dotProduct(rel, u), dotProduct(rel, v), 0.0);
    };

    RDGeom::Point3D P2 = proj2D(centroidToTest);
    std::vector<RDGeom::Point3D> ring2D; ring2D.reserve(ringPts.size());
    for (const auto& rp : ringPts) ring2D.push_back(proj2D(rp));

    // Ray orizzontale verso +x
    RDGeom::Point3D Pfar(P2.x + 1e6, P2.y, 0.0);

    int count = 0;
    const size_t N = ring2D.size();
    for (size_t k = 0; k < N; ++k) {
        RDGeom::Point3D A = P2, B = Pfar;                  
        RDGeom::Point3D e1 = ring2D[k], e2 = ring2D[(k+1)%N];
        if (doSegmentsIntersect(A, B, e1, e2)) ++count;
    }
    return (count % 2) == 1;
}


// ------------------------------------------------------- INTERACTIONS --------------------------------------------------------------------------

/**
 * @defgroup Interactions Interaction Functions
 * @brief Functions for detecting and analyzing molecular interactions.
 * @{
 */

/**
 * @brief Detects hydrophobic interactions between two molecules.
 * 
 * This function searches for hydrophobic atoms (as defined by SMARTS pattern matches)
 * in both molecules. For each pair of hydrophobic atoms, it calculates the distance
 * and determines whether they meet the criteria for a hydrophobic interaction.
 * 
 * Matching interactions are printed and written to CSV.
 * 
 * @param molA First molecule (protein or ligand)
 * @param molB Second molecule (ligand or protein)
 * @param molA_patterns Found SMARTS pattern matches in molA
 * @param molB_patterns Found SMARTS pattern matches in molB
 * @param conformer_molA Conformer of molA containing 3D coordinates
 * @param conformer_molB Conformer of molB containing 3D coordinates
 * @param protA_ligB Flag indicating whether molA is the protein and molB is the ligand
 */
void findHydrophobicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Hydrophobic);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Hydrophobic);

    //Check that there is at list one Hydrophobic pattern found on both protein and ligand if yes serches and prints the bonds
    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        RDGeom::Point3D pos_a, pos_b;    //are needed to easly manage x,y,z cordinates that will be feeded to the output funcion
        float distance;
        unsigned int indx_molA;     //will contain the atom index for molA in order to calculate distances
        unsigned int indx_molB;
        std::string atom_id_molA, atom_id_molB;

        for (const auto& matchVectA : tmpA->second){  //for every element of the vector containing Hydrophobic matches in molA_patterns.patterMatches
                indx_molA = matchVectA.at(0).second;  //gets the index number of the atom in molA that we whant to check
                pos_a = conformer_molA.getAtomPos(indx_molA);
            for(const auto& matchVectB : tmpB->second){ //for every element of the vector containing Hydrophobic matches in molB_patterns.patternMatches
                indx_molB = matchVectB.at(0).second;
                pos_b = conformer_molB.getAtomPos(indx_molB);
                distance = calculateDistance(pos_a, pos_b);

                if (distance <= DISTANCE_HYDROPHOBIC){
                    getProtLigAtomID(molA, molB, indx_molA, indx_molB, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Hydrophobic\n";
                    output(molA.name, molB.name, atom_id_molA, "Hydrophobic", pos_a.x, pos_a.y, pos_a.z, atom_id_molB, "Hydrophobic", pos_b.x, pos_b.y, pos_b.z, "Hydrophobic", distance, protA_ligB);
                }
            }
        }
    }
}

/**
 * @brief Detects hydrogen bonds between two molecules.
 * 
 * The function searches for hydrogen donor-hydrogen pairs in one molecule
 * and hydrogen acceptors in the other. It calculates the distance between donor
 * and acceptor atoms, and the angle formed by donor-hydrogen-acceptor.
 * 
 * When a valid interaction is found, it is recorded to CSV via `output()`.
 * 
 * @param molA First molecule (protein or ligand)
 * @param molB Second molecule (ligand or protein)
 * @param molA_patterns SMARTS pattern matches in molA
 * @param molB_patterns SMARTS pattern matches in molB
 * @param conformer_molA Conformer of molA with 3D coordinates
 * @param conformer_molB Conformer of molB with 3D coordinates
 * @param protA_ligB True if molA is protein and molB is ligand, false if reversed
 */
void findHydrogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Hydrogen_donor_H);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Hydrogen_acceptor);
    float distance;
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

                if(distance <= DISTANCE_HYDROGENBOND && isAngleInRange(angle, MIN_ANGLE_HYDROGENBOND, MAX_ANGLE_HYDROGENBOND)){
                    getProtLigAtomID(molA, molB, id_hydrogen, id_acceptor, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Hydrogen bond\n";
                    output(molA.name, molB.name, atom_id_molA, "Hydrogen donor", pos_hydrogen.x, pos_hydrogen.y, pos_hydrogen.z, atom_id_molB, "Hydrogen acceptor", pos_acceptor.x, pos_acceptor.y, pos_acceptor.z, "Hydrogen Bond", distance, protA_ligB);
                }
            }
        }
    }
}

/**
 * @brief Detects halogen bonds between two molecules.
 * 
 * The function looks for halogen-donor/halogen atoms in one molecule
 * and halogen-acceptor/any atoms in the other. It computes:
 * - The distance between donor and acceptor
 * - The angle between donor-halogen-acceptor
 * - The angle between halogen-acceptor-any
 * 
 * Matching interactions are printed and written to CSV.
 * 
 * @param molA First molecule (protein or ligand)
 * @param molB Second molecule (ligand or protein)
 * @param molA_patterns SMARTS pattern matches in molA
 * @param molB_patterns SMARTS pattern matches in molB
 * @param conformer_molA 3D conformer of molA
 * @param conformer_molB 3D conformer of molB
 * @param protA_ligB True if molA is protein and molB is ligand, false if reversed
 */
void findHalogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Halogen_donor_halogen);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Halogen_acceptor_any);
    float distance;

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

                if(distance <= DISTANCE_HALOGENBOND && isAngleInRange(firstAngle, MIN_ANGLE1_HALOGENBOND, MAX_ANGLE1_HALOGENBOND) && isAngleInRange(secondAngle, MIN_ANGLE2_HALOGENBOND, MAX_ANGLE2_HALOGENBOND)){
                    getProtLigAtomID(molA, molB, id_halogen, id_acceptor, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Halogen bond\n";
                    output(molA.name, molB.name, atom_id_molA, "Halogen donor", pos_halogen.x, pos_halogen.y, pos_halogen.z, atom_id_molB, "Halogen acceptor", pos_acceptor.x, pos_acceptor.y, pos_acceptor.z, "Halogen Bond", distance, protA_ligB);
                }
        
            }
        }
    }
}

/**
 * @brief Detects ionic interactions between two molecules.
 * 
 * This function checks for:
 * - Cation–anion interactions: direct distance between charged atoms.
 * - Cation–aromatic ring interactions: geometric validation involving the ring centroid and normal.
 * 
 * If the interaction is detected, it is printed and recorded via the `output()` function.
 * 
 * @param molA First molecule (protein or ligand)
 * @param molB Second molecule (ligand or protein)
 * @param molA_patterns SMARTS pattern matches in molA
 * @param molB_patterns SMARTS pattern matches in molB
 * @param conformer_molA 3D conformer of molA
 * @param conformer_molB 3D conformer of molB
 * @param protA_ligB True if molA is protein and molB is ligand, false otherwise
 */
void findIonicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Cation);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Anion);
    unsigned int indx_molA;
    unsigned int indx_molB;
    RDGeom::Point3D pos_a, pos_b;
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

                if (distance <= DISTANCE_IONIC){
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

                if (distance <= DISTANCE_IONIC){
                    normal = calculateNormalVector(pos_points_ring.at(0), pos_points_ring.at(1), pos_points_ring.at(2));    //finds the normal vector to the plane defined by the aromatic ring atoms
                    pos_c = normal + centroid; // it' a point on the line normal to the ring and passing throw the centroid
                    angle = calculateAngle(centroid, pos_c, pos_a); // calculates the angle that must be <30 for the Ionic bond requirements
                    if((!isAngleInRange(angle, MIN_ANGLE_IONIC, MAX_ANGLE_IONIC)) || angle == MIN_ANGLE_IONIC || angle == MAX_ANGLE_IONIC){  //pos_c and pos_a can be on different sides of the aromatic ring plane
                        getProtLigAtomID(molA, molB, indx_molA, indx_molB, atom_id_molA, atom_id_molB, protA_ligB);
                        std::cout << "Ionic\n";
                        output(molA.name, molB.name, atom_id_molA, "Cation", pos_a.x, pos_a.y, pos_a.z, atom_id_molB, "Aromatic_ring", centroid.x, centroid.y, centroid.z, "Ionic", distance, protA_ligB);  // For aromatic ring the name of the last atom in the vector conteining pair <atom of the pattern, atom of the molecule> and the position of the centroid are printed.
                    }
                }
            }
        }
    }
}

// two planes facing each other: SANDWICH | two planes perpendicular: T-SHAPE

/**
 * @brief Detects π-stacking interactions between two molecules (CPU).
 *
 * Finds aromatic rings in both molecules, builds centroids and plane normals (from 3 adjacent atoms),
 * and checks geometry to detect:
 * - **SANDWICH**: nearly parallel planes within distance/angle thresholds
 * - **T-SHAPE**: nearly perpendicular planes plus inside-polygon check (ray casting after projection)
 *
 * Angles use |cos| with clamping to map into [0°,90°], matching the GPU semantics.
 * On detection, the interaction is printed and recorded via `output()`.
 *
 * @param molA First molecule (protein or ligand)
 * @param molB Second molecule (ligand or protein)
 * @param molA_patterns SMARTS pattern matches in molA
 * @param molB_patterns SMARTS pattern matches in molB
 * @param conformer_molA 3D conformer of molA
 * @param conformer_molB 3D conformer of molB
 * @param protA_ligB True if molA is protein and molB is ligand, false otherwise
 */

void findPiStacking(const Molecule& molA, const Molecule& molB,
                    const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns,
                    const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB,
                    const bool protA_ligB)
{
    auto itA = molA_patterns.patternMatches.find(Pattern::Aromatic_ring);
    auto itB = molB_patterns.patternMatches.find(Pattern::Aromatic_ring);
    if (itA == molA_patterns.patternMatches.end() || itB == molB_patterns.patternMatches.end()) return;

    // Prepara anelli A
    std::vector<std::vector<RDGeom::Point3D>> ringsA_points;
    std::vector<RDGeom::Point3D> centroidsA, normalsA;
    std::vector<unsigned int> ringA_id;
    ringsA_points.reserve(itA->second.size());
    for (const auto& matchA : itA->second) {
        std::vector<RDGeom::Point3D> pts; pts.reserve(matchA.size());
        for (const auto& p : matchA) pts.push_back(conformer_molA.getAtomPos(p.second));
        ringsA_points.push_back(pts);
        centroidsA.push_back(calculateCentroid(ringsA_points.back()));
        RDGeom::Point3D nA = calculateNormalVector(ringsA_points.back().at(0),
                                                   ringsA_points.back().at(1),
                                                   ringsA_points.back().at(2));
        normalsA.push_back(nA);
        ringA_id.push_back(matchA.back().second);
    }

    // Prepara anelli B
    std::vector<std::vector<RDGeom::Point3D>> ringsB_points;
    std::vector<RDGeom::Point3D> centroidsB, normalsB;
    std::vector<unsigned int> ringB_id;
    ringsB_points.reserve(itB->second.size());
    for (const auto& matchB : itB->second) {
        std::vector<RDGeom::Point3D> pts; pts.reserve(matchB.size());
        for (const auto& p : matchB) pts.push_back(conformer_molB.getAtomPos(p.second));
        ringsB_points.push_back(pts);
        centroidsB.push_back(calculateCentroid(ringsB_points.back()));
        RDGeom::Point3D nB = calculateNormalVector(ringsB_points.back().at(0),
                                                   ringsB_points.back().at(1),
                                                   ringsB_points.back().at(2));
        normalsB.push_back(nB);
        ringB_id.push_back(matchB.back().second);
    }

    if (ringsA_points.empty() || ringsB_points.empty()) return;

    std::string atom_id_molA, atom_id_molB;

    for (size_t i = 0; i < ringsA_points.size(); ++i) {
        const RDGeom::Point3D& cA = centroidsA[i];
        const RDGeom::Point3D& nA = normalsA[i];

        for (size_t j = 0; j < ringsB_points.size(); ++j) {
            const RDGeom::Point3D& cB = centroidsB[j];
            const RDGeom::Point3D& nB = normalsB[j];

            RDGeom::Point3D AB = cB - cA;
            RDGeom::Point3D BA = cA - cB;

            const float distance    = calculateDistance(cA, cB);
            const float planesAngle = calculateVectorAngle(nA, nB);
            const float angleA      = calculateVectorAngle(nA, AB);
            const float angleB      = calculateVectorAngle(nB, BA);

            // SANDWICH
            if (distance <= DISTANCE_SANDWICH &&
                isAngleInRange(planesAngle, MIN_PLANES_ANGLE_SANDWICH, MAX_PLANES_ANGLE_SANDWICH) &&
                isAngleInRange(angleA,      MIN_NORMAL_CENTROID_ANGLE_SANDWICH, MAX_NORMAL_CENTROID_ANGLE_SANDWICH) &&
                isAngleInRange(angleB,      MIN_NORMAL_CENTROID_ANGLE_SANDWICH, MAX_NORMAL_CENTROID_ANGLE_SANDWICH))
            {
                getProtLigAtomID(molA, molB, ringA_id[i], ringB_id[j], atom_id_molA, atom_id_molB, protA_ligB);
                std::cout << "Pi Stacking - SANDWICH\n";
                output(molA.name, molB.name, atom_id_molA, "Aromatic_ring", cA.x, cA.y, cA.z,
                       atom_id_molB, "Aromatic_ring", cB.x, cB.y, cB.z,
                       "Pi Stacking", distance, protA_ligB);
                continue;
            }

            // T-SHAPE
            if (distance <= DISTANCE_TSHAPE &&
                isAngleInRange(planesAngle, MIN_PLANES_ANGLE_TSHAPE, MAX_PLANES_ANGLE_TSHAPE) &&
                isAngleInRange(angleA,      MIN_NORMAL_CENTROID_ANGLE_TSHAPE, MAX_NORMAL_CENTROID_ANGLE_TSHAPE) &&
                isAngleInRange(angleB,      MIN_NORMAL_CENTROID_ANGLE_TSHAPE, MAX_NORMAL_CENTROID_ANGLE_TSHAPE))
            {
                const bool inside = centroidInsideRing(cB, ringsA_points[i], nA);
                if (inside) {
                    getProtLigAtomID(molA, molB, ringA_id[i], ringB_id[j], atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Pi Stacking - T-SHAPE\n";
                    output(molA.name, molB.name, atom_id_molA, "Aromatic_ring", cA.x, cA.y, cA.z,
                           atom_id_molB, "Aromatic_ring", cB.x, cB.y, cB.z,
                           "Pi Stacking", distance, protA_ligB);
                }
            }
        }
    }
}


/**
 * @brief Detects metal coordination interactions between two molecules.
 * 
 * The function searches for atoms labeled as "Metal" in one molecule and "Chelated" in the other,
 * based on SMARTS pattern matching. It calculates the Euclidean distance between pairs of atoms.
 * 
 * Valid interactions are recorded using the `output()` function.
 * 
 * @param molA First molecule (protein or ligand)
 * @param molB Second molecule (ligand or protein)
 * @param molA_patterns SMARTS pattern matches in molA
 * @param molB_patterns SMARTS pattern matches in molB
 * @param conformer_molA 3D conformer of molA
 * @param conformer_molB 3D conformer of molB
 * @param protA_ligB True if molA is protein and molB is ligand, false otherwise
 */
void findMetalCoordination(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Metal);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Chelated);

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())){
        RDGeom::Point3D pos_a, pos_b; 
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

                if (distance <= DISTANCE_METAL){
                    getProtLigAtomID(molA, molB, indx_molA, indx_molB, atom_id_molA, atom_id_molB, protA_ligB);
                    std::cout << "Metal\n";
                    output(molA.name, molB.name, atom_id_molA, "Metal", pos_a.x, pos_a.y, pos_a.z, atom_id_molB, "Chelated", pos_b.x, pos_b.y, pos_b.z, "Metal", distance, protA_ligB);
                }
            }
        }
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
void identifyInteractions(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns, const RDKit::Conformer& proteinConformer, const RDKit::Conformer& ligandConformer){
    // every function will need to serch all the interactions of that type and for every one found call the output function that adds them to the CSV file
    // considering some interactions can be formed both ways (cation-anion ; anion-cation) we call the find function two times  
    
    NVTX_PUSH("Hydrophobic Interaction");
    findHydrophobicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    NVTX_POP(); // Hydrophobic Interaction

    NVTX_PUSH("Hydrogen Bond");
    findHydrogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findHydrogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);
    NVTX_POP(); // Hydrogen Bond

    NVTX_PUSH("Halogen Bond");
    findHalogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findHalogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);
    NVTX_POP(); // Halogen Bond

    NVTX_PUSH("Ionic Interaction");
    findIonicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findIonicInteraction(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);
    NVTX_POP(); // Ionic Interaction

    NVTX_PUSH("Pi Stacking");
    findPiStacking(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    NVTX_POP(); // Pi Stacking

    NVTX_PUSH("Metal Coordination");
    findMetalCoordination(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true);
    findMetalCoordination(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false);
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
/** @} */

// ------------------------------------------------------- MAIN and INPUT ----------------------------------------------------------------------------------------

/**
 * @brief Performs a lightweight in-place sanitization using a subset of RDKit operations (FINDRADICALS, KEKULIZE, SETAROMATICITY, SETCONJUGATION, SETHYBRIDIZATION, SYMMRINGS).
 * @param rw Mutable molecule (`RWMol`) to sanitize in place.
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
        long fileSizeLong = ftell(file);
        if (fileSizeLong < 0) {
            fclose(file);
            std::cerr << "ftell() failed on " << argv[i] << "\n";
            continue;
        }
        fseek(file, 0, SEEK_SET);

        size_t fileSize = static_cast<size_t>(fileSizeLong);

        fileContent = static_cast<char*>(malloc(fileSize + 1));
        if (!fileContent) {
            std::cerr << "Malloc error\n";
            fclose(file);
            return;
        }

        size_t nread = fread(fileContent, 1, fileSize, file);
        if (nread != fileSize) {
            fclose(file);
            free(fileContent);
            throw std::runtime_error("Short read: expected " + std::to_string(fileSize) +
                                    " bytes, got " + std::to_string(nread));
        }

        fileContent[fileSize] = '\0';
        fclose(file);

        std::unique_ptr<RDKit::ROMol> mol;

        try {
            // Parse without sanitize e without removing H
            if (i == 1) {
                // PROTEINA (PDB)
                mol.reset(RDKit::PDBBlockToMol(fileContent, false, false)); // (content, sinitize, remove H)
            } else {
                // LIGANDO (MOL2)
                mol.reset(RDKit::Mol2BlockToMol(fileContent, false, false));
            }
            if (!mol) throw std::runtime_error("RDKit returned null molecule");

            // Passa a RWMol per le operazioni MolOps
            RDKit::RWMol rw(*mol);

            // sanitize più leggera (inizializza anche le RingInfo)
            try {
                sanitize_light_inplace(rw);
            } catch (const std::exception &se) {
                std::cerr << "Warning: sanitizeMol(light) failed for " << argv[i]
                          << " -> " << se.what() << "\n";
            }

            RDKit::MolOps::addHs(rw, false, true); // (rows, explicitOnly, addCoords)

            // Torna a ROMol per lo storage
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
int main(int argc, char *argv[]) {

    NVTX_PUSH("TotalProgram");

    NVTX_PUSH("Input");

    std::vector<Molecule> molVector; ///< Vector of all the molecules with their name, (the first element is always a protein, the other are ligands)
    FoundPatterns proteinPatterns;  ///< Declares a FoundPattern struct where to save all the pattern found in the protein
    FoundPatterns ligandPatterns;   ///< Declares a FoundPattern struct where to save all the pattern found in the ligand, the same will be used for all ligand passed in input.

    //the CSV file is created and inicialized with the HEADER line in the main
    initializeFile("interactions.csv");

    // Prints the files passed from line (argc, argv)
    if(argc >= 2){
        printf("Ci sono %d file passati:\n", argc - 1);
        std::cout << "1-" << "Protein: " << argv[1] << std::endl;
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

    for(int i = 1; i < argc - 1; i++){ // For every ligand
        NVTX_PUSH("IdentifyLigandSubstructs");
        identifySubstructs(molVector.at(i), ligandPatterns); // Identifies all the istances of patterns inside the ligand
        printFoundPatterns(ligandPatterns);
        NVTX_POP(); // IdentifyLigandSubstructs
        
        const RDKit::Conformer& ligandConformer = molVector.at(i).mol->getConformer();  
        
        NVTX_PUSH("IdentifyInteractions");
        identifyInteractions(molVector.at(0), molVector.at(i), proteinPatterns, ligandPatterns, proteinConformer, ligandConformer); //Identifies all the interactions between protein and ligand and adds the to the CSV file
        NVTX_POP(); // IdentifyInteractions

        ligandPatterns.patternMatches.clear();
    } 

    NVTX_POP(); // TotalProgram

    std::cout << "\nNuber of found interactions: " << g_interaction_count << '\n';

    return EXIT_SUCCESS;
}
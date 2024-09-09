    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
    #include "helpers.cuh"
    #include "main.hpp"

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

    // HYDROPHOBIC 
    #define DISTANCE_HYDROPHOBIC 4.5

    // HYDROGEN BOND
    #define DISTANCE_HYDROGENBOND 3.5
    #define MIN_ANGLE_HYDROGENBOND 130
    #define MAX_ANGLE_HYDROGENBOND 180

    // HALOGEN BOND
    #define DISTANCE_HALOGENBOND 3.5
    #define MIN_ANGLE1_HALOGENBOND 130
    #define MAX_ANGLE1_HALOGENBOND 180
    #define MIN_ANGLE2_HALOGENBOND 80
    #define MAX_ANGLE2_HALOGENBOND 140

    // IONIC
    #define DISTANCE_IONIC 4.5
    #define MIN_ANGLE_IONIC 30
    #define MAX_ANGLE_IONIC 150

    // PI STACKING - SANDWICH
    #define DISTANCE_SANDWICH 5.5
    #define MIN_PLANES_ANGLE_SANDWICH 0
    #define MAX_PLANES_ANGLE_SANDWICH 30
    #define MIN_NORMAL_CENTROID_ANGLE_SANDWICH 0
    #define MAX_NORMAL_CENTROID_ANGLE_SANDWICH 33

    // PI STACKING - T SHAPE
    #define DISTANCE_TSHAPE 6.5
    #define MIN_PLANES_ANGLE_TSHAPE 50
    #define MAX_PLANES_ANGLE_TSHAPE 90
    #define MIN_NORMAL_CENTROID_ANGLE_TSHAPE 0
    #define MAX_NORMAL_CENTROID_ANGLE_TSHAPE 30

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


    bool doSegmentsIntersect(RDGeom::Point3D &a1, RDGeom::Point3D &b1, RDGeom::Point3D &a2, RDGeom::Point3D &b2){ //checks if two COMPLANAR segments intersect
        RDGeom::Point3D a1a2 = a1 - a2;
        RDGeom::Point3D b1b2 = b1 - b2;
        RDGeom::Point3D a1b1 = a1 - b1;

        double a =  a1a2.x, b = -b1b2.x, c = a1a2.y, d = -b1b2.y, e = a1a2.z, f = -b1b2.z; //fill the coeficients in the matrix rapresenting the equations
        double det = a * d - b * c; //calculates the det of the matrix to check if there are solutions

        if(fabs(det) < 1e-10) return false; //checks if the det = 0 it means that there are no solutions the segments are or parallel or the same segment

        double t = (d * a1b1.x -b * a1b1.y) / det; //solves the parametric equation using Cramer's rule
        double s = (-c * a1b1.x + a * a1b1.y) / det;

        return (t >= 0 && t<= 1 && s >= 0 && s <= 1); //checks that the intersection point is within both segments
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

    // float calculateDistance(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b){  //calculates euclidian distance between 2 points located in a 3D space
    //     return (pos_a - pos_b).length();
    // }

    float calculateDistance(const RDGeom::Point3D &pos_a, const RDGeom::Point3D &pos_b){
        float x_diff = pos_a.x - pos_b.x; 
        float y_diff = pos_a.y - pos_b.y; 
        float z_diff = pos_a.z - pos_b.z;  

        return std::sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
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


    //Having three points located in a 3D space, imagine them forming a triangle: this function calculates the angle in degreeson of the vertex pos_a 
    float calculateAngle(RDGeom::Point3D &pos_a, RDGeom::Point3D &pos_b, RDGeom::Point3D &pos_c){
        float ab = calculateDistance(pos_a, pos_b);
        float bc = calculateDistance(pos_b, pos_c);
        float ac = calculateDistance(pos_a, pos_c);

        return (acos((pow(ab, 2) + pow(ac, 2) - pow(bc, 2)) / (2 * ab * ac))) * (180.0 / M_PI);
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


    //calculates the angle in degrees between two vectors (the smallest angle of the incidents infinite lines that are formed extending the vectors)
    float calculateVectorAngle(RDGeom::Point3D &vect_a, RDGeom::Point3D &vect_b){
        float dot = dotProduct(vect_a, vect_b);
        float norms = norm(vect_a) * norm(vect_b);
        float angle = std::acos(abs(dot / norms));
        return angle * 180 / M_PI; 
    }

    //TODO: questa si dovrà chiamare caluclateVectorAngle e per l'altra si trova un altro nome
    float calculateActualVectorAngle(RDGeom::Point3D &vect_a, RDGeom::Point3D &vect_b){ //calculates the angle in degrees between two vectors
        return std::acos(dotProduct(vect_a, vect_b) / ((norm(vect_a)) * (norm(vect_b))) * 180 / M_PI);
    }

    bool isGreaterThenNinety(float value){ //takes a value, returns true if its greater or equal to 90, false if not
        return value >= 90 ? true : false;
    }

    // ------------------------------------------------------- INTERACTIONS --------------------------------------------------------------------------




    // Dichiarazione della funzione wrapper definita in kernel.cu

    extern void launchDistanceKernel2D(float* d_posA_x, float* d_posA_y, float* d_posA_z,
                                float* d_posB_x, float* d_posB_y, float* d_posB_z,
                                float* d_distances, int numA, int numB, int blockSizeX, int blockSizeY);

    extern void launchHydrogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                                         float* d_hydrogen_x, float* d_hydrogen_y, float* d_hydrogen_z,
                                         float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                                         float* d_distances, float* d_angles,
                                         int numDonors, int numAcceptors, int blockSizeX, int blockSizeY);

    extern void launchHalogenBondKernel(float* d_donor_x, float* d_donor_y, float* d_donor_z,
                        float* d_halogen_x, float* d_halogen_y, float* d_halogen_z,
                        float* d_acceptor_x, float* d_acceptor_y, float* d_acceptor_z,
                        float* d_any_x, float* d_any_y, float* d_any_z,
                        float* d_distances, float* d_firstAngles, float* d_secondAngles,
                        int numDonors, int numAcceptors, int blockSizeX, int blockSizeY,
                        float maxDistance, float minAngle1, float maxAngle1,
                        float minAngle2, float maxAngle2);

    extern void launchIonicInteractionsKernel_CationAnion(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                          float* d_anion_x, float* d_anion_y, float* d_anion_z,
                                                          float* d_distances, int numCations, int numAnions, 
                                                          int blockSizeX, int blockSizeY, float maxDistance);

    extern void launchIonicInteractionsKernel_CationRing(float* d_cation_x, float* d_cation_y, float* d_cation_z,
                                                         float* d_ring_centroid_x, float* d_ring_centroid_y, float* d_ring_centroid_z,
                                                         float* d_ring_normal_x, float* d_ring_normal_y, float* d_ring_normal_z,
                                                         float* d_distances, float* d_angles, int numCations, int numRings, 
                                                         int blockSizeX, int blockSizeY, float maxDistance, float minAngle, float maxAngle);
   




void findHydrophobicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions){
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Hydrophobic);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Hydrophobic);

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())) {
        // Usa cudaMallocHost per allocare memoria pinned per i vettori di A e B
        float *posA_x, *posA_y, *posA_z;
        float *posB_x, *posB_y, *posB_z;
        float *distances_host;  // Output sulla CPU

        cudaMallocHost(&posA_x, tmpA->second.size() * sizeof(float));
        cudaMallocHost(&posA_y, tmpA->second.size() * sizeof(float));
        cudaMallocHost(&posA_z, tmpA->second.size() * sizeof(float));

        cudaMallocHost(&posB_x, tmpB->second.size() * sizeof(float));
        cudaMallocHost(&posB_y, tmpB->second.size() * sizeof(float));
        cudaMallocHost(&posB_z, tmpB->second.size() * sizeof(float));

        // Allocazione della memoria pinned per le distanze
        cudaMallocHost(&distances_host, tmpA->second.size() * tmpB->second.size() * sizeof(float));

        // Estrarre le posizioni atomiche da molA usando RDKit
        size_t idx = 0;
        for (const auto& matchVectA : tmpA->second) {
            unsigned int indx_molA = matchVectA.at(0).second;
            RDGeom::Point3D posA = conformer_molA.getAtomPos(indx_molA);
            posA_x[idx] = posA.x;
            posA_y[idx] = posA.y;
            posA_z[idx] = posA.z;
            idx++;
        }

        // Estrarre le posizioni atomiche da molB usando RDKit
        idx = 0;
        for (const auto& matchVectB : tmpB->second) {
            unsigned int indx_molB = matchVectB.at(0).second;
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
        const int num_streams = 200;
        cudaStream_t streams[num_streams];
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamCreate(&streams[i]);
        }

        // Trasferisci **B** solo una volta, al di fuori del ciclo
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

            // Dimensioni dei blocchi e griglie
            int blockSizeX = 16;
            int blockSizeY = 16;
            dim3 threadsPerBlock(blockSizeX, blockSizeY);
            dim3 blocksPerGrid((widthA + blockSizeX - 1) / blockSizeX, 
                               (tmpB->second.size() + blockSizeY - 1) / blockSizeY);

            // Lancia il kernel per ogni chunk di A contro l'intero set di B
            launchDistanceKernel2D(d_posA_x + lowerA, d_posA_y + lowerA, d_posA_z + lowerA,
                                   d_posB_x, d_posB_y, d_posB_z,
                                   d_distances + lowerA * tmpB->second.size(), widthA, tmpB->second.size(), blockSizeX, blockSizeY, streams[stream]);

            // Copia i risultati parziali dalla GPU alla CPU per il chunk di A
            cudaMemcpyAsync(distances_host + lowerA * tmpB->second.size(), d_distances + lowerA * tmpB->second.size(),
                            widthA * tmpB->second.size() * sizeof(float), cudaMemcpyDeviceToHost, streams[stream]);
        }

        // Sincronizza tutti gli stream
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }

        // Post-processamento sulla CPU per identificare interazioni idrofobiche
        for (size_t i = 0; i < tmpA->second.size(); ++i) {
            for (size_t j = 0; j < tmpB->second.size(); ++j) {
                float distance = distances_host[i * tmpB->second.size() + j];
                if (distance <= DISTANCE_HYDROPHOBIC) {
                    std::string atom_id_molA, atom_id_molB;
                    getProtLigAtomID(molA, molB, i, j, atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions)
                        std::cout << "Hydrophobic\n";
                    output(molA.name, molB.name, atom_id_molA, "Hydrophobic", posA_x[i], posA_y[i], posA_z[i],
                        atom_id_molB, "Hydrophobic", posB_x[j], posB_y[j], posB_z[j], "Hydrophobic", distance, protA_ligB);
                }
            }
        }

        // Pulizia della memoria GPU
        cudaFree(d_posA_x);
        cudaFree(d_posA_y);
        cudaFree(d_posA_z);
        cudaFree(d_posB_x);
        cudaFree(d_posB_y);
        cudaFree(d_posB_z);
        cudaFree(d_distances);

        // Pulizia della memoria host allocata con cudaMallocHost
        cudaFreeHost(posA_x);
        cudaFreeHost(posA_y);
        cudaFreeHost(posA_z);
        cudaFreeHost(posB_x);
        cudaFreeHost(posB_y);
        cudaFreeHost(posB_z);
        cudaFreeHost(distances_host);

        // Distruzione degli stream
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
    }
}


    void findHydrogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Hydrogen_donor_H);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Hydrogen_acceptor);
    float distance;

    if ((molA_pattern != molA_patterns.patternMatches.end()) && (molB_pattern != molB_patterns.patternMatches.end())) {  
        std::vector<float> donor_x, donor_y, donor_z;
        std::vector<float> hydrogen_x, hydrogen_y, hydrogen_z;
        std::vector<float> acceptor_x, acceptor_y, acceptor_z;

        // Estrazione delle coordinate da molA (donatore e idrogeno)
        for (const auto& matchVect_molA : molA_pattern->second) {
            int id_donor = matchVect_molA.at(0).second;
            int id_hydrogen = matchVect_molA.at(1).second;

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
            RDGeom::Point3D pos_acceptor = conformer_molB.getAtomPos(id_acceptor);

            acceptor_x.push_back(pos_acceptor.x);
            acceptor_y.push_back(pos_acceptor.y);
            acceptor_z.push_back(pos_acceptor.z);
        }

        // Allocazione della memoria sulla GPU per le coordinate e i risultati
        float *d_donor_x, *d_donor_y, *d_donor_z;
        float *d_hydrogen_x, *d_hydrogen_y, *d_hydrogen_z;
        float *d_acceptor_x, *d_acceptor_y, *d_acceptor_z;
        float *d_distances, *d_angles;

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
        cudaMalloc(&d_angles, donor_x.size() * acceptor_x.size() * sizeof(float));

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

        // Definizione delle dimensioni di blocchi e griglie per il kernel CUDA
        int blockSizeX = 16;
        int blockSizeY = 16;
        dim3 threadsPerBlock(blockSizeX, blockSizeY);
        dim3 blocksPerGrid((donor_x.size() + blockSizeX - 1) / blockSizeX,
                           (acceptor_x.size() + blockSizeY - 1) / blockSizeY);

        // Lancia il kernel CUDA per calcolare distanze e angoli
        launchHydrogenBondKernel(d_donor_x, d_donor_y, d_donor_z,
                         d_hydrogen_x, d_hydrogen_y, d_hydrogen_z,
                         d_acceptor_x, d_acceptor_y, d_acceptor_z,
                         d_distances, d_angles,
                         donor_x.size(), acceptor_x.size(), blockSizeX, blockSizeY);


        // Copia dei risultati dalla GPU alla CPU
        std::vector<float> distances(donor_x.size() * acceptor_x.size());
        std::vector<float> angles(donor_x.size() * acceptor_x.size());
        cudaMemcpy(distances.data(), d_distances, distances.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(angles.data(), d_angles, angles.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Post-processamento sulla CPU per trovare i legami a idrogeno
        for (size_t i = 0; i < donor_x.size(); ++i) {
            for (size_t j = 0; j < acceptor_x.size(); ++j) {
                float distance = distances[i * acceptor_x.size() + j];
                float angle = angles[i * acceptor_x.size() + j];

                if (distance <= DISTANCE_HYDROGENBOND && isAngleInRange(angle, MIN_ANGLE_HYDROGENBOND, MAX_ANGLE_HYDROGENBOND)) {
                    std::string atom_id_molA, atom_id_molB;
                    getProtLigAtomID(molA, molB, i, j, atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions)
                        std::cout << "Hydrogen bond\n";
                    output(molA.name, molB.name, atom_id_molA, "Hydrogen donor", hydrogen_x[i], hydrogen_y[i], hydrogen_z[i],
                           atom_id_molB, "Hydrogen acceptor", acceptor_x[j], acceptor_y[j], acceptor_z[j], "Hydrogen Bond", distance, protA_ligB);
                }
            }
        }

        // Pulizia della memoria GPU
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
        cudaFree(d_angles);
    }
}


    void findHalogenBond(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
    auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Halogen_donor_halogen);
    auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Halogen_acceptor_any);
    
    float maxDistance = DISTANCE_HALOGENBOND;
    float minAngle1 = MIN_ANGLE1_HALOGENBOND, maxAngle1 = MAX_ANGLE1_HALOGENBOND;
    float minAngle2 = MIN_ANGLE2_HALOGENBOND, maxAngle2 = MAX_ANGLE2_HALOGENBOND;

    if (molA_pattern != molA_patterns.patternMatches.end() && molB_pattern != molB_patterns.patternMatches.end()) {
        std::vector<float> donor_x, donor_y, donor_z;
        std::vector<float> halogen_x, halogen_y, halogen_z;
        std::vector<float> acceptor_x, acceptor_y, acceptor_z;
        std::vector<float> any_x, any_y, any_z;

        // Estrazione delle coordinate da molA (donatori e alogeni)
        for (const auto& matchVectA : molA_pattern->second) {
            int id_donor = matchVectA.at(0).second;
            int id_halogen = matchVectA.at(1).second;

            RDGeom::Point3D pos_donor = conformer_molA.getAtomPos(id_donor);
            RDGeom::Point3D pos_halogen = conformer_molA.getAtomPos(id_halogen);

            donor_x.push_back(pos_donor.x);
            donor_y.push_back(pos_donor.y);
            donor_z.push_back(pos_donor.z);

            halogen_x.push_back(pos_halogen.x);
            halogen_y.push_back(pos_halogen.y);
            halogen_z.push_back(pos_halogen.z);
        }

        // Estrazione delle coordinate da molB (accettori e atomi generici)
        for (const auto& matchVectB : molB_pattern->second) {
            int id_acceptor = matchVectB.at(0).second;
            int id_any = matchVectB.at(1).second;

            RDGeom::Point3D pos_acceptor = conformer_molB.getAtomPos(id_acceptor);
            RDGeom::Point3D pos_any = conformer_molB.getAtomPos(id_any);

            acceptor_x.push_back(pos_acceptor.x);
            acceptor_y.push_back(pos_acceptor.y);
            acceptor_z.push_back(pos_acceptor.z);

            any_x.push_back(pos_any.x);
            any_y.push_back(pos_any.y);
            any_z.push_back(pos_any.z);
        }

        // Numero di donatori e accettori
        int numDonors = donor_x.size();
        int numAcceptors = acceptor_x.size();

        // Allocazione memoria GPU
        float *d_donor_x, *d_donor_y, *d_donor_z;
        float *d_halogen_x, *d_halogen_y, *d_halogen_z;
        float *d_acceptor_x, *d_acceptor_y, *d_acceptor_z;
        float *d_any_x, *d_any_y, *d_any_z;
        float *d_distances, *d_firstAngles, *d_secondAngles;

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
        cudaMalloc(&d_firstAngles, numDonors * numAcceptors * sizeof(float));
        cudaMalloc(&d_secondAngles, numDonors * numAcceptors * sizeof(float));

        // Copia i dati sulla GPU
        cudaMemcpy(d_donor_x, donor_x.data(), numDonors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_donor_y, donor_y.data(), numDonors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_donor_z, donor_z.data(), numDonors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_halogen_x, halogen_x.data(), numDonors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_halogen_y, halogen_y.data(), numDonors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_halogen_z, halogen_z.data(), numDonors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acceptor_x, acceptor_x.data(), numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acceptor_y, acceptor_y.data(), numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_acceptor_z, acceptor_z.data(), numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_any_x, any_x.data(), numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_any_y, any_y.data(), numAcceptors * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_any_z, any_z.data(), numAcceptors * sizeof(float), cudaMemcpyHostToDevice);

        // Lancia il kernel CUDA per calcolare distanze e angoli
        int blockSizeX = 16;
        int blockSizeY = 16;
        launchHalogenBondKernel(d_donor_x, d_donor_y, d_donor_z,
                                d_halogen_x, d_halogen_y, d_halogen_z,
                                d_acceptor_x, d_acceptor_y, d_acceptor_z,
                                d_any_x, d_any_y, d_any_z,
                                d_distances, d_firstAngles, d_secondAngles,
                                numDonors, numAcceptors, blockSizeX, blockSizeY,
                                maxDistance, minAngle1, maxAngle1, minAngle2, maxAngle2);

        // Copia dei risultati dalla GPU alla CPU
        std::vector<float> distances(numDonors * numAcceptors);
        std::vector<float> firstAngles(numDonors * numAcceptors);
        std::vector<float> secondAngles(numDonors * numAcceptors);
        cudaMemcpy(distances.data(), d_distances, numDonors * numAcceptors * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(firstAngles.data(), d_firstAngles, numDonors * numAcceptors * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(secondAngles.data(), d_secondAngles, numDonors * numAcceptors * sizeof(float), cudaMemcpyDeviceToHost);

        // Post-processamento sulla CPU per verificare le interazioni e stampare i risultati
        for (int i = 0; i < numDonors; ++i) {
            for (int j = 0; j < numAcceptors; ++j) {
                if (distances[i * numAcceptors + j] > 0) {  // Solo interazioni valide (distanze positive)
                    std::string atom_id_molA, atom_id_molB;
                    getProtLigAtomID(molA, molB, i, j, atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions)
                        std::cout << "Halogen bond\n";
                    output(molA.name, molB.name, atom_id_molA, "Halogen donor", halogen_x[i], halogen_y[i], halogen_z[i],
                           atom_id_molB, "Halogen acceptor", acceptor_x[j], acceptor_y[j], acceptor_z[j],
                           "Halogen Bond", distances[i * numAcceptors + j], protA_ligB);
                }
            }
        }

        // Pulizia della memoria GPU
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
        cudaFree(d_firstAngles);
        cudaFree(d_secondAngles);
    }
}


    void findIonicInteraction(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Cation);
    auto tmpB_anion = molB_patterns.patternMatches.find(Pattern::Anion);
    auto tmpB_ring = molB_patterns.patternMatches.find(Pattern::Aromatic_ring);

    std::vector<float> cation_x, cation_y, cation_z;
    std::vector<float> anion_x, anion_y, anion_z;
    std::vector<float> ring_centroid_x, ring_centroid_y, ring_centroid_z;
    std::vector<float> ring_normal_x, ring_normal_y, ring_normal_z;

    // Estrai le coordinate dei cationi
    if (tmpA != molA_patterns.patternMatches.end()) {
        for (const auto& matchVectA : tmpA->second) {
            int indx_molA = matchVectA.at(0).second;
            RDGeom::Point3D pos_a = conformer_molA.getAtomPos(indx_molA);
            cation_x.push_back(pos_a.x);
            cation_y.push_back(pos_a.y);
            cation_z.push_back(pos_a.z);
        }
    }

    // Estrai le coordinate degli anioni
    if (tmpB_anion != molB_patterns.patternMatches.end()) {
        for (const auto& matchVectB : tmpB_anion->second) {
            int indx_molB = matchVectB.at(0).second;
            RDGeom::Point3D pos_b = conformer_molB.getAtomPos(indx_molB);
            anion_x.push_back(pos_b.x);
            anion_y.push_back(pos_b.y);
            anion_z.push_back(pos_b.z);
        }
    }

    // Estrai le coordinate per i centri degli anelli aromatici e i loro vettori normali
    if (tmpB_ring != molB_patterns.patternMatches.end()) {
        for (const auto& matchVectB : tmpB_ring->second) {
            std::vector<RDGeom::Point3D> pos_points_ring;
            for (const auto& pairs_molB : matchVectB) {
                int indx_molB = pairs_molB.second;
                RDGeom::Point3D pos_b = conformer_molB.getAtomPos(indx_molB);
                pos_points_ring.push_back(pos_b);
            }

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

    // Numero di cationi, anioni e anelli aromatici
    int numCations = cation_x.size();
    int numAnions = anion_x.size();
    int numRings = ring_centroid_x.size();

    // Allocazione memoria GPU
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

    // Lancia il kernel per il calcolo delle distanze tra cationi e anioni
    int blockSizeX = 16;
    int blockSizeY = 16;
    float maxDistance = 4.0f;

    launchIonicInteractionsKernel_CationAnion(d_cation_x, d_cation_y, d_cation_z,
                                          d_anion_x, d_anion_y, d_anion_z,
                                          d_distances_anion, numCations, numAnions,
                                          blockSizeX, blockSizeY, maxDistance);


    // Lancia il kernel per il calcolo delle distanze tra cationi e anelli aromatici
    launchIonicInteractionsKernel_CationRing(d_cation_x, d_cation_y, d_cation_z,
                                         d_ring_centroid_x, d_ring_centroid_y, d_ring_centroid_z,
                                         d_ring_normal_x, d_ring_normal_y, d_ring_normal_z,
                                         d_distances_ring, d_angles_ring, numCations, numRings,
                                         blockSizeX, blockSizeY, maxDistance, 30.0f, 150.0f);


    // Copia dei risultati dalla GPU alla CPU
    std::vector<float> distances_anion(numCations * numAnions);
    std::vector<float> distances_ring(numCations * numRings);
    std::vector<float> angles_ring(numCations * numRings);
    cudaMemcpy(distances_anion.data(), d_distances_anion, numCations * numAnions * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(distances_ring.data(), d_distances_ring, numCations * numRings * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(angles_ring.data(), d_angles_ring, numCations * numRings * sizeof(float), cudaMemcpyDeviceToHost);

    // Post-processamento per cationi-anioni
    for (int i = 0; i < numCations; ++i) {
        for (int j = 0; j < numAnions; ++j) {
            float distance = distances_anion[i * numAnions + j];
            if (distance > 0) {
                std::string atom_id_molA, atom_id_molB;
                getProtLigAtomID(molA, molB, i, j, atom_id_molA, atom_id_molB, protA_ligB);
                if (printInteractions)
                    std::cout << "Ionic\n";
                output(molA.name, molB.name, atom_id_molA, "Cation", cation_x[i], cation_y[i], cation_z[i],
                       atom_id_molB, "Anion", anion_x[j], anion_y[j], anion_z[j], "Ionic", distance, protA_ligB);
            }
        }
    }

    // Post-processamento per cationi-anelli aromatici
    for (int i = 0; i < numCations; ++i) {
        for (int j = 0; j < numRings; ++j) {
            float distance = distances_ring[i * numRings + j];
            float angle = angles_ring[i * numRings + j];
            if (distance > 0 && angle > 0) {
                std::string atom_id_molA, atom_id_molB;
                getProtLigAtomID(molA, molB, i, j, atom_id_molA, atom_id_molB, protA_ligB);
                if (printInteractions)
                    std::cout << "Ionic with aromatic ring\n";
                output(molA.name, molB.name, atom_id_molA, "Cation", cation_x[i], cation_y[i], cation_z[i],
                       atom_id_molB, "Aromatic_ring", ring_centroid_x[j], ring_centroid_y[j], ring_centroid_z[j], "Ionic", distance, protA_ligB);
            }
        }
    }

    // Pulizia della memoria GPU
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


    //two planes facing each other: SANDWICH | two planes perpendicular: T-SHAPE
    void findPiStacking(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions){
        auto molA_pattern = molA_patterns.patternMatches.find(Pattern::Aromatic_ring);
        auto molB_pattern = molB_patterns.patternMatches.find(Pattern::Aromatic_ring);
        unsigned int id_pointA, id_pointB;
        RDGeom::Point3D pos_pointA, pos_pointB;
        float distRequired;
        float distance;

        if ((molA_pattern != molA_patterns.patternMatches.end()) && (molB_pattern != molB_patterns.patternMatches.end())){
            float planesAngle;

            float normalCentroidAngle_A, normalCentroidAngle_B;

            std::string atom_id_molA, atom_id_molB;

            RDGeom::Point3D centroidA, centroidB, normalA, normalB, centroidsVector;
            
            std::vector<RDGeom::Point3D> pos_ringA, pos_ringB;

            for (const auto& matchVect_molA : molA_pattern->second){ // for each aromatic ring found in molA
                pos_ringA.clear();
                for(const auto& pair_molA : matchVect_molA){ // creates the aromatic ring A as a vector of points
                    id_pointA = pair_molA.second; 
                    pos_pointA = conformer_molA.getAtomPos(id_pointA);
                    pos_ringA.push_back(pos_pointA);  
                }
                centroidA = calculateCentroid(pos_ringA);

                for(const auto& matchVect_molB : molB_pattern->second){ // for each aromatic ring found in molB
                    pos_ringB.clear();
                    for(const auto& pair_molB : matchVect_molB){ // creates the aromatic ring B as a vector of points
                        id_pointB = pair_molB.second;  
                        pos_pointB = conformer_molB.getAtomPos(id_pointB);
                        pos_ringB.push_back(pos_pointB);  
                    }
                    centroidB = calculateCentroid(pos_ringB); //TODO: controllare se conviene spostare il calcolo dei centroidi e della distanza dentro agli if

                    distance = calculateDistance(centroidA, centroidB); // gets the distance between the two centroids 

                    normalA = calculateNormalVector(pos_ringA.at(0), pos_ringA.at(2), pos_ringA.at(3)); // finds the normal vector of the plane of the aromatic ring A
                    normalB = calculateNormalVector(pos_ringB.at(0), pos_ringB.at(2), pos_ringB.at(3)); // finds the normal vector of the plane of the aromatic ring B

                    planesAngle = calculateVectorAngle(normalA, normalB); // finds the angle between the two aromatic rings

                    if(isAngleInRange(planesAngle, MIN_PLANES_ANGLE_SANDWICH, MAX_PLANES_ANGLE_SANDWICH)){ // SANDWICH

                        centroidsVector = centroidB - centroidA; // calculates the vector that links the two centroids

                        normalCentroidAngle_A = calculateVectorAngle(centroidsVector, normalA); //calculate the angle between the vector that links the two centroids and the normal of ring A
                        normalCentroidAngle_B = calculateVectorAngle(centroidsVector, normalB); //calculate the angle between the vector that links the two centroids and the normal of ring B

                        if(distance <= DISTANCE_SANDWICH && isAngleInRange(normalCentroidAngle_A, MIN_NORMAL_CENTROID_ANGLE_SANDWICH, MAX_NORMAL_CENTROID_ANGLE_SANDWICH) && isAngleInRange(normalCentroidAngle_B, MIN_NORMAL_CENTROID_ANGLE_SANDWICH, MAX_NORMAL_CENTROID_ANGLE_SANDWICH)){
                            getProtLigAtomID(molA, molB, id_pointA, id_pointB, atom_id_molA, atom_id_molB, protA_ligB);
                            if(printInteractions)
                                std::cout << "Pi Stacking - SANDWICH \n";
                            output(molA.name, molB.name, atom_id_molA, "Aromatic_ring", centroidA.x, centroidA.y, centroidA.z,  atom_id_molB, "Aromatic_ring", centroidB.x, centroidB.y, centroidB.z, "Pi Stacking", distance, protA_ligB);
                        }
                    }
                    else if(isAngleInRange(planesAngle, MIN_PLANES_ANGLE_TSHAPE, MAX_PLANES_ANGLE_TSHAPE)){ // T SHAPE

                        centroidsVector = centroidB - centroidA; //calculates the vector that links the two centroids

                        normalCentroidAngle_A = calculateVectorAngle(centroidsVector, normalA); //calculate the angle between the vector that links the two centroids and the normal of ring A
                        normalCentroidAngle_B = calculateVectorAngle(centroidsVector, normalB); //calculate the angle between the vector that links the two centroids and the normal of ring B

                        //TODO: manca il check del quarto punto della docu

                        
                        RDGeom::Point3D P1 = centroidB + normalA * calculateDistance(pos_ringA.at(1), pos_ringA.at(2), pos_ringA.at(3), centroidB) * (isGreaterThenNinety(calculateActualVectorAngle(centroidsVector, normalA)) ? 1 : -1); //finds the point P1

                        int count = 0;

                        for(int k = 0; k < pos_ringA.size(); k++){ //checks if the segment P1-centroidA intersects with every segment of ringA
                            if(doSegmentsIntersect(P1, centroidA, pos_ringA.at(k), pos_ringA.at((k+1)%pos_ringA.size()))) count ++; //counts the number of intersections
                        }

                        if(distance <= DISTANCE_TSHAPE && isAngleInRange(normalCentroidAngle_A, MIN_NORMAL_CENTROID_ANGLE_TSHAPE, MAX_NORMAL_CENTROID_ANGLE_TSHAPE) && isAngleInRange(normalCentroidAngle_B, MIN_NORMAL_CENTROID_ANGLE_TSHAPE, MAX_NORMAL_CENTROID_ANGLE_TSHAPE) && count < 1){
                            getProtLigAtomID(molA, molB, id_pointA, id_pointB, atom_id_molA, atom_id_molB, protA_ligB);
                            if(printInteractions)
                                std::cout << "Pi Stacking - T-SHAPE \n";
                            output(molA.name, molB.name, atom_id_molA, "Aromatic_ring", centroidA.x, centroidA.y, centroidA.z,  atom_id_molB, "Aromatic_ring", centroidB.x, centroidB.y, centroidB.z, "Pi Stacking", distance, protA_ligB);
                        }
                    }
                }
            }
        }
    }

 void findMetalCoordination(const Molecule& molA, const Molecule& molB, const FoundPatterns& molA_patterns, const FoundPatterns& molB_patterns, const RDKit::Conformer& conformer_molA, const RDKit::Conformer& conformer_molB, const bool protA_ligB, const bool printInteractions) {
    auto tmpA = molA_patterns.patternMatches.find(Pattern::Metal);
    auto tmpB = molB_patterns.patternMatches.find(Pattern::Chelated);

    if ((tmpA != molA_patterns.patternMatches.end()) && (tmpB != molB_patterns.patternMatches.end())) {
        std::vector<float> metal_x, metal_y, metal_z;
        std::vector<float> chelated_x, chelated_y, chelated_z;

        // Estrai le coordinate per i metalli
        for (const auto& matchVectA : tmpA->second) {
            unsigned int indx_molA = matchVectA.at(0).second;
            RDGeom::Point3D pos_a = conformer_molA.getAtomPos(indx_molA);
            metal_x.push_back(pos_a.x);
            metal_y.push_back(pos_a.y);
            metal_z.push_back(pos_a.z);
        }

        // Estrai le coordinate per i chelati
        for (const auto& matchVectB : tmpB->second) {
            unsigned int indx_molB = matchVectB.at(0).second;
            RDGeom::Point3D pos_b = conformer_molB.getAtomPos(indx_molB);
            chelated_x.push_back(pos_b.x);
            chelated_y.push_back(pos_b.y);
            chelated_z.push_back(pos_b.z);
        }

        // Numero di metalli e chelati
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

        // Copia dei dati sulla GPU
        cudaMemcpy(d_metal_x, metal_x.data(), numMetals * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_metal_y, metal_y.data(), numMetals * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_metal_z, metal_z.data(), numMetals * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chelated_x, chelated_x.data(), numChelated * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chelated_y, chelated_y.data(), numChelated * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chelated_z, chelated_z.data(), numChelated * sizeof(float), cudaMemcpyHostToDevice);

        // Lancia il kernel CUDA per calcolare le distanze
        int blockSizeX = 16;
        int blockSizeY = 16;
        float distRequired = 2.8f;
        launchDistanceKernel2D(d_metal_x, d_metal_y, d_metal_z,
                                      d_chelated_x, d_chelated_y, d_chelated_z,
                                      d_distances, numMetals, numChelated, blockSizeX, blockSizeY, 0); //stream momentaneamente a 0

        // Copia dei risultati dalla GPU alla CPU
        std::vector<float> distances(numMetals * numChelated);
        cudaMemcpy(distances.data(), d_distances, numMetals * numChelated * sizeof(float), cudaMemcpyDeviceToHost);

        // Post-processamento sulla CPU per controllare le distanze e stampare i risultati
        for (int i = 0; i < numMetals; ++i) {
            for (int j = 0; j < numChelated; ++j) {
                float distance = distances[i * numChelated + j];
                if (distance <= distRequired) { 
                    std::string atom_id_molA, atom_id_molB;
                    getProtLigAtomID(molA, molB, i, j, atom_id_molA, atom_id_molB, protA_ligB);
                    if (printInteractions)
                        std::cout << "Metal\n";
                    output(molA.name, molB.name, atom_id_molA, "Metal", metal_x[i], metal_y[i], metal_z[i],
                           atom_id_molB, "Chelated", chelated_x[j], chelated_y[j], chelated_z[j],
                           "Metal", distance, protA_ligB);
                }
            }
        }

        // Pulizia della memoria GPU
        cudaFree(d_metal_x);
        cudaFree(d_metal_y);
        cudaFree(d_metal_z);
        cudaFree(d_chelated_x);
        cudaFree(d_chelated_y);
        cudaFree(d_chelated_z);
        cudaFree(d_distances);
    }
}

    void identifyInteractions(const Molecule& protein, const Molecule& ligand, const FoundPatterns& proteinPatterns, const FoundPatterns& ligandPatterns, const RDKit::Conformer& proteinConformer, const RDKit::Conformer& ligandConformer, const bool printInteractions){
        // every function will need to serch all the interactions of that type and for every one found call the output function that adds them to the CSV file
        // considering some interactions can be formed both ways (cation-anion ; anion-cation) we call the find function two times  
        
        findHydrophobicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);

        findHydrogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
        findHydrogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);

        findHalogenBond(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
        findHalogenBond(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);

        findIonicInteraction(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
        findIonicInteraction(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);

        findPiStacking(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);

        findMetalCoordination(protein, ligand, proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true, printInteractions);
        findMetalCoordination(ligand, protein, ligandPatterns, proteinPatterns, ligandConformer, proteinConformer, false, printInteractions);
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

        CPUTimer cpu_timer;
        CPUTimer overall_cpu_timer;

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
            printf("There are %d files passed as args:\n", argc - 1);
            std::cout << "1-Protein: " << argv[1] << std::endl;
            for(int i = 2; i < argc; i++) {
                std::cout << i << "-Ligand: " << argv[i] << std::endl;
            }
        }

        overall_cpu_timer.start();
        cpu_timer.start();
        input(argv, argc, molVector);
        cpu_timer.stop("Input function");

        cpu_timer.start();
        identifySubstructs(molVector.at(0), proteinPatterns); // Identifies all the istances of patterns inside the protein
        // printFoundPatterns(proteinPatterns);
        cpu_timer.stop("IdentifySubstructs of protein");
        
        //cpu_timer.start();
        const RDKit::Conformer& proteinConformer = molVector.at(0).mol->getConformer(); //Conformer is a class that represents the 2D or 3D conformation of a molecule
        //cpu_timer.stop("Get conformer of protein");

        for(int i = 1; i < argc - 1; i++){ // For every ligand
            cpu_timer.start();
            identifySubstructs(molVector.at(i), ligandPatterns); // Identifies all the istances of patterns inside the ligand
            // printFoundPatterns(ligandPatterns);
            cpu_timer.stop("Identify Substruct of ligand #" + std::to_string(i));
            
            //cpu_timer.start();
            const RDKit::Conformer& ligandConformer = molVector.at(i).mol->getConformer();  
            //cpu_timer.stop("Get conformer of ligand #" + std::to_string(i));

            cpu_timer.start();    
            identifyInteractions(molVector.at(0), molVector.at(i), proteinPatterns, ligandPatterns, proteinConformer, ligandConformer, true); //Identifies all the interactions between protein and ligand and adds the to the CSV file
            cpu_timer.stop("Find interactions #" + std::to_string(i));

            ligandPatterns.patternMatches.clear();
        }

        overall_cpu_timer.stop("Overall time spent on the CPU for main operations");

        return EXIT_SUCCESS;
    }

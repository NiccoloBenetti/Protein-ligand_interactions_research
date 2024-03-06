## Data structure
- Basic data structure to describe a molecule: RDKit::RWMol or RDKit::ROMol

## Parsing
- Parse a string that describe the protein (in PDB format): RDKit::PDBBlockToMol
- Parse a string that describe the ligand (in Mol2 format): RDKit::Mol2BlockToMol

> NOTE: we need the hydrogen when we parse the molecule, i.e. remove_hs = false

## Match a smiles

Input: a SMILES pattern that represent the "query" object, and the target molecule

1. Get a conformation of the target molecule, i.e. the coordinates: methord "getConformer()" of an RDKit Mol
2. Build a RDKit Mol from the SMARTS pattern: RDKit::SmartsToMol
3. Match a molecule inside another: RDKit::SubstructMatch . The output is a sequence of matches

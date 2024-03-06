# Rules for intermolecular molecular interactions

This is a compact representation of the intermolecular interactions as modeled by prolif.
We updated the information written in the related paper [1], with the changes introduced in the code (version 1.1.0).

## Notations
In this document we denote a single atom with square brackets, e.g. with `[Hydrophobic]` it means an atom wich is hydrophobic.
This atom typing will be defined using the [SMARTS pattern](#SMARTS-pattern).
The `-` character represents intramolecular interaction, while `...` represents intermolecular interactions.

Usually an intermolecular interaction involve one atom for each molecule.
In this case the SMARTS pattern will describe a single atom.
When the interaction involves more than one atom of the molecule, the SMARTS pattern include more atoms of the same molecule.
In this way, for all the interactions, we always use one pattern for each molecule.


## Known interactions

This is the list of all the intermolecular interactions that we model.
Besides the SMARTS pattern, more conditions apply.
The idea of this section is to state all of them.

### Hydrophobic interaction

This interaction occurs between two hydrophobic atoms,i.e. `[hydrophobic] ... [hydrophobic]`.
The following are the interaction requirements:
1. atom-atom distance must be lower or equal than 4.5 Angstrom


### Hydrogen bond

This interaction occurs between an hydrogen donor and an hydrogen acceptor, where the hydrogen is shared, i.e. `[hydrogen_donor]-[hydrogen] ... [hydrogen_acceptor]`.
The following are the interaction requirements:
1. the atom-atom distance between donor and acceptor must be lower or equal than 3.5 Angstrom
2. the angle between the donor-hydrogen atoms and the hydrogen-acceptor atoms must be between 130 and 180 degree.

### Halogen bond

This interaction is similar to the hydrogen bond, with the difference that the shared atom is an halogen, i.e. `[halogen_donor]-[halogen] ... [halogen_acceptor]-[any]`.
The following are the interaction requirements:
1. the atom-atom distance between donor and acceptor must be lower or equal than 3.5 Angstrom
2. the angle between the donor-halogen atoms and the halogen-acceptor atoms must be between 130 and 180 degrees.
3. the angle between the halogen-acceptor atoms and the acceptor-any atoms must be between 80 and 140 degrees.


### Ionic interaction

This interaction is between two ions, i.e. `[cation] ... [anion]`, or between a cation and an aromatic ring, i.e. `[cation] ... [aromatic_ring]`.
The following are the interaction requirements between two ions:
1. the atom-atom distance between anion and cation must be lower or equal than 4.5 Angstrom

The following are the interaction requirements between a cation and an aromatic ring:
1. the distance between the cation and the centroid of the aromatic ring must be lower or equal than 4.5 Angstrom
2. the angle between the normal to the plane defined by the aromatic ring and the cation-ring centroid vector must be between 0 and 30 degrees.


### Pi stacking

This interaction is between two aromatic rings, i.e. `[aromatic_ring] ... [aromatic_ring]`.
The following are the interaction requirements between two aromatic rings when they face each other:
1. the distance between the two ring centroid must be lower or equal than 5.5 Angstrom
2. the angle between the two planes should be between 0 and 30 degrees
3. the angle between the normal to the plane and a centroid must be between 0 and 33 degrees

The following are the interaction requirements between two aromatic rings when they are perpendicular:
1. the distance between the two ring centroids must be lower or equal than 6.5 Angstrom
2. the angle between the two planes should be between 50 and 90 degrees
3. the angle between the normal to the plane and a centroid must be between 0 and 30 degrees
4. the centroid of the perpendicular ring falls within the other molecule ring

### Metal

This interaction is between a metal atom and a chelated atom, i.e. `[metal] ... [chelated]`.
The following are the interaction requirements:
1. atom-atom distance must be lower or equal than 2.8 Angstrom

## SMARTS pattern

This is the list of the SMARTS pattern that identify one or more atoms.
We can have more pattern for the same name, with a different number of atoms.

| Pattern name            | Num atoms | SMARTS string                                      |
|-------------------------|-----------|----------------------------------------------------|
| `hydrophobic`           |  1        | `[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]` |
| `hydrogen_donor-H`      |  2        | `[$([O,S;+0]),$([N;v3,v4&+1]),n+0]-[H]` |
| `hydrogen_acceptor`     |  1        | `[#7&!$([nX3])&!$([NX3]-*=[O,N,P,S])&!$([NX3]-[a])&!$([Nv4&+1]),O&!$([OX2](C)C=O)&!$(O(~a)~a)&!$(O=N-*)&!$([O-]-N=O),o+0,F&$(F-[#6])&!$(F-[#6][F,Cl,Br,I])]` |
| `halogen_donor-halogen` |  2        | `[#6,#7,Si,F,Cl,Br,I]-[Cl,Br,I,At]` |
| `halogen_acceptor-any`  |  2        | `[#7,#8,P,S,Se,Te,a;!+{1-}][*]` |
| `anion`                 |  1        | `[-{1-},$(O=[C,S,P]-[O-])]` |
| `cation`                |  1        | `[+{1-},$([NX3&!$([NX3]-O)]-[C]=[NX3+])]` |
| `aromatic_ring`         |  5 or 6   | `[a;r5]1:[a;r5]:[a;r5]:[a;r5]:[a;r5]:1` or `[a;r6]1:[a;r6]:[a;r6]:[a;r6]:[a;r6]:[a;r6]:1` |
| `metal`                 |  1        | `[Ca,Cd,Co,Cu,Fe,Mg,Mn,Ni,Zn]` |
| `chelated`              |  1        | `[O,#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4]),-{1-};!+{1-}]` |


## Reference

> [1] Bouysset, Cédric, and Sébastien Fiorucci. "ProLIF: a library to encode molecular interactions as fingerprints." Journal of Cheminformatics 13 (2021): 1-9.
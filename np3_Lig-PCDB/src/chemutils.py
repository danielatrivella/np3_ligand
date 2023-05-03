import rdkit.Chem as Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

atoms_num_outer_e = {'B': 3,
                     'C': 4,
                     'N': 5, 'P': 5,
                     'O': 6, 'S': 6, 'Se': 6,
                     'Cl': 7, 'F': 7, 'Br': 7, 'I': 7
                     }
SN_hybridization_label = ['sp', 'sp2', 'sp3', 'sp3d', 'sp3d2', 'sp3d3'] # SN-2

def carbon_atom(atom):
    new_atom_C = Chem.Atom('C')
    # new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom_C.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom_C

# make all mol atoms equal Carbon if atoms = True; and make all bonds single if bonds TRUE except for AROMATIC bonds
def simplify_mol(mol, atoms = True, bonds=True):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        if atoms:
            new_atom = carbon_atom(atom)
        else:
            new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if bonds and (not bond.GetIsAromatic() or bond.GetIsConjugated()):
            bt = Chem.BondType.SINGLE
        elif bond.GetIsAromatic() and not bond.GetIsConjugated():
            bt = Chem.BondType.AROMATIC
        else:
            bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        Chem.Kekulize(new_mol)
    return new_mol

def makeSimpleBonds(mol):
    for b in mol.GetBonds():
        if not b.GetIsAromatic():
            b.SetBondType(Chem.BondType.SINGLE)

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_mol_sanitize(smiles):
    mol = Chem.MolFromSmiles(smiles,sanitize=False)
    if mol is None: 
        return None
    # sanitize without valence error
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol,
                     Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                     Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                     Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                     catchErrors=True)
    Chem.Kekulize(mol)
    return mol

def restore_aromatics(mol):
    if Chem.SanitizeMol(mol) != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        print("Failed to restore the aromatic bond type")

def get_mol(smiles, kekule = True, addHs = False):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        mol = add_CNOFH_charges(smiles, kekule=kekule)
        if mol is None:
            print("*** Could not fix smiles", smiles, " ***")
            return None
        else:
            print("*** Valence of smiles", smiles,"fixed with extra charge ***")
    if addHs:
        mol = Chem.AddHs(mol)
    if kekule:
        Chem.Kekulize(mol)
    return mol

def get_smiles(mol, kekule = True):
    return Chem.MolToSmiles(mol, kekuleSmiles=kekule)

#########
# Code extracted from: Official implementation of our Junction Tree Variational Autoencoder https://arxiv.org/abs/1802.04364
# Github: https://github.com/wengong-jin/icml18-jtnn
#########
def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D

def charge_zero(mol):
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)


def add_CNOFH_charges(smiles, kekule=True):
    m = Chem.MolFromSmiles(smiles,sanitize=False)
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if ps:
        for p in ps:
            if p.GetType()=='AtomValenceException':
                at = m.GetAtomWithIdx(p.GetAtomIdx())
                explicitValence = at.GetExplicitValence() - at.GetFormalCharge()
                # print(at.GetAtomicNum(), at.GetSymbol(), at.GetExplicitValence(), at.GetFormalCharge())
                # C
                if at.GetAtomicNum() == 6 and explicitValence >= 5:
                    at.SetFormalCharge(explicitValence - 4)
                # N
                elif at.GetAtomicNum() == 7 and explicitValence >= 4:
                    at.SetFormalCharge(explicitValence - 3)
                # O
                elif at.GetAtomicNum() == 8 and explicitValence >= 3:
                    at.SetFormalCharge(explicitValence - 2)
                # H or F
                elif at.GetAtomicNum() in [1, 9] and explicitValence >= 2:
                    at.SetFormalCharge(explicitValence - 1)
    return sanitize(m, kekule=kekule)

def fix_smiles(smiles):
    m = get_mol(smiles)
    if m:
        return get_smiles(m)
    return None

def print_molTree(tree):
    for node in tree.nodes:
        print(node.smiles)
        print(node.clique)

def steric_number_label(atom):
    resonanceCorrection = 0
    # detect nitro group, correct Oxygen hybridization that is represented with a simple bond to the nitrogen and received an hydrogen
    if atom.GetSymbol() == 'O' and atom.GetDegree() == 2 and (atom.GetNeighbors()[0].GetFormalCharge() == 1 and atom.GetNeighbors()[0].GetSymbol() == 'N' and atom.GetNeighbors()[1].GetSymbol() == 'H'):
        resonanceCorrection = -1
    sigma_bonds = atom.GetDegree() + atom.GetImplicitValence() + resonanceCorrection # number of bonds (simple, double or triple) + number of hidrogen bonds
    lone_pairs_e = max((atoms_num_outer_e[atom.GetSymbol()] - atom.GetTotalValence() - atom.GetFormalCharge())/2, 0.0)
    SN = sigma_bonds+lone_pairs_e
    label = SN_hybridization_label[int(SN-2)]
    # if resonanceLabel:
    #     label = label + 'R'
    return label

# get the atom object, a list with the atoms in each rings and the ring aromaticity;
# return the atom label depending in which rings it appears or empty if it does not appear in any ring
# labels: C3,C4,C5,C6,C7,CA5,CA6,CA7
def atom_label_ring_size(atom, atom_rings, aromatic_rings):
    if not atom.IsInRing():
        return ""
    # label atoms ring size using precedence based on geometric restriction: aromatic > small ring > big ring
    # check if it appears in multiple rings
    count_rings = [r.count(atom.GetIdx()) for r in atom_rings]
    number_rings = count_rings.count(1)
    if number_rings == 0: # atom appeared in a ring with size > 7
        return ""
    else:
        atom_rings = [atom_rings[i] for i in range(len(atom_rings)) if count_rings[i] == 1]
        aromatic_rings = [aromatic_rings[i] for i in range(len(aromatic_rings)) if count_rings[i] == 1]
        #
        if sum(aromatic_rings) > 0: # atom appears in an aromatic ring with size <= 7
            atom_rings_size = [len(atom_rings[i]) for i in range(number_rings)
                               if aromatic_rings[i]]
            label_ring_type = "CA"
        else:
            atom_rings_size = [len(atom_rings[i]) for i in range(number_rings)]
            label_ring_type = "C"
        # if number_rings > 1:
        #     label_ring_size = "C" * number_rings  # number of rings that the atom appears
        # else:
        # else the atom appear in only one ring
        # if atom.GetIsAromatic():
        #     label_ring_size = "CA"
        # else:
        #     label_ring_size = "C"
        label_ring_size = label_ring_type + str(min(atom_rings_size))
    return label_ring_size

# To detect aromatic rings, I would loop over the bonds in each ring and
# flag the ring as aromatic if all bonds are aromatic:
def isRingAromatic(mol, bondRing):
        for id in bondRing:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True

# label the mol atoms with their SN and ring occurrence;  rings with size > 7 are ignored
def label_mol_atoms_SN(mol, notH = True):
    labels = []
    ri = mol.GetRingInfo()
    atom_rings = [list(x) for x in ri.AtomRings() if len(x) <= 7]
    aromatic_rings = [isRingAromatic(mol, x) for x in ri.BondRings() if len(x) <= 7]
    for a in mol.GetAtoms():
        if a.GetSymbol() == "H":
            if notH:
                continue
            else:
                labels.append("-1")
        else:
            labels.append(steric_number_label(a)+atom_label_ring_size(a, atom_rings, aromatic_rings))
    return labels

# if steric number True label using the SP hybridization, else use the atom symbol
def label_mol_atoms(mol, steric_number=True, notH = True):
    labels = []
    ri = mol.GetRingInfo()
    atom_rings = [list(x) for x in ri.AtomRings() if len(x) <= 7]
    aromatic_rings = [isRingAromatic(mol, x) for x in ri.BondRings() if len(x) <= 7]
    for a in mol.GetAtoms():
        if a.GetSymbol() == "H":
            if notH:
                continue
            else:
                labels.append("-1")
        else:
            if steric_number:
                labels.append(steric_number_label(a)+atom_label_ring_size(a, atom_rings, aromatic_rings))
            else:
                labels.append(a.GetSymbol() + atom_label_ring_size(a, atom_rings, aromatic_rings))
    return labels

def print_atoms_valence(mol, symbol='N'):
    Chem.Kekulize(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == symbol:
            bond_count = 0
            for bond in atom.GetBonds():
                bond_count = bond_count +bond.GetBondTypeAsDouble()
            print(str(atom.GetIdx())+" isInCycle "+str(atom.IsInRing())+" SN Hybridization "+steric_number_label(atom)+" Rdkit hybridization "+str(atom.GetHybridization())+" implicit "+str(atom.GetImplicitValence())+" explicit "+
                  str(atom.GetExplicitValence())+' charge '+str(atom.GetFormalCharge())+ ' total valence '+
                  str(atom.GetTotalValence())+ ' degree '+str(atom.GetDegree())+' bond count '+str(bond_count))

def print_mol_bonds(mol, atoms=None):
    if not atoms:
        atoms = range(mol.GetNumAtoms())
    for a in atoms:
        atom = mol.GetAtomWithIdx(a)
        print(atom.GetSymbol())
        for bond in atom.GetBonds():
            print(str(bond.GetBondType()) + ' ' + str(bond.GetBondTypeAsDouble())+ ' isAromatic '+str(bond.GetIsAromatic()))

def sanitize(mol, kekule = True):
    try:
        smiles = get_smiles(mol, kekule=kekule)
        mol = get_mol(smiles, kekule=kekule)
    except Exception as e:
        print(e)
        return None
    return mol

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def neutralizeRadicalsFormalCharge(mol):
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons() >= 1 or a.GetFormalCharge() >= 1:
            a.SetNumRadicalElectrons(0)
            a.SetFormalCharge(0)

def get_clique_mol(mol, atoms, kekule=True):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=kekule)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol

# decompose a mol in features, where the features are cycles or single bonds not in cycles
# bonds_simple: if TRUE return all bonds, even those in rings - all rings will receive makeSimpleBonds for general matching
def mol_decomp_features(mol, cycle_bonds_simple):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [[0]]
    #
    # get list of edges in the mol that do not appear in a ring
    edges = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        # if simple bonds and not aromatics, also add bond to the edges list
        # TODO only add edges of cycles greater than 6 even if aromatic
        if not bond.IsInRing() or (cycle_bonds_simple and not bond.GetIsAromatic()):
            edges.append([a1, a2])
    # get list of simple rings
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # concate the found features
    features = edges+ssr
    return features  # lists with the atoms idxs that compose each cycle or edge

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

#Bond type not considered because all aromatic (so SINGLE matches DOUBLE)
def ring_bond_equal(b1, b2, reverse=False):
    b1 = (b1.GetBeginAtom(), b1.GetEndAtom())
    if reverse:
        b2 = (b2.GetEndAtom(), b2.GetBeginAtom())
    else:
        b2 = (b2.GetBeginAtom(), b2.GetEndAtom())
    return atom_equal(b1[0], b2[0]) and atom_equal(b1[1], b2[1])

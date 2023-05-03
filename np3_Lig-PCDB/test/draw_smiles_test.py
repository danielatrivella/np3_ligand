import rdkit.Chem as Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem
import pandas as pd
import os

atoms_num_outer_e = {'B': 3,
                     'C': 4,
                     'N': 5, 'P': 5,
                     'O': 6, 'S': 6,
                     'Cl': 7, 'F': 7, 'Br': 7, 'I': 7
                     }
SN_hybridization_label = ['sp', 'sp2', 'sp3', 'sp3d', 'sp3d2'] # SN-2

def fix_valence_charge(mol, kekule=True):
    for atom in mol.GetAtoms():
        # print(atom.GetDegree(), atom.GetSymbol(), atom.GetBonds)
        if atom.GetSymbol() == 'N':
            explicitValence = 0
            for bond in atom.GetBonds():
                explicitValence = explicitValence + bond.GetBondTypeAsDouble()
            if explicitValence > 3:
                atom.SetFormalCharge(int(explicitValence-3))
        elif atom.GetSymbol() == 'O':
            explicitValence = 0
            for bond in atom.GetBonds():
                explicitValence = explicitValence + bond.GetBondTypeAsDouble()
            if explicitValence > 2:
                atom.SetFormalCharge(int(explicitValence - 2))
        elif atom.GetSymbol() == 'C':
            explicitValence = 0
            for bond in atom.GetBonds():
                explicitValence = explicitValence + bond.GetBondTypeAsDouble()
            if explicitValence > 4:
                atom.SetFormalCharge(int(explicitValence - 4))
    return sanitize(mol, kekule=kekule)

def get_smiles(mol, kekule = True):
    return Chem.MolToSmiles(mol, kekuleSmiles=kekule)

def get_mol(smiles, kekule = True, addHs = False):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = fix_valence_charge(mol, kekule=kekule)
        if mol is None:
            return None
        else:
            print("Valence fixed with extra charge")
    if addHs:
        mol = Chem.AddHs(mol)
    if kekule:
        Chem.Kekulize(mol)
    return mol

def sanitize(mol, kekule = True):
    try:
        smiles = get_smiles(mol, kekule=kekule)
        mol = get_mol(smiles, kekule=kekule)
    except Exception as e:
        return None
    return mol

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            atom.SetAtomMapNum(atom.GetIdx())

def draw_mol_annotation(smiles, lig_code):
    mol = get_mol(smiles, kekule=False, addHs=True)
    mol_with_atom_index(mol)
    d = rdMolDraw2D.MolDraw2DSVG(500, 500)
    mol.GetAtomWithIdx(2).SetProp('atomNote', 'foo')
    mol.GetBondWithIdx(0).SetProp('bondNote', 'bar')
    d.drawOptions().addAtomIndices = True
    # AllChem.EmbedMolecule(mol)
    AllChem.Compute2DCoords(mol)
    d.DrawMolecule(mol)
    d.FinishDrawing()
    with open('test/ligands_label/ligands_draw/'+lig_code+'_'+smiles+'.png', 'w') as f:
        f.write(d.GetDrawingText())


lig_file = 'test/ligands_label/ligands_code_smiles.csv'
lig_data = pd.read_csv(lig_file, skipinitialspace=True)
os.makedirs('test/ligands_label/ligands_draw/', exist_ok=True)
for lig_r in lig_data.iterrows():
    print(lig_r[1][1])
    draw_mol_annotation(lig_r[1][0], lig_r[1][1])

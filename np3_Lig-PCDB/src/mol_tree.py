#########
# Code extracted from: Official implementation of our Junction Tree Variational Autoencoder https://arxiv.org/abs/1802.04364
# Github: https://github.com/wengong-jin/icml18-jtnn
# * Modified the main call
#########
import rdkit, sys
import rdkit.Chem as Chem
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble, decode_stereo, mol_decomp_features, restore_aromatics, makeSimpleBonds, simplify_mol, steric_number_label

class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique] #copy
        self.neighbors = []
        
    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf: #Leaf node, no need to mark 
                continue
            for cidx in nei_node.clique:
                #allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label
    
    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x:x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands,aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i,cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0: cands = new_cands

        if len(cands) > 0:
            self.cands, _ = zip(*cands)
            self.cands = list(self.cands)
        else:
            self.cands = []

# bonds_simple: decompose return all edges that are not aromatic as features, after decomposition make all bonds in a
#               cycle equal single type and restore their aromatics (generalize cycles that share the same atoms to represent cycles of a given size)
# features_simple: if TRUE wont merge cycles that share more than 2 atoms and wont add single nodes;
#                  if FALSE will follow JTVAE algorithm to decompose the SMILES
#                  return bonds and simple cycles
class MolTree(object):

    def __init__(self, smiles, features_simpler=False, cycle_simpler=False, atoms_simpler=False, bonds_simpler=False):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        #
        if self.mol is None:
            sys.exit("Could not fix valence charge\n\n")

        # make all atoms equal carbon and all bonds as single except aromatics
        if atoms_simpler or bonds_simpler:
            self.mol = simplify_mol(self.mol, atoms_simpler, bonds_simpler)
        #
        #Stereo Generation (currently disabled)
        #mol = Chem.MolFromSmiles(smiles)
        #self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        #self.smiles2D = Chem.MolToSmiles(mol)
        #self.stereo_cands = decode_stereo(self.smiles2D)
        # decompose the smiles, if simple is True do not merge cycles or add single nodes
        if not features_simpler:
            cliques, edges = tree_decomp(self.mol)
        else:
            # if cycle bonds simple is true also return the cycles bonds,
            # except if simple bonds and simple atoms are also true in which case there is no need to return the cycle bonds
            # # because they will also be simplified and with no extra info
            cliques = mol_decomp_features(self.mol, (cycle_simpler and (not bonds_simpler or not atoms_simpler)))

        self.nodes = []
        root = 0
        for i,c in enumerate(cliques):
            # print(i)
            cmol = get_clique_mol(self.mol, c)
            if len(c) > 2: # restore aromatics from rings
                # TODO only make simple bonds if cycle greater than 6
                if cycle_simpler:
                    #makeSimpleBonds(cmol) # make single bonds in the cycle
                    cmol = simplify_mol(cmol, True, True) # make single bonds and simple atoms - return the simple cycle representing its size

                restore_aromatics(cmol)
            # neutralizeRadicals(cmol)
            csmiles = get_smiles(cmol)
            # if 'H' in csmiles:
            #     csmiles = csmiles.replace('H','')
            node = MolTreeNode(csmiles, c)
            self.nodes.append(node)
            if min(c) == 0: root = i
        # if not features_simple:
        #     for x,y in edges:
        #         self.nodes[x].add_neighbor(self.nodes[y])
        #         self.nodes[y].add_neighbor(self.nodes[x])
        #
        #     if root > 0:
        #         self.nodes[0],self.nodes[root] = self.nodes[root],self.nodes[0]
        #
        #     for i,node in enumerate(self.nodes):
        #         node.nid = i + 1
        #         if len(node.neighbors) > 1: #Leaf node mol is not marked
        #             set_atommap(node.mol, node.nid)
        #         node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()

def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx: continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


if __name__ == "__main__":
    import sys
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    if len(sys.argv) >=6:
        print(sys.argv)
        output_vocab = sys.argv[1]
        simple_features = bool(sys.argv[2] == "True")
        simple_bonds_cycles = bool(sys.argv[3] == "True")
        simple_atoms = bool(sys.argv[4] == "True")
        simple_bonds = bool(sys.argv[5] == "True")
    else:
        output_vocab = 'vocabulary.txt'
        simple_features = False
        simple_bonds_cycles = False
        simple_atoms = False
        simple_bonds = False
    print('Simple features: '+str(simple_features))
    print('Simple bonds cycles: ' + str(simple_bonds_cycles))
    print('Simple atoms: ' + str(simple_atoms))
    print('Simple bonds: ' + str(simple_bonds))
    cset = set()

    i=0
    print("\n** Start creating vocabulary **\n")
    for line in sys.stdin:
        i = i + 1
        if i%100 == 0:
            print("line "+str(i))
        smiles = line.split()[0]
        # print("line " + str(i)+" smiles "+smiles)
        mol = MolTree(smiles, simple_features, simple_bonds_cycles, simple_atoms, simple_bonds)
        for c in mol.nodes:
            cset.add(c.smiles)
    from pathlib import Path

    print("\nVocabulary:\n")
    with open(output_vocab, 'w') as fo:
        for x in cset:
            print(x)
            fo.write(str(x) + "\n")
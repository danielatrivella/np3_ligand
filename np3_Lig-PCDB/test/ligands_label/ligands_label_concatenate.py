#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:23:10 2020

@author: Luiz np3
"""


import pandas as pd

csv = "test/ligands_label/ligands_code_smiles.csv"

dfCode = pd.read_csv(csv)

atom = []
SN = []
Label = []
for code in dfCode.lig_code:
    code = code.strip()
    aux = pd.read_csv("test/ligands_label/"+code+".csv")
    atm_str = ""
    SN_str = ""
    lbl_str = ""
    for idx in aux.index:
       atm_str += str(aux["Atom"][idx])
       atm_str += ";"
       SN_str += str(aux["SN"][idx])
       SN_str += ";"
       lbl_str += str(aux["Label"][idx])
       lbl_str += ";"
    
    atm_str = atm_str[:len(atm_str)-1]
    SN_str = SN_str[:len(SN_str)-1]
    lbl_str = lbl_str[:len(lbl_str)-1]
    
    atom.append(atm_str)
    SN.append(SN_str)
    Label.append(lbl_str)
    

dfCode["Atoms"] = atom
dfCode["SNs"] = SN
dfCode["Labels"] = Label

dfCode.to_csv("test/ligands_label/ligands_label.csv", index=False)

"""Process either monodentate or bidentate ligands generated as enriched
SMILES.

COPIED FROM DIFFERENT CONTEXT, LOOSE CONNECTIONS LIKELY.

The input is expected to be a text file with one enriched SMILES per line, where substitute atoms
have been added to the SMILES string to encode the connection IDs, the indices of each ligand's
coordination atoms. The set of ligands used to train the generating model must also be provided.

The output is expected to be a `.csv` file with columns for both a canonical `SMILES` string and
the associated 0-indexed connection IDs, as well as the original `enriched_SMILES` string from
which the two former fields are extracted.

Additional columns like `atom_count` are also permitted.

Usage:

python process_generated_ligands.py -d monodentate -i sample_monodentates.txt \
                                    -t ../ligand_generation/data/training_set_monodentates.csv \
                                    -o processed_generated_monodentates.csv

python process_generated_ligands.py -d bidentate -i sample_bidentates.txt \
                                    -t ../ligand_generation/data/training_set_bidentates.csv \
                                    -o processed_generated_bidentates.csv
"""


import argparse
import re

# from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from rdkit import Chem


def get_args(arg_list: list = None) -> argparse.Namespace:
    """Parse arguments from command line.

    Args:
        arg_list (list, optional): Automatically obtained from the command line if provided.
        If no arguments are given but default arguments are defined, the latter are used.

    Returns:
        argparse.Namespace: Dictionary-like class that contains the driver arguments.
    """
    parser = argparse.ArgumentParser(
        description="Strip substitute atoms from generated ligands"
    )
    parser.add_argument(
        "--denticity",
        "-d",
        choices=[
            "monodentate",
            "bidentate",
        ],
        required=True,
        help="""Select one of the two denticity modes supported""",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        type=Path,
        required=True,
        help="Input file with generated enriched SMILES",
    )
    parser.add_argument(
        "--training_set",
        "-t",
        type=Path,
        required=True,
        help="File with enriched SMILES used when training the generating model",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=Path,
        required=True,
        help="Output file with enriched SMILES, canonical SMILES, and connection IDs",
    )
    return parser.parse_args(arg_list)


def remove_training_duplicates(
    processed_generated: dict, training_set_path: Path
) -> dict:
    """Remove generated ligands which are also in the training set.

    Args:
        processed_generated (dict): generated ligands with canonical SMILES as key.
        training_set_path (Path): a .csv file containing the field "enriched_smiles".

    Returns:
        (dict): updated dict of generated ligands without ligands in training set.
    """
    df_train = pd.read_csv(training_set_path)
    training_set = set([x.rstrip() for x in df_train.enriched_smiles.values.tolist()])
    for smiles in training_set:
        mol = Chem.MolFromSmiles(smiles)
        mol, _ = process_substitute_attachment_points_bidentate(mol)
        canon_smiles, _ = canon_remap(mol)
        if canon_smiles in processed_generated:
            del processed_generated[canon_smiles]
    return processed_generated


def count_dative_bidentate_ir(smiles):
    regex = r"->\[Ir\]<-"
    matches = re.findall(regex, smiles)
    return len(matches)


def count_berylium(smiles):
    regex = r"Be"
    matches = re.findall(regex, smiles)
    return len(matches)


def count_iridium(smiles):
    regex = r"Ir"
    matches = re.findall(regex, smiles)
    return len(matches)


def count_dative(smiles):
    regex1 = r"->"
    matches1 = re.findall(regex1, smiles)
    regex2 = r"<-"
    matches2 = re.findall(regex2, smiles)
    return len(matches1) + len(matches2)


def clear_enriched(enriched_smiles: list, denticity: str) -> list:
    """Screen enriched SMILES to omit generated instances without required
    properties.

    Args:
        enriched_smiles (list): enriched SMILES strings.
        denticity (str): either "monodentate" or "bidentate".

    Returns:
        list: enriched SMILES that have cleared the string-based screening.
    """
    clear_enriched_smiles = []

    if denticity == "monodentate":  # TODO Magnus please check this clause.
        for smiles in enriched_smiles:
            if count_iridium(smiles) == 1 and count_dative(smiles) == 1:
                clear_enriched_smiles.append(smiles)
    elif denticity == "bidentate":
        for smiles in enriched_smiles:
            if (
                count_dative_bidentate_ir(smiles) == 1
                and count_iridium(smiles) == 1
                and count_dative(smiles) == 2
            ):
                clear_enriched_smiles.append(smiles)

    return clear_enriched_smiles


def canon_remap(mol: Chem.rdchem.Mol) -> Tuple[str, dict]:
    """Get the canonical SMILES and associated re-mapping of atom indices.

    Note the assumption that the SMILES strings rely on implicit hydrogens.

    Args:
        mol (Chem.rdchem.Mol): a Mol object created from de-enriched SMILES.

    Returns:
        str: canonical SMILES.
        dict: mapping of previous atom indices to atom indices in canonical SMILES.
    """
    mol = Chem.RemoveHs(mol)
    canon_smiles = Chem.MolToSmiles(mol)
    order = eval(mol.GetProp("_smilesAtomOutputOrder"))
    hless_mapped_ids = {}
    for i in range(0, len(mol.GetAtoms())):
        mapped_id = np.where(np.array(order) == i)[0][0]
        hless_mapped_ids[i] = mapped_id
    return canon_smiles, hless_mapped_ids


def count_atoms_hfull(mol: Chem.rdchem.Mol) -> int:
    """Count the number of atoms in the molecule.

    Args:
        mol (Chem.rdchem.Mol): molecule whose number of atoms is being counted.

    Returns:
        atom_count (int): number of atoms in moleculre, including hydrogen atoms.
    """
    rwmol = Chem.RWMol(mol)
    atom_count = 0
    molwh = Chem.AddHs(rwmol)
    for _ in molwh.GetAtoms():
        atom_count += 1
    assert atom_count == molwh.GetNumAtoms()
    return atom_count


def remove_single_atom(mol, idx):
    """Function that removes an atom at specified idx.

    Args:
        mol (Chem.rdchem.Mol): mol object from which to remove `idx`'th atom.
        idx (int): index of atom to remove from the input mol.

    Returns:
        Chem.rdchem.Mol: revised mol with the `idx`'th atom removed and remaining atom indices
        updated.
    """
    res = Chem.RWMol(mol)
    res.BeginBatchEdit()
    res.RemoveAtom(idx)
    res.CommitBatchEdit()
    Chem.SanitizeMol(res)
    return res.GetMol()


def process_substitute_attachment_points_bidentate(
    mol: Chem.rdchem.Mol,
):  # TODO rename?
    """De-enrich a Mol from enriched SMILES of bidentate ligands and extract
    connection indices.

    Args:
        mol (Chem.rdchem.Mol): a Mol representing an enriched SMILES string.

    Returns:
        Chem.rdchem.Mol: cleaned Mol without the enrichment of Ir and Be atoms.
        list: integers for connection atom indices as indicated by the removed enrichment.
    """
    # Determine the Ir substitute atom index.
    substitute_smarts = Chem.MolFromSmarts("[Ir]")
    matches = mol.GetSubstructMatches(substitute_smarts)

    # If there are several matches for Ir we discard the mol.
    connection_ids = None
    if len(matches) > 1:
        new_mol = None
    elif not matches:
        new_mol = None
    else:
        # Get the Ir Atom object.
        mol.GetAtomWithIdx(matches[0][0])

        # Get neighbors of Ir.
        neighbors = mol.GetAtomWithIdx(matches[0][0]).GetNeighbors()

        # For bidentates, there should be exactly 2 neighbors.
        if len(neighbors) != 2:
            print("Bidentate ligand requires exactly two neighbors of the Ir.")
            return (
                None,
                None,
            )  # TODO incorporate/adjust how None, None returns are handled.

        try:
            tmp_mol = remove_single_atom(mol, matches[0][0])
        except Exception as e:
            print("Single atom removal failed with error: ")
            print(e)
            return None, None

        # Loop through neighbors to create connection_ids list.
        connection_ids = []
        for n in neighbors:
            id = n.GetIdx()

            # If Be, we get the id later, after Be removal.
            if n.GetSymbol() == "Be":
                continue

            # Since we are removing the Ir atom, higher atom indices than the Ir atom will
            # decrease by one. We therefore need to track how this affects connection ids.
            if id > matches[0][0]:
                new_id = id - 1
                connection_ids.append(new_id)
            else:
                connection_ids.append(id)

        # Finally, if either of the neighbors to the Ir is a Be, we need to remove them as well.
        be_match = tmp_mol.GetSubstructMatches(Chem.MolFromSmarts("[Be]"))

        if be_match:
            res = Chem.RWMol(tmp_mol)
            res.BeginBatchEdit()

            for match in be_match:
                carbene_neighbor = res.GetAtomWithIdx(match[0]).GetNeighbors()[0]
                carbene_neighbor_idx = carbene_neighbor.GetIdx()

                # We explicitely ensure that the carbon atom now is a carbene with 2 radical
                # electrons and 0 hydrogens.
                carbene_neighbor.SetNumRadicalElectrons(2)
                carbene_neighbor.SetNoImplicit(True)
                carbene_neighbor.SetNumExplicitHs(0)

                res.RemoveAtom(match[0])

                # If the index of the carbene is larger than that of the attached Be, the updated
                # index of that carbene decreases by the number of Be removed.
                idx_decrease = sum(i[0] < carbene_neighbor_idx for i in be_match)
                # If a previously identified non-carbene neighbor has a higher index than a
                # carbene neighbor, the former will also have its index lowered by one.
                if connection_ids and idx_decrease == 0 and len(be_match) == 1:
                    connection_ids[0] -= 1

                carbene_neighbor_idx = carbene_neighbor_idx - idx_decrease
                connection_ids.append(carbene_neighbor_idx)

            res.CommitBatchEdit()
            Chem.SanitizeMol(res)
            tmp_mol = res.GetMol()
        new_mol = tmp_mol
    return new_mol, connection_ids


def process_substitute_attachment_points_monodentate(mol):
    """Remove Be or Li substitutes on mol objects.

    Args:
        mol (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Determine which attachment. Not working for Ir yet.
    substitute_smarts = Chem.MolFromSmarts("[Be,Li]")

    matches = mol.GetSubstructMatches(substitute_smarts)
    # If there are several matches for monodentates, then molecule should be discarded.
    connect_id = None
    if len(matches) > 1:
        new_mol = None
    elif not matches:
        new_mol = None
    else:
        match_atom = mol.GetAtomWithIdx(matches[0][0])

        # We remove the substitute here. Since Li or Be is always 0, the coordinating atom
        # will get id 0 after removal.
        try:
            new_mol = remove_single_atom(mol, matches[0][0])
            connect_id = 0
        except Exception as e:
            print("Single atom removal failed with error: ")
            print(e)
            return None, None

        # If Be, then the carbon Be was connected to should be a Carbene.
        if match_atom.GetSymbol() == "Be":
            # For monodentates, the Be will always be the first atom. Therefore,
            # the carbene will be the new first atom.
            # 2 radical atoms to create the carbene.
            new_mol.GetAtomWithIdx(0).SetNumRadicalElectrons(2)

            # Ensure that there are no hydrogens on the Carbon atom
            new_mol.GetAtomWithIdx(0).SetNoImplicit(True)
            new_mol.GetAtomWithIdx(0).SetNumExplicitHs(0)

    return new_mol, connect_id


# if __name__ == "__main__":
#     args = get_args()

#     with open(args.input_file, "r") as f:
#         enriched_smiles = [line.rstrip() for line in f.readlines()]

#     enriched_smiles = clear_enriched(enriched_smiles, args.denticity)

#     # Process enriched SMILES list for duplicates, first just internally:
#     enriched_smiles_set = set()
#     enriched_smiles_temp = []
#     for smiles in enriched_smiles:
#         if smiles not in enriched_smiles_set:
#             enriched_smiles_temp.append(smiles)
#             enriched_smiles_set.add(smiles)
#     enriched_smiles = enriched_smiles_temp

#     processed_generated = {}  # TODO rename this variable more meaningfully.
#     for smiles in enriched_smiles:
#         mol = Chem.MolFromSmiles(smiles)
#         if args.denticity == "monodentate":
#             (
#                 clean_mol,
#                 connection_ids,
#             ) = process_substitute_attachment_points_monodentate(mol)
#         elif args.denticity == "bidentate":
#             clean_mol, connection_ids = process_substitute_attachment_points_bidentate(
#                 mol
#             )
#         canon_smiles, hless_mapped_ids = canon_remap(clean_mol)
#         connection_ids = sorted([hless_mapped_ids[x] for x in connection_ids])
#         atom_count = count_atoms_hfull(clean_mol)
#         # TODO ? electron_count = count_electrons_hfull(clean_mol)
#         # https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html
#         if (
#             canon_smiles in processed_generated
#         ):  # Despite enriched SMILES not being duplicates.
#             if processed_generated[canon_smiles]["connection_ids"] != connection_ids:
#                 # Preparing to screen out all ligands that are identical as canonicalized SMILES
#                 # but have multiple sets of connection points.
#                 processed_generated[canon_smiles] = {
#                     "mol": None,
#                     "enriched_SMILES": smiles,
#                     "connection_ids": "delete me",
#                 }
#             else:
#                 continue
#         else:
#             processed_generated[canon_smiles] = {
#                 "mol": mol,
#                 "enriched_SMILES": smiles,
#                 "connection_ids": connection_ids,
#                 "atom_count": atom_count,
#             }
#     delkeys = []
#     for smiles in processed_generated.keys():
#         if processed_generated[smiles]["connection_ids"] == "delete me":
#             delkeys.append(smiles)
#     for key in delkeys:
#         del processed_generated[key]
#     # Remove external duplicates, i.e., generated ligands that recreate training instances.
#     processed_generated = remove_training_duplicates(
#         processed_generated, args.training_set
#     )
#     data = defaultdict(list)
#     for canon_smiles, val in sorted(
#         processed_generated.items(), key=lambda x: x[1]["atom_count"]
#     ):
#         data["atom_count"].append(val["atom_count"])
#         data["canonical_SMILES"].append(canon_smiles)
#         data["connection_ids"].append(val["connection_ids"])
#         data["enriched_SMILES"].append(val["enriched_SMILES"])

#     df = pd.DataFrame(data)
#     df.to_csv(args.output_file, index=False)

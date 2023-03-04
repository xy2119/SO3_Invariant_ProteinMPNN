# SO3_Invariant_ProteinMPNN
## 📌 Master Thesis on Geometric Deep Learning on Protein Design
This is the code for the master thesis on introducing rotation invariant representation to ProteinMPNN, supervised by <a href="https://www.imperial.ac.uk/people/s.angioletti-uberti">Professor Stefano Angioletti-Uberti</a>
<!-- ABOUT THE RESEARCH -->
<h2 id="about-the-research"> :pencil: About The Research</h2>

**ProteinMPNN** is a message passing neural network that aims to find an amino acid sequence that will fold into a given structure. The full network is composed of an encoder and a decoder with 3 layers each. The network takes as inputs the 3D coordinates and computes the following information for each residue: (i) the distance between the N, Cα, C, O and a virtual Cβ atom, (ii) the Cα − Cα − Cα frame orientation and rotation, (iii) the backbone dihedral angles, (iv) the distances to the 48 closest residues. 
 
![image](./images/ProteinMPNN.png)

Building rotation invariant representation using [**Sperical Harmonics**](https://stevejtrettel.site/code/2022/spherical-harmonics), 3D coordinates of residue are expanded into radial and spherical basis, the combination of coefficients constitute an invariant representation.
![image](./images/Spherical_Expand.jpg)

Sperical Harmonics Visualisation [Website](https://stevejtrettel.site/code/2022/spherical-harmonics)



## Dataset
Protein Data Bank

## Prerequisites

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>

<!--This project is written in Python programming language. <br>-->
The following are the major open source packages utilised in this project:
* Numpy
* SciPy
* Matplotlib
* Scikit-Learn
* Pytorch


<h2 id="folder-structure"> Folder Structure</h2>


```
📦 SO3_Invariant_ProteinMPNN
├─ .ipynb_checkpoints
│  ├─ README-checkpoint.md
│  ├─ protein_mpnn_run-checkpoint.py
│  └─ protein_mpnn_utils-checkpoint.py
├─ Group_Theory.pdf
├─ LICENSE
├─ README.md
├─ UniProt
│  ├─ README.md
│  └─ all_proteins.csv
├─ __pycache__
│  ├─ protein_mpnn_utils.cpython-310.pyc
│  └─ protein_mpnn_utils.cpython-38.pyc
├─ ca_model_weights
│  ├─ .DS_Store
│  ├─ Ca_OR_100e.pt
│  ├─ Ca_RSH3_100e.pt
│  ├─ v_48_002.pt
│  ├─ v_48_010.pt
│  └─ v_48_020.pt
├─ dataset
│  ├─ test_hetero.csv
│  └─ test_homo.csv
├─ examples
│  ├─ submit_example_1.sh
│  ├─ submit_example_2.sh
│  ├─ submit_example_3.sh
│  ├─ submit_example_3_score_only.sh
│  ├─ submit_example_3_score_only_from_fasta.sh
│  ├─ submit_example_4.sh
│  ├─ submit_example_4_non_fixed.sh
│  ├─ submit_example_5.sh
│  ├─ submit_example_6.sh
│  ├─ submit_example_7.sh
│  └─ submit_example_8.sh
├─ helper_scripts
│  ├─ assign_fixed_chains.py
│  ├─ make_bias_AA.py
│  ├─ make_bias_per_res_dict.py
│  ├─ make_fixed_positions_dict.py
│  ├─ make_pos_neg_tied_positions_dict.py
│  ├─ make_tied_positions_dict.py
│  ├─ other_tools
│  │  ├─ make_omit_AA.py
│  │  └─ make_pssm_dict.py
│  ├─ parse_multiple_chains.out
│  ├─ parse_multiple_chains.py
│  └─ parse_multiple_chains.sh
├─ images
│  ├─ ProteinMPNN.png
│  └─ Spherical_Expand.jpg
├─ inputs
│  ├─ PDB_complexes
│  │  └─ pdbs
│  │     ├─ 3HTN.pdb
│  │     └─ 4YOW.pdb
│  ├─ PDB_homooligomers
│  │  └─ pdbs
│  │     ├─ 4GYT.pdb
│  │     └─ 6EHB.pdb
│  └─ PDB_monomers
│     └─ pdbs
│        ├─ 5L33.pdb
│        └─ 6MRR.pdb
├─ notebooks
│  ├─ PDB_Spherical_Harmonics_processing.ipynb
│  ├─ ProteinMPNN_Spherical_Harmonics_train.ipynb
│  ├─ demo_CA_SH_wAF2.ipynb
│  ├─ find_pdb_domain.ipynb
│  └─ mpnn_wAF2.py
├─ outputs
│  ├─ example_1_outputs
│  │  ├─ parsed_pdbs.jsonl
│  │  └─ seqs
│  │     ├─ 5L33.fa
│  │     └─ 6MRR.fa
│  ├─ example_2_outputs
│  │  ├─ assigned_pdbs.jsonl
│  │  ├─ parsed_pdbs.jsonl
│  │  └─ seqs
│  │     ├─ 3HTN.fa
│  │     └─ 4YOW.fa
│  ├─ example_3_outputs
│  │  └─ seqs
│  │     └─ 3HTN.fa
│  ├─ example_3_score_only_outputs
│  │  └─ score_only
│  │     ├─ 3HTN.npz
│  │     └─ 3HTN_0.npz
│  ├─ example_4_non_fixed_outputs
│  │  ├─ assigned_pdbs.jsonl
│  │  ├─ fixed_pdbs.jsonl
│  │  ├─ parsed_pdbs.jsonl
│  │  └─ seqs
│  │     ├─ 3HTN.fa
│  │     └─ 4YOW.fa
│  ├─ example_4_outputs
│  │  ├─ assigned_pdbs.jsonl
│  │  ├─ fixed_pdbs.jsonl
│  │  ├─ parsed_pdbs.jsonl
│  │  └─ seqs
│  │     ├─ 3HTN.fa
│  │     └─ 4YOW.fa
│  ├─ example_5_outputs
│  │  ├─ assigned_pdbs.jsonl
│  │  ├─ fixed_pdbs.jsonl
│  │  ├─ parsed_pdbs.jsonl
│  │  ├─ seqs
│  │  │  ├─ 3HTN.fa
│  │  │  └─ 4YOW.fa
│  │  └─ tied_pdbs.jsonl
│  ├─ example_6_outputs
│  │  ├─ parsed_pdbs.jsonl
│  │  ├─ seqs
│  │  │  ├─ 4GYT.fa
│  │  │  └─ 6EHB.fa
│  │  └─ tied_pdbs.jsonl
│  ├─ example_7_outputs
│  │  ├─ parsed_pdbs.jsonl
│  │  └─ unconditional_probs_only
│  │     ├─ 5L33.npz
│  │     └─ 6MRR.npz
│  ├─ example_8_outputs
│  │  ├─ bias_pdbs.jsonl
│  │  ├─ parsed_pdbs.jsonl
│  │  └─ seqs
│  │     ├─ 5L33.fa
│  │     └─ 6MRR.fa
│  └─ training_test_output
│     └─ seqs
│        └─ 5L33.fa
├─ protein_mpnn_run.py
├─ protein_mpnn_utils.py
├─ training
│  ├─ LICENSE
│  ├─ README.md
│  ├─ colab_training_example.ipynb
│  ├─ exp_020
│  │  ├─ log.txt
│  │  └─ model_weights
│  │     └─ epoch_last.pt
│  ├─ model_utils.py
│  ├─ plot_training_results.ipynb
│  ├─ submit_exp_020.sh
│  ├─ test_inference.sh
│  ├─ training.py
│  └─ utils.py
├─ vanilla_model_weights
│  ├─ v_48_002.pt
│  ├─ v_48_010.pt
│  ├─ v_48_020.pt
│  ├─ v_48_030.pt
│  ├─ vanilla_OR_100e.pt
│  ├─ vanilla_RSH3_100e.pt
│  └─ vanilla_RSH4_100e.pt
└─ weekly_notes.pdf
```


## 🎯 RoadMap



## Future Work
Benchmarking more physics-inspired representation methods

## Acknowledgements
My appreciation goes to Professor Stefano Angioletti-Uberti and PhD researcher Shanil from <a href="https://www.softnanolab.org/">SoftNanoLab</a> for their attentive guidance and great access to computing resources.

## Contributing
If you have any questions or suggestions towards this repository, feel free to contact me at xy2119@ic.ac.uk.

Any kind of enhancement or contribution is welcomed!

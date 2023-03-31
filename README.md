# SO3_Invariant_ProteinMPNN
## 📌 Master Thesis on Geometric Deep Learning on Protein Design
This is the code for the master thesis on introducing rotation invariant representation to ProteinMPNN, supervised by <a href="https://www.imperial.ac.uk/people/s.angioletti-uberti">Professor Stefano Angioletti-Uberti</a>
<!-- ABOUT THE RESEARCH -->
<h2 id="about-the-research"> :pencil: About The Research</h2>

**ProteinMPNN** is a message passing neural network that aims to find an amino acid sequence that will fold into a given structure. The full network is composed of an encoder and a decoder with 3 layers each. The network takes as inputs the 3D coordinates and computes the following information for each residue: (i) the distance between the N, Cα, C, O and a virtual Cβ atom, (ii) the Cα − Cα − Cα frame orientation and rotation, (iii) the backbone dihedral angles, (iv) the distances to the 48 closest residues. 
 
![image](./images/ProteinMPNN.png)

Building rotation invariant representation using [**Spherical Harmonics**](https://stevejtrettel.site/code/2022/spherical-harmonics), 3D coordinates of residue are expanded into radial and spherical basis, the combination of coefficients constitute an invariant representation.
![image](./images/Spherical_Expand.jpg)

Spherical Harmonics Visualisation [Website](https://stevejtrettel.site/code/2022/spherical-harmonics)



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
📦 
├─ .ipynb_checkpoints
│  ├─ README-checkpoint.md
│  ├─ protein_mpnn_run-checkpoint.py
│  └─ protein_mpnn_utils-checkpoint.py
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
│  ├─ Ca_OR.pt
│  ├─ Ca_RSH1.pt
│  ├─ Ca_RSH2.pt
│  ├─ Ca_RSH3.pt
│  ├─ Ca_RSH4.pt
│  ├─ Ca_RSH5.pt
│  ├─ Ca_RSH6.pt
│  ├─ Ca_benchmark.pt
│  ├─ log
│  │  ├─ Ca_OR_log.txt
│  │  ├─ Ca_RSH3_log.txt
│  │  ├─ Ca_RSH4_log.txt
│  │  ├─ Ca_RSH5_log.txt
│  │  ├─ Ca_RSH6_log.txt
│  │  ├─ Ca_benchmark_log.txt
│  │  └─ training_comparison.html
│  ├─ v_48_002.pt
│  ├─ v_48_010.pt
│  └─ v_48_020.pt
├─ dataset
│  ├─ stats
│  │  ├─ test
│  │  │  ├─ CATH_1_histogram.html
│  │  │  ├─ CATH_2_histogram.html
│  │  │  ├─ CATH_3_histogram.html
│  │  │  └─ sequence_length_histogram.html
│  │  ├─ train
│  │  │  ├─ CATH_1_histogram.html
│  │  │  ├─ CATH_2_histogram.html
│  │  │  ├─ CATH_3_histogram.html
│  │  │  └─ sequence_length_histogram.html
│  │  └─ valid
│  │     ├─ CATH_1_histogram.html
│  │     ├─ CATH_2_histogram.html
│  │     ├─ CATH_3_histogram.html
│  │     └─ sequence_length_histogram.html
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
│  ├─ .ipynb_checkpoints
│  │  ├─ SO3_invariant_representations-checkpoint.ipynb
│  │  └─ mpnn_wAF2-checkpoint.py
│  ├─ ProteinMPNN_EDA.ipynb
│  ├─ SO3_invariant_representations.ipynb
│  ├─ mpnn_wAF2.py
│  ├─ pipeline_demo.ipynb
│  └─ retrieve_pdb_domain.ipynb
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
├─ presentation
│  ├─ .ipynb_checkpoints
│  │  └─ Group_Theory-checkpoint.pdf
│  ├─ Group_Theory.pdf
│  ├─ vanilla_RSH4
│  │  ├─ output
│  │  │  └─ 1O91
│  │  │     ├─ out_seq_0_model_0.pdb
│  │  │     ├─ out_seq_1_model_0.pdb
│  │  │     ├─ out_seq_2_model_0.pdb
│  │  │     ├─ out_seq_3_model_0.pdb
│  │  │     ├─ out_seq_4_model_0.pdb
│  │  │     ├─ out_seq_5_model_0.pdb
│  │  │     ├─ out_seq_6_model_0.pdb
│  │  │     └─ out_seq_7_model_0.pdb
│  │  ├─ probs
│  │  │  └─ 1O91
│  │  │     ├─ 1O91.npz
│  │  │     ├─ log_probs.html
│  │  │     └─ probs.html
│  │  ├─ scores
│  │  │  └─ 1O91
│  │  │     └─ 1O91.npz
│  │  └─ seqs
│  │     └─ 1O91
│  │        └─ 1O91.fa
│  └─ weekly_notes.pdf
├─ protein_mpnn_run.py
├─ protein_mpnn_utils.py
├─ results
│  ├─ Ca_OR_histograms_CATH_1.html
│  ├─ Ca_OR_histograms_CATH_2.html
│  ├─ Ca_OR_histograms_CATH_3.html
│  ├─ Ca_OR_test_hetero_mpnn_result.csv
│  ├─ Ca_OR_test_homo_mpnn_result.csv
│  ├─ Ca_RSH3_histograms_CATH_1.html
│  ├─ Ca_RSH3_histograms_CATH_2.html
│  ├─ Ca_RSH3_histograms_CATH_3.html
│  ├─ Ca_RSH3_test_hetero_mpnn_result.csv
│  ├─ Ca_RSH3_test_homo_mpnn_result.csv
│  ├─ Ca_RSH4_histograms_CATH_1.html
│  ├─ Ca_RSH4_histograms_CATH_2.html
│  ├─ Ca_RSH4_histograms_CATH_3.html
│  ├─ Ca_RSH4_test_hetero_mpnn_result.csv
│  ├─ Ca_RSH4_test_homo_mpnn_result.csv
│  ├─ Ca_RSH5_histograms_CATH_1.html
│  ├─ Ca_RSH5_histograms_CATH_2.html
│  ├─ Ca_RSH5_histograms_CATH_3.html
│  ├─ Ca_RSH5_test_hetero_mpnn_result.csv
│  ├─ Ca_RSH5_test_homo_mpnn_result.csv
│  ├─ Ca_benchmark_histograms_CATH_1.html
│  ├─ Ca_benchmark_histograms_CATH_2.html
│  ├─ Ca_benchmark_histograms_CATH_3.html
│  ├─ Ca_benchmark_test_hetero_mpnn_result.csv
│  ├─ Ca_benchmark_test_homo_mpnn_result.csv
│  ├─ Ca_model_CATH_1_hist_comparison.html
│  ├─ Ca_model_CATH_1_table.html
│  ├─ Ca_model_CATH_1_violin_comparison.html
│  ├─ Ca_model_CATH_2_hist_comparison.html
│  ├─ Ca_model_CATH_2_table.html
│  ├─ Ca_model_CATH_2_violin_comparison.html
│  ├─ Ca_model_CATH_3_hist_comparison.html
│  ├─ Ca_model_CATH_3_table.html
│  ├─ Ca_model_CATH_3_violin_comparison.html
│  ├─ Ca_training_comparison.html
│  ├─ ProteinMPNN_EDA.ipynb
│  ├─ vanilla_OR_histograms_CATH_1.html
│  ├─ vanilla_OR_histograms_CATH_2.html
│  ├─ vanilla_OR_histograms_CATH_3.html
│  ├─ vanilla_OR_test_hetero_mpnn_result.csv
│  ├─ vanilla_OR_test_homo_mpnn_result.csv
│  ├─ vanilla_RSH3_histograms_CATH_1.html
│  ├─ vanilla_RSH3_histograms_CATH_2.html
│  ├─ vanilla_RSH3_histograms_CATH_3.html
│  ├─ vanilla_RSH3_test_hetero_mpnn_result.csv
│  ├─ vanilla_RSH3_test_homo_mpnn_result.csv
│  ├─ vanilla_RSH4_histograms_CATH_1.html
│  ├─ vanilla_RSH4_histograms_CATH_2.html
│  ├─ vanilla_RSH4_histograms_CATH_3.html
│  ├─ vanilla_RSH4_test_hetero_mpnn_result.csv
│  ├─ vanilla_RSH4_test_homo_mpnn_result.csv
│  ├─ vanilla_RSH5_histograms_CATH_1.html
│  ├─ vanilla_RSH5_histograms_CATH_2.html
│  ├─ vanilla_RSH5_histograms_CATH_3.html
│  ├─ vanilla_RSH5_test_hetero_mpnn_result.csv
│  ├─ vanilla_RSH5_test_homo_mpnn_result.csv
│  ├─ vanilla_benchmark_histograms_CATH_1.html
│  ├─ vanilla_benchmark_histograms_CATH_2.html
│  ├─ vanilla_benchmark_histograms_CATH_3.html
│  ├─ vanilla_benchmark_test_hetero_mpnn_result.csv
│  ├─ vanilla_benchmark_test_homo_mpnn_result.csv
│  ├─ vanilla_model_CATH_1_hist_comparison.html
│  ├─ vanilla_model_CATH_1_table.html
│  ├─ vanilla_model_CATH_1_violin_comparison.html
│  ├─ vanilla_model_CATH_2_hist_comparison.html
│  ├─ vanilla_model_CATH_2_table.html
│  ├─ vanilla_model_CATH_2_violin_comparison.html
│  ├─ vanilla_model_CATH_3_hist_comparison.html
│  ├─ vanilla_model_CATH_3_table.html
│  ├─ vanilla_model_CATH_3_violin_comparison.html
│  └─ vanilla_training_comparison.html
├─ training
│  ├─ .ipynb_checkpoints
│  │  ├─ README-checkpoint.md
│  │  ├─ generic_train-checkpoint.pbs
│  │  └─ generic_train-checkpoint.py
│  ├─ README.md
│  ├─ generic_model_utils.py
│  ├─ generic_outputs
│  │  ├─ generic_vanilla_RSH4.e7150422
│  │  ├─ generic_vanilla_RSH4.o7150422
│  │  ├─ generic_vanilla_RSH4_resumed.e7177597
│  │  └─ generic_vanilla_RSH4_resumed.o7177597
│  ├─ generic_train.pbs
│  ├─ generic_train.py
│  ├─ generic_train_resume.pbs
│  └─ utils.py
└─ vanilla_model_weights
   ├─ log
   │  ├─ training_comparison.html
   │  ├─ vanilla_OR_log.txt
   │  ├─ vanilla_RSH3_log.txt
   │  ├─ vanilla_RSH4_log.txt
   │  ├─ vanilla_RSH5_log.txt
   │  ├─ vanilla_RSH6_log.txt
   │  └─ vanilla_benchmark_log.txt
   ├─ v_48_002.pt
   ├─ v_48_010.pt
   ├─ v_48_020.pt
   ├─ v_48_030.pt
   ├─ vanilla_OR.pt
   ├─ vanilla_RSH1.pt
   ├─ vanilla_RSH2.pt
   ├─ vanilla_RSH3.pt
   ├─ vanilla_RSH4.pt
   ├─ vanilla_RSH5.pt
   ├─ vanilla_RSH6.pt
   └─ vanilla_benchmark.pt
```
 

## 🎯 RoadMap



## Future Work
Benchmarking more physics-inspired representation methods

## Acknowledgements
My appreciation goes to Professor Stefano Angioletti-Uberti and PhD researcher Shanil from <a href="https://www.softnanolab.org/">SoftNanoLab</a> for their attentive guidance and great access to computing resources.

## Contributing
If you have any questions or suggestions towards this repository, feel free to contact me at xy2119@ic.ac.uk.

Any kind of enhancement or contribution is welcomed!

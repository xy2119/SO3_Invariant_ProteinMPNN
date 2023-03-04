#@title Setup AlphaFold

# import libraries
from IPython.utils import io
import os,sys,re

if "af_backprop" not in sys.path:
  import tensorflow as tf
  import jax
  import jax.numpy as jnp
  import numpy as np
  import matplotlib
  from matplotlib import animation
  import matplotlib.pyplot as plt
  from IPython.display import HTML
  import tqdm.notebook
  TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'

  with io.capture_output() as captured:
    # install ALPHAFOLD
    if not os.path.isdir("af_backprop"):
      %shell git clone https://github.com/sokrypton/af_backprop.git
      %shell pip -q install biopython dm-haiku ml-collections py3Dmol
      %shell wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py
    if not os.path.isdir("params"):
      %shell mkdir params
      %shell curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar | tar x -C params

  if not os.path.exists("MMalign"):
    # install MMalign
    os.system("wget -qnc https://zhanggroup.org/MM-align/bin/module/MMalign.cpp")
    os.system("g++ -static -O3 -ffast-math -o MMalign MMalign.cpp")

  def mmalign(pdb_a,pdb_b):
    # pass to MMalign
    output = os.popen(f'./MMalign {pdb_a} {pdb_b}')
    # parse outputs
    parse_float = lambda x: float(x.split("=")[1].split()[0])
    tms = []
    for line in output:
      line = line.rstrip()
      if line.startswith("TM-score"): tms.append(parse_float(line))
    return tms

  # configure which device to use
  try:
    # check if TPU is available
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()
    print('Running on TPU')
    DEVICE = "tpu"
  except:
    if jax.local_devices()[0].platform == 'cpu':
      print("WARNING: no GPU detected, will be using CPU")
      DEVICE = "cpu"
    else:
      print('Running on GPU')
      DEVICE = "gpu"
      # disable GPU on tensorflow
      tf.config.set_visible_devices([], 'GPU')

  # import libraries
  sys.path.append('af_backprop')
  from utils import update_seq, update_aatype, get_plddt, get_pae
  import colabfold as cf
  from alphafold.common import protein as alphafold_protein
  from alphafold.data import pipeline
  from alphafold.model import data, config
  from alphafold.common import residue_constants
  from alphafold.model import model as alphafold_model

# custom functions
def clear_mem():
  backend = jax.lib.xla_bridge.get_backend()
  for buf in backend.live_buffers(): buf.delete()

def setup_model(max_len):
  clear_mem()

  # setup model
  cfg = config.model_config("model_3_ptm")
  cfg.model.num_recycle = 0
  cfg.data.common.num_recycle = 0
  cfg.data.eval.max_msa_clusters = 1
  cfg.data.common.max_extra_msa = 1
  cfg.data.eval.masked_msa_replace_fraction = 0
  cfg.model.global_config.subbatch_size = None

  # get params
  model_param = data.get_model_haiku_params(model_name="model_3_ptm", data_dir=".")
  model_runner = alphafold_model.RunModel(cfg, model_param, is_training=False, recycle_mode="none")

  model_params = []
  for k in [1,2,3,4,5]:
    if k == 3:
      model_params.append(model_param)
    else:
      params = data.get_model_haiku_params(model_name=f"model_{k}_ptm", data_dir=".")
      model_params.append({k: params[k] for k in model_runner.params.keys()})

  seq = "A" * max_len
  length = len(seq)
  feature_dict = {
      **pipeline.make_sequence_features(sequence=seq, description="none", num_res=length),
      **pipeline.make_msa_features(msas=[[seq]], deletion_matrices=[[[0]*length]])
  }
  inputs = model_runner.process_features(feature_dict,random_seed=0)

  def runner(I, params):
    # update sequence
    inputs = I["inputs"]
    inputs.update(I["prev"])

    seq = jax.nn.one_hot(I["seq"],20)
    update_seq(seq, inputs)
    update_aatype(inputs["target_feat"][...,1:], inputs)

    # mask prediction
    mask = jnp.arange(inputs["residue_index"].shape[0]) < I["length"]
    inputs["seq_mask"] = inputs["seq_mask"].at[:].set(mask)
    inputs["msa_mask"] = inputs["msa_mask"].at[:].set(mask)
    inputs["residue_index"] = jnp.where(mask, inputs["residue_index"], 0)

    # get prediction
    key = jax.random.PRNGKey(0)
    outputs = model_runner.apply(params, key, inputs)

    prev = {"init_msa_first_row":outputs['representations']['msa_first_row'][None],
            "init_pair":outputs['representations']['pair'][None],
            "init_pos":outputs['structure_module']['final_atom_positions'][None]}
    
    aux = {"final_atom_positions":outputs["structure_module"]["final_atom_positions"],
           "final_atom_mask":outputs["structure_module"]["final_atom_mask"],
           "plddt":get_plddt(outputs),"pae":get_pae(outputs),
           "length":I["length"], "seq":I["seq"], "prev":prev,
           "residue_idx":inputs["residue_index"][0]}
    return aux

  return jax.jit(runner), model_params, {"inputs":inputs, "length":max_length}

def save_pdb(outs, filename, Ls=None):
  '''save pdb coordinates'''
  p = {"residue_index":outs["residue_idx"] + 1,
       "aatype":outs["seq"],
       "atom_positions":outs["final_atom_positions"],
       "atom_mask":outs["final_atom_mask"],
       "plddt":outs["plddt"]}
  p = jax.tree_map(lambda x:x[:outs["length"]], p)
  b_factors = 100 * p.pop("plddt")[:,None] * p["atom_mask"]
  p = alphafold_protein.Protein(**p,b_factors=b_factors)
  pdb_lines = alphafold_protein.to_pdb(p)

  with open(filename, 'w') as f:
    f.write(pdb_lines)
  if Ls is not None:
    pdb_lines = cf.read_pdb_renum(filename, Ls)
    with open(filename, 'w') as f:
      f.write(pdb_lines)
 
 
    
    
#@title ### Run AlphaFold

num_models = 3 #@param ["1","2","3","4","5"] {type:"raw"}
num_recycles = 1 #@param ["0","1","2","3"] {type:"raw"}

outs = []
positions = []
plddts = []
paes = []
LS = []


out_folder='.'                    # Path to a folder to output sequences, e.g. /home/out/
# Build paths for experiment
base_folder = out_folder
if base_folder[-1] != '/':
    base_folder = base_folder + '/'

################################ AF2 Start #####################################

info_list=[]
print(f"mpnn_seq_num af2_model_num   avg_pLDDT avg_pAE TMscore")
for design in mpnn_result[['pdb_path','designed_sequence','designed_sequence_no']].itertuples():
    s = design.Index
    ori_sequence = design.designed_sequence
    ori_sequence_no = design.designed_sequence_no
    pdb_path = design.pdb_path
    name_= pdb_path.split("/")[-1][:-4]

    Ls = [len(s) for s in ori_sequence.replace(":","/").split("/")]
    LS.append(Ls)
    sequence = re.sub("[^A-Z]","",ori_sequence)
    length = len(sequence)

    # avoid recompiling if length within 25
    if "max_len" not in dir() or length > max_len or (max_len - length) > 25:
      max_len = length + 25
      runner, params, I = setup_model(max_len)

    outs.append([])
    positions.append([])
    plddts.append([])
    paes.append([])

    r = -1
    # pad sequence to max length
    seq = np.array([residue_constants.restype_order.get(aa,0) for aa in sequence])
    seq = np.pad(seq,[0,max_len-length],constant_values=-1)
    I["inputs"]['residue_index'][:] = cf.chain_break(np.arange(max_len), Ls, length=32)
    I.update({"seq":seq, "length":length})

    # for each model
    for n in range(num_models):
      # restart recycle
      I["prev"] = {'init_msa_first_row': np.zeros([1, max_len, 256]),
                  'init_pair': np.zeros([1, max_len, max_len, 128]),
                  'init_pos': np.zeros([1, max_len, 37, 3])}
      for r in range(num_recycles + 1):
        O = runner(I, params[n])
        O = jax.tree_map(lambda x:np.asarray(x), O)
        I["prev"] = O["prev"]
         
      
      positions[-1].append(O["final_atom_positions"][:length])
      plddts[-1].append(O["plddt"][:length])
      paes[-1].append(O["pae"][:length,:length])
      outs[-1].append(O)
      save_pdb(outs[-1][-1], base_path + f"/output/{name_}/out_seq_{s}_model_{n}.pdb", Ls=LS[-1])
      tmscores = mmalign(pdb_path, base_path + f"/output/{name_}/out_seq_{s}_model_{n}.pdb")
      print(f"   {s}  \t\t{n}\t\t{plddts[-1][-1].mean():.3} \t{paes[-1][-1].mean():.3} \t{tmscores[-1]:.3}")

    ################################ AF2 End #####################################

      info={'mpnn_index':s,
      'pdb_name':name_, 
      'designed_sequence_no':ori_sequence_no,
      'af2_model_no':n,
      'average_pLDDTs':plddts[-1][-1].mean(),
      'average_pAE':paes[-1][-1].mean(),
      'TMscore':tmscores[-1]}
      info_list.append(info)

      af2_result=pd.DataFrame(info_list)
      af2_result.to_csv(f'./results/{model_type}_af2_result.csv') 
      print('AF2 result saved!')
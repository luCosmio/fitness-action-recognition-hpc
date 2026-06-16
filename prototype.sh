#!/bin/bash
#SBATCH --job-name=cv_pipeline
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu          # Sostituisci con la partizione GPU del tuo cluster
#SBATCH --gres=gpu:1             # Richiede 1 GPU
#SBATCH --cpus-per-task=8        # Richiede 8 CPU per dataloader e unzip veloce
#SBATCH --mem=32G                # RAM di sistema
#SBATCH --time=12:00:00          # Walltime massimo

# ==========================================
# 1. SETUP AMBIENTE E VARIABILI HPC
# ==========================================
module purge
module load cuda/12.2

# Pathing Assoluto
HOME_PROJ_DIR="$HOME/projects/project_work_cv"
export SCRATCH_WORKSPACE="$SCRATCH/project_work_cv_$SLURM_JOB_ID"

# Inizializzazione Virtual Environment
source "$HOME_PROJ_DIR/venv3.12.11/bin/activate"

echo "=== Slurm Job ID: $SLURM_JOB_ID ==="
echo "Node: $SLURMD_NODENAME | CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Workspace: $SCRATCH_WORKSPACE"

# Creazione dell'albero di directory volatile nello Scratch
mkdir -p "$SCRATCH_WORKSPACE/dataset"
mkdir -p "$SCRATCH_WORKSPACE/models"
mkdir -p "$SCRATCH_WORKSPACE/inputs"
mkdir -p "$SCRATCH_WORKSPACE/outputs"

# ==========================================
# 2. PRE-COMPUTE I/O (HOME -> SCRATCH)
# ==========================================
echo "[$(date +'%H:%M:%S')] Staging I/O: Copia e decompressione dataset..."
unzip -q "$HOME_PROJ_DIR/dataset.zip" -d "$SCRATCH_WORKSPACE/dataset/"

# (Opzionale) Copia pesi offline se usi R-CNN senza internet
cp "$HOME_PROJ_DIR/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth" "$SCRATCH_WORKSPACE/models/"

# Copia i file da analizzare per il Task 4 (Inference)
cp -r "$HOME_PROJ_DIR/inputs/"* "$SCRATCH_WORKSPACE/inputs/" 2>/dev/null || true

# ==========================================
# 3. ESECUZIONE PYTHON SUL NODO GPU
# ==========================================
echo "[$(date +'%H:%M:%S')] Innesco Python CLI..."

# Cambia il flag --task a "diagnostics", "extract", "train", "inference", o "all"
python "$HOME_PROJ_DIR/main.py" \
    --task diagnostics \
    --feature-tag "seq_30_skip_2_stride_dyn" \
    --model-tag "bilstm_v2"

# ==========================================
# 4. POST-COMPUTE I/O (SCRATCH -> HOME)
# ==========================================
echo "[$(date +'%H:%M:%S')] Sincronizzazione Artefatti (Scratch -> Home)..."

# Raccoglie i modelli salvati
mkdir -p "$HOME_PROJ_DIR/models"
cp "$SCRATCH_WORKSPACE/models/"*.pth "$HOME_PROJ_DIR/models/" 2>/dev/null || true

# Raccoglie output (video, CSV, plot, log)
mkdir -p "$HOME_PROJ_DIR/outputs/run_$SLURM_JOB_ID"
cp -r "$SCRATCH_WORKSPACE/outputs/"* "$HOME_PROJ_DIR/outputs/run_$SLURM_JOB_ID/"

# Comprime e salva le feature estratte per futuri addestramenti
echo "[$(date +'%H:%M:%S')] Archiviazione Tensori PyTorch..."
cd "$SCRATCH_WORKSPACE" || exit
zip -rq "$HOME_PROJ_DIR/features_seq_30_skip_2_stride_dyn.zip" "features_seq_30_skip_2_stride_dyn/"

echo "[$(date +'%H:%M:%S')] Job Completato. Output salvato in: $HOME_PROJ_DIR/outputs/run_$SLURM_JOB_ID"
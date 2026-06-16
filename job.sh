#!/bin/bash
#SBATCH --job-name=cv_pipeline
#SBATCH --output=/hpc/home/%u/projects/project_work_cv/logs/%j_%x.log
#SBATCH --partition=gpuResB
#SBATCH --qos=gpuResB_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00   # Walltime (Max: 1-00:00:00 su gpuSlim)

# ==========================================
# 0. PIPELINE CONFIGURATION
# ==========================================
# --- Pipeline Hyperparameters ---
TASK_ID="diagnostics"   # Options: diagnostics, extract, train, inference, all
FEATURE_TAG="seq_30_skip_2_stride_dyn"
MODEL_TAG="bilstm_v1"
BATCH_SIZE=64
LSTM_BATCH_SIZE=64

# --- Pathing & File Naming ---
PROJECT_NAME="project_work_cv"
VENV_NAME=".venv"
DATASET_NAME="dataset.zip"
PYTHON_SCRIPT="pipeline.py"

# --- Derived Variables (no edit) ---
FEATURE_DIR_NAME="features_${FEATURE_TAG}"
FEATURE_ZIP_NAME="${FEATURE_DIR_NAME}.zip"
HOME_PROJ_DIR="/hpc/home/$USER/projects/$PROJECT_NAME"
export SCRATCH_WORKSPACE="/tmp/$USER/job_$SLURM_JOB_ID"
mkdir -p "$HOME_PROJ_DIR/logs"

# --- Bash Logging Function ---
hpc_log() {
    echo "[$(date +'%H:%M:%S')] [HPC_ORCHESTRATOR] $1"
}

# ==========================================
# 1. ENVIRONMENT SETUP & SIGNAL HANDLING
# ==========================================
# Catch SIGTERM sent by SLURM on graceTime (15 minutes)
trap 'hpc_log "RICEVUTO SIGTERM (Fine Tempo o Preemption). Avvio salvataggio di emergenza!"; emergency_sync; exit 143' SIGTERM

module purge
module load cuda/12.2

# Initialize Virtual Environment
source "$HOME_PROJ_DIR/$VENV_NAME/bin/activate"

hpc_log "=== STAGE 1: RESOURCE ALLOCATION ==="
hpc_log "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | CPUs: $SLURM_CPUS_PER_TASK"
hpc_log "Scratch Workspace: $SCRATCH_WORKSPACE"
hpc_log "Remote Debug Path: /scratchnet/$SLURMD_NODENAME/$USER/job_$SLURM_JOB_ID"

export PYTHONUNBUFFERED=1

# Creating directory structure on local node (Scratch)
mkdir -p "$SCRATCH_WORKSPACE/dataset"
mkdir -p "$SCRATCH_WORKSPACE/models"
mkdir -p "$SCRATCH_WORKSPACE/inputs"
mkdir -p "$SCRATCH_WORKSPACE/outputs"

# Syncing function: Scratch -> Home
sync_to_home() {
    hpc_log "Artifacts sync (Scratch -> Home)..."

    # Export trained model weights
    mkdir -p "$HOME_PROJ_DIR/models"
    cp "$SCRATCH_WORKSPACE/models/"*.pth "$HOME_PROJ_DIR/models/" 2>/dev/null || true

    # Export outputs (video, CSV, plot, log)
    mkdir -p "$HOME_PROJ_DIR/outputs/run_${SLURM_JOB_ID}_${TASK_ID}"
    cp -r "$SCRATCH_WORKSPACE/outputs/"* "$HOME_PROJ_DIR/outputs/run_${SLURM_JOB_ID}_${TASK_ID}/" 2>/dev/null || true

    # Export extracted features (zipped)
    if [[ "$TASK_ID" == "extract" || "$TASK_ID" == "all" ]]; then
        hpc_log "Archiviazione Tensori PyTorch..."
        cd "$SCRATCH_WORKSPACE" || exit
        if [ -d "$FEATURE_DIR_NAME" ]; then
            zip -rq "$HOME_PROJ_DIR/features/$FEATURE_ZIP_NAME" "$FEATURE_DIR_NAME/"
        fi
    fi
}

emergency_sync() {
    hpc_log "!!! EMERGENCY SYNC !!!"
    # Kill any running Python processes to free up resources for sync
    pkill -TERM -P $$ python
    sleep 2
    sync_to_home
    hpc_log "Emergency sync completed. Exiting job."
}

# ==========================================
# 2. PRE-COMPUTE I/O (HOME -> SCRATCH)
# ==========================================
hpc_log "=== STAGE 2: PRE-COMPUTE I/O ==="

hpc_log "Staging pre-trained models..."
cp -r "$HOME_PROJ_DIR/models/"* "$SCRATCH_WORKSPACE/models/" 2>/dev/null || true

hpc_log "Staging dataset..."
if [ -f "$HOME_PROJ_DIR/$DATASET_NAME" ]; then
    unzip -q "$HOME_PROJ_DIR/$DATASET_NAME" -d "$SCRATCH_WORKSPACE/dataset/"
else
    hpc_log "WARNING: $DATASET_NAME not found in $HOME_PROJ_DIR. Feature extraction will fail."
fi

# Copy files needed for inference (e.g., videos, CSVs, etc.) to the scratch workspace
cp -r "$HOME_PROJ_DIR/inputs/"* "$SCRATCH_WORKSPACE/inputs/" 2>/dev/null || true

# If pre-extracted features exist, copy them to scratch (unzipped) for faster access during inference/training
if [ -f "$HOME_PROJ_DIR/features/$FEATURE_ZIP_NAME" ]; then
    hpc_log "Staging I/O: Unzipping pre-calculated features ($FEATURE_ZIP_NAME)..."
    unzip -q "$HOME_PROJ_DIR/features/$FEATURE_ZIP_NAME" -d "$SCRATCH_WORKSPACE/"
fi

# ==========================================
# 3. PYTHON EXECUTION ON GPU
# ==========================================
hpc_log "=== STAGE 3: PYTHON EXECUTION ==="
hpc_log "Starting task: $TASK_ID..."
hpc_log "-------------------------------------------------"

python "$HOME_PROJ_DIR/$PYTHON_SCRIPT" \
    --task "$TASK_ID" \
    --feature-tag "$FEATURE_TAG" \
    --model-tag "$MODEL_TAG" \
    --batch-size "$BATCH_SIZE" \
    --lstm-batch-size "$LSTM_BATCH_SIZE"

EXIT_CODE=$?
hpc_log "-------------------------------------------------"

if [ $EXIT_CODE -ne 0 ]; then
    hpc_log "CRITICAL ERROR: Python returned exit code $EXIT_CODE."
else
    hpc_log "Python pipeline completed successfully (Exit 0)."
fi

# ==========================================
# 4. POST-COMPUTE I/O & CLEANUP
# ==========================================
hpc_log "=== STAGE 4: POST-COMPUTE & TEARDOWN ==="
sync_to_home

hpc_log "Cleaning Scratch Area ($SCRATCH_WORKSPACE)..."
rm -rf "$SCRATCH_WORKSPACE"
hpc_log "Job Completed."
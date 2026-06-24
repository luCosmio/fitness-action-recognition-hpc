import os
import glob
import copy
import csv
import sys
import random
import logging
import queue
import threading
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from abc import ABC, abstractmethod
from collections import deque, Counter
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger("FitnessTracker")


def set_deterministic_environment(seed: int = 42) -> None:
    """
    Configures the environment for deterministic execution to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    # --- DYNAMIC INJECTIONS (Managed by CLI args via job.sh) ---
    FEATURE_TAG = "default_tag"
    MODEL_TAG = "default_model"
    BATCH_SIZE = 16
    LSTM_BATCH_SIZE = 16

    # --- HPC PATHS STRATEGY ---
    WORKSPACE_DIR = os.getenv("SCRATCH_WORKSPACE", "/tmp/project_work_cv")
    DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
    MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
    OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "outputs")
    INPUT_DIR = os.path.join(WORKSPACE_DIR, "inputs")

    # --- STATIC DATA PATHS ---
    OFFLINE_RCNN_WEIGHTS = os.path.join(
        MODELS_DIR, "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
    )
    REAL_DIR = os.path.join(DATASET_DIR, "final_kaggle_with_additional_video")
    SYNTH_DIR = os.path.join(DATASET_DIR, "synthetic_dataset", "synthetic_dataset")

    # --- DYNAMIC PATHS (Dependent on FEATURE_TAG & MODEL_TAG) ---
    @classmethod
    def get_features_dir(cls):
        return os.path.join(cls.WORKSPACE_DIR, f"features_{cls.FEATURE_TAG}")

    @classmethod
    def get_weights_path(cls):
        return os.path.join(cls.MODELS_DIR, f"{cls.MODEL_TAG}.pth")

    # --- TASK HYPERPARAMETERS & PIPELINE ---
    TARGET_CLASSES = {
        "squat": 0,
        "push-up": 1,
        "barbell biceps curl": 2,
        "shoulder press": 3,
    }
    CLASSES_UI = ["squat", "push-up", "bicep curl", "shoulder press"]
    SEQ_LENGTH = 30
    SEQ_STRIDE = 10
    BATCH_SIZE = 16
    FRAME_SKIP = 2
    MAX_DIM = 640
    UI_BASE_WIDTH = 1920

    # --- DATASET AUGMENTATION HYPERPARAMETERS ---
    SYNTHETIC_IDENTIFIER = "_syn_"
    SYNTHETIC_STRIDE = 10
    SYNTHETIC_MAX_SEQ = 10
    # Organic videos balancing
    # {class_idx: divisor}. No class in dict => divisor=1 (no boost)
    ORGANIC_STRIDE_DIVISORS = {1: 3, 2: 4}
    MAX_DROPOUT_RATIO = 0.20

    # --- TRAINING PARAMETERS ---
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    SCHEDULER_PATIENCE = 2
    SCHEDULER_FACTOR = 0.5

    # --- HARDWARE & INFERENCE HYPERPARAMETERS ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIN_DETECTION_CONFIDENCE = 0.5

    # --- BILSTM HYPERPARAMETERS ---
    LSTM_BATCH_SIZE = 16
    LSTM_INPUT_SIZE = 8
    LSTM_HIDDEN_SIZE = 32
    LSTM_NUM_LAYERS = 1
    LSTM_NUM_CLASSES = len(TARGET_CLASSES)
    LSTM_FC_HIDDEN_SIZE = 16
    LSTM_DROPOUT_RATE = 0.5
    STREAM_CONFIDENCE_THRESHOLD = 0.55
    # --- NORMALIZATION STRATEGY ---
    # Indices 0-8 (Angles): Max 180.0
    # Index 9 (BBox Ratio): Estimated Max 3.0
    # FEATURE_SCALERS = [1.0 / 180.0] * 9 + [1.0 / 3.0]

    # --- FSM CONSTANTS ---
    FPS = 30
    USER_PAUSE = 1.5
    VOTING_FRAMES = 20
    ROM_THRESHOLD = 35.0
    MUTEX_TIMEOUT_FRAMES = VOTING_FRAMES + (USER_PAUSE * FPS)

    # --- KALMAN FILTER HYPERPARAMETERS ---
    KALMAN_PROCESS_VAR = 1e-3  # 1e-3
    KALMAN_MEASURE_VAR = 0.1  # 0.1
    NUM_KEYPOINTS = 17
    NUM_KALMAN_FILTERS = NUM_KEYPOINTS * 2

    # --- COCO KEYPOINTS MAPPING ---
    # Torchvision Keypoint R-CNN
    COCO = {
        "L_SHOULDER": 5,
        "R_SHOULDER": 6,
        "L_ELBOW": 7,
        "R_ELBOW": 8,
        "L_WRIST": 9,
        "R_WRIST": 10,
        "L_HIP": 11,
        "R_HIP": 12,
        "L_KNEE": 13,
        "R_KNEE": 14,
        "L_ANKLE": 15,
        "R_ANKLE": 16,
    }

    @classmethod
    def setup_environment(cls) -> logging.Logger:
        """Creates required directories within the active workspace."""
        dirs_to_create = [
            cls.WORKSPACE_DIR,
            cls.DATASET_DIR,
            cls.get_features_dir(),
            cls.MODELS_DIR,
            cls.OUTPUT_DIR,
            cls.INPUT_DIR,
        ]
        for d in dirs_to_create:
            os.makedirs(d, exist_ok=True)

        log_file_path = os.path.join(cls.OUTPUT_DIR, "pipeline_batch.log")

        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        root_logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s", datefmt="%H:%M:%S"
        )

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_file_path, mode="w", delay=False)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        logger = logging.getLogger("FitnessTracker")
        logger.info(
            f"Compute Environment initialized. Target workspace: {cls.WORKSPACE_DIR}"
        )
        return logger


# CELL 3. BACKEND BUSINESS LOGIC
class KalmanFilter:
    """Standard 1D Kalman filter implementation for angle smoothing."""

    def __init__(
        self,
        process_variance=Config.KALMAN_PROCESS_VAR,
        measurement_variance=Config.KALMAN_MEASURE_VAR,
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement = 0.0
        self.error_covariance = 1.0

    def update(self, measurement):
        prediction_error_covariance = self.error_covariance + self.process_variance
        kalman_gain = prediction_error_covariance / (
            prediction_error_covariance + self.measurement_variance
        )
        self.estimated_measurement = self.estimated_measurement + kalman_gain * (
            measurement - self.estimated_measurement
        )
        self.error_covariance = (1 - kalman_gain) * prediction_error_covariance
        return self.estimated_measurement


class PoseUtils:
    """Utility class for geometric calculations and keypoint extraction."""

    @staticmethod
    def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculates the angle in degrees between points a, b, c (b is the vertex)."""
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return float(np.degrees(angle))

    @staticmethod
    def calculate_absolute_y_angle(a: np.ndarray, b: np.ndarray) -> float:
        """Calculates the angle in degrees of vector ab relative to the vertical Y-axis."""
        a, b = np.array(a), np.array(b)
        ba = a - b
        norm = np.linalg.norm(ba)
        if norm < 1e-6:
            return 0.0
        cosine_angle = ba[1] / norm
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return float(np.degrees(angle))

    @staticmethod
    def calculate_bbox_ratio(keypoints_dict: dict) -> float:
        """Calculates W/H ratio from active keypoints."""
        xs = [pt[0] for pt in keypoints_dict.values()]
        ys = [pt[1] for pt in keypoints_dict.values()]
        if not xs or not ys:
            return 1.0
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        return float(w / (h + 1e-6))

    @staticmethod
    def extract_key_angles(
        keypoints: np.ndarray, kalman_filters: list, return_raw: bool = False
    ):
        """Extracts 8 relative joint angles (View-Invariant) for kinematic analysis."""
        filtered = {}
        raw = {}
        # filters and updates keypoints in Config.COCO
        for name, idx in Config.COCO.items():
            x, y, v = keypoints[idx]
            kf_x = kalman_filters[idx * 2]
            kf_y = kalman_filters[idx * 2 + 1]

            if return_raw:
                raw[name] = np.array([x, y])

            if v > 0.5:
                # keypoint is visible: updates filter and stores value
                filtered[name] = np.array([kf_x.update(x), kf_y.update(y)])
            else:
                # Dead Reckoning: uses last measurement
                filtered[name] = np.array(
                    [kf_x.estimated_measurement, kf_y.estimated_measurement]
                )

        def _calc_angles(d: dict) -> list:

            return [
                PoseUtils.calculate_angle(
                    d["R_SHOULDER"], d["R_ELBOW"], d["R_WRIST"]
                ),  # 0: Right Elbow
                PoseUtils.calculate_angle(
                    d["L_SHOULDER"], d["L_ELBOW"], d["L_WRIST"]
                ),  # 1: Left Elbow
                PoseUtils.calculate_angle(
                    d["R_HIP"], d["R_SHOULDER"], d["R_ELBOW"]
                ),  # 2: Right Shoulder
                PoseUtils.calculate_angle(
                    d["L_HIP"], d["L_SHOULDER"], d["L_ELBOW"]
                ),  # 3: Left Shoulder
                PoseUtils.calculate_angle(
                    d["R_SHOULDER"], d["R_HIP"], d["R_KNEE"]
                ),  # 4: Right Hip
                PoseUtils.calculate_angle(
                    d["L_SHOULDER"], d["L_HIP"], d["L_KNEE"]
                ),  # 5: Left Hip
                PoseUtils.calculate_angle(
                    d["R_HIP"], d["R_KNEE"], d["R_ANKLE"]
                ),  # 6: Right Knee
                PoseUtils.calculate_angle(
                    d["L_HIP"], d["L_KNEE"], d["L_ANKLE"]
                ),  # 7: Left Knee
            ]

        angles_filtered = _calc_angles(filtered)
        if return_raw:
            return angles_filtered, _calc_angles(raw)
        return angles_filtered

    @staticmethod
    def extract_raw_key_angles(
        keypoints: np.ndarray, kalman_filters=None, return_raw: bool = False
    ):
        """Extracts the 10 features directly, bypassing Kalman filtering."""
        raw = {}
        for name, idx in Config.COCO.items():
            x, y, v = keypoints[idx]
            raw[name] = np.array([x, y])

        def _calc_features(d: dict) -> list:
            torso_r = PoseUtils.calculate_absolute_y_angle(d["R_SHOULDER"], d["R_HIP"])
            torso_l = PoseUtils.calculate_absolute_y_angle(d["L_SHOULDER"], d["L_HIP"])

            return [
                PoseUtils.calculate_angle(d["R_SHOULDER"], d["R_ELBOW"], d["R_WRIST"]),
                PoseUtils.calculate_angle(d["L_SHOULDER"], d["L_ELBOW"], d["L_WRIST"]),
                PoseUtils.calculate_angle(d["R_HIP"], d["R_SHOULDER"], d["R_ELBOW"]),
                PoseUtils.calculate_angle(d["L_HIP"], d["L_SHOULDER"], d["L_ELBOW"]),
                PoseUtils.calculate_angle(d["R_SHOULDER"], d["R_HIP"], d["R_KNEE"]),
                PoseUtils.calculate_angle(d["L_SHOULDER"], d["L_HIP"], d["L_KNEE"]),
                PoseUtils.calculate_angle(d["R_HIP"], d["R_KNEE"], d["R_ANKLE"]),
                PoseUtils.calculate_angle(d["L_HIP"], d["L_KNEE"], d["L_ANKLE"]),
                (torso_r + torso_l) / 2.0,
                PoseUtils.calculate_bbox_ratio(d),
            ]

        features_raw = _calc_features(raw)
        if return_raw:
            return features_raw, features_raw
        return features_raw


class FitnessSequenceDataset(Dataset):
    """Loader for pre-extracted temporal features."""

    def __init__(self, features_dir: str):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(features_dir, "*.pt")))
        if not self.files:
            logger.warning(
                f"No .pt files found in {features_dir}. Feature extraction required."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return torch.load(self.files[idx], weights_only=True)


class DataCoordinator:
    """Handles dataset splitting to prevent data leakage across video IDs."""

    @staticmethod
    def group_split(dataset: FitnessSequenceDataset, train_ratio: float = 0.8):
        class_video_groups = {}

        for idx, file_path in enumerate(dataset.files):
            filename = os.path.basename(file_path)
            video_id = filename.split("_seq")[0]

            if "_org_" in video_id:
                class_prefix = f"{video_id.split('_org_')[0]}_org"
            elif "_syn_" in video_id:
                class_prefix = f"{video_id.split('_syn_')[0]}_syn"
            else:
                class_prefix = "unknown"

            if class_prefix not in class_video_groups:
                class_video_groups[class_prefix] = {}

            class_video_groups[class_prefix].setdefault(video_id, []).append(idx)

        train_indices = []
        val_indices = []

        for cls, video_groups in class_video_groups.items():
            unique_video_ids = sorted(list(video_groups.keys()))
            random.shuffle(unique_video_ids)

            split_point = int(len(unique_video_ids) * train_ratio)
            train_vids = unique_video_ids[:split_point]
            val_vids = unique_video_ids[split_point:]

            for vid in train_vids:
                train_indices.extend(video_groups[vid])
            for vid in val_vids:
                val_indices.extend(video_groups[vid])

        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    @staticmethod
    def get_loaders(features_dir: str, batch_size: Optional[int] = None):

        b_size = batch_size if batch_size is not None else Config.LSTM_BATCH_SIZE
        try:
            num_workers = int(os.getenv("SLURM_CPUS_PER_TASK", 2))
        except ValueError:
            num_workers = 2

        dataset = FitnessSequenceDataset(features_dir)
        if len(dataset) == 0:
            return None, None

        train_subset, val_subset = DataCoordinator.group_split(dataset)

        train_loader = DataLoader(
            train_subset,
            batch_size=b_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=b_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        logger.info(
            f"Dataloaders ready. Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Workers: {num_workers}"
        )
        return train_loader, val_loader


# CELL 4. BACKEND MODELS LOGIC
class PoseEstimator:
    """Encapsulates the Keypoint R-CNN model and single frame inference."""

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        logger.info(f"Loading Keypoint R-CNN on {self.device}...")
        self.model = keypointrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, progress=False
        )
        try:
            state_dict = torch.load(
                self.config.OFFLINE_RCNN_WEIGHTS,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            logger.info("Local Keypoint R-CNN weights loaded successfully.")
        except FileNotFoundError:
            logger.error(
                f"CRITICAL: Offline R-CNN weights not found at {self.config.OFFLINE_RCNN_WEIGHTS}"
            )
            logger.error(
                "Please download them via wget and stage them in the MODELS_DIR before running."
            )
            sys.exit(1)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        logger.info("Keypoint R-CNN ready.")

    @torch.no_grad()
    def predict(self, frame):
        """
        Performs inference on a single frame.
        Returns keypoints, scores, boxes, and highest detection confidence.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        predictions = self.model(tensor)

        if len(predictions[0]["scores"]) == 0:
            return None, None, None, 0.0

        best_idx = torch.argmax(predictions[0]["scores"]).item()
        confidence = predictions[0]["scores"][best_idx].cpu().item()

        if confidence < self.config.MIN_DETECTION_CONFIDENCE:
            return None, None, None, confidence

        keypoints = predictions[0]["keypoints"][best_idx].cpu().numpy()
        scores = predictions[0]["keypoints_scores"][best_idx].cpu().numpy()
        boxes = predictions[0]["boxes"][best_idx].cpu().numpy()
        return keypoints, scores, boxes, confidence

    @torch.no_grad()
    def predict_batch(self, frame_batch: List[np.ndarray]):
        """
        Batch inference for higher throughput.
        Returns a list of tuples: (keypoints, scores, boxes, confidence)
        Invalid detections return None for arrays to maintain temporal indexing.
        """
        if not frame_batch:
            return []
        tensors = torch.stack(
            [self.transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frame_batch]  # type: ignore
        ).to(self.device)

        predictions = self.model(tensors)

        results = []
        for pred in predictions:
            if len(pred["scores"]) == 0:
                results.append((None, None, None, 0.0))
                continue

            best_idx = torch.argmax(pred["scores"]).item()
            confidence = pred["scores"][best_idx].cpu().item()

            if confidence < self.config.MIN_DETECTION_CONFIDENCE:
                results.append((None, None, None, confidence))
                continue

            keypoints = pred["keypoints"][best_idx].cpu().numpy()
            scores = pred["keypoints_scores"][best_idx].cpu().numpy()
            boxes = pred["boxes"][best_idx].cpu().numpy()
            results.append((keypoints, scores, boxes, confidence))

        return results


class FitnessClassifier(nn.Module):
    """
    BiLSTM Temporal Classifier configurable via Config class.
    """

    def __init__(self, config):
        super().__init__()
        logger.info(f"Initializing FitnessClassifier model on {config.DEVICE}...")

        # self.register_buffer(
        #    "scale_mask", torch.tensor(config.FEATURE_SCALERS, dtype=torch.float32)
        # )
        lstm_dropout = config.LSTM_DROPOUT_RATE if config.LSTM_NUM_LAYERS > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.LSTM_INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.fc1 = nn.Linear(config.LSTM_HIDDEN_SIZE * 2, config.LSTM_FC_HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.LSTM_DROPOUT_RATE)
        self.out = nn.Linear(config.LSTM_FC_HIDDEN_SIZE, config.LSTM_NUM_CLASSES)

        self.default_threshold = config.STREAM_CONFIDENCE_THRESHOLD
        self.weights_loaded = False
        logger.info("FitnessClassifier model ready.")

    def forward(self, x):
        # Broadcasting: (Batch, Seq, Features) * (Features,) -> Scaled Tensor
        # x = x * self.scale_mask
        # x = x / 180.0  # Normalization for angles (0-180)
        _, (hn, _) = self.lstm(x)

        # hn shape: (num_layers * num_directions, batch, hidden_size)
        # Extracts final step from the top layer of both forward and backward directions
        hidden = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)

        x = self.dropout(self.relu(self.fc1(hidden)))
        return self.out(x)

    def load_weights(
        self, filepath: str, device: torch.device, strict_match: bool = True
    ):
        """
        Loads pre-trained weights into the model.
        """
        try:
            state_dict = torch.load(filepath, map_location=device, weights_only=True)
            self.load_state_dict(state_dict, strict=strict_match)
            self.to(device)
            self.weights_loaded = True
            logger.info(f"Model weights successfully loaded from {filepath}")
        except RuntimeError as e:
            logger.error(
                f"Topology mismatch during weight loading. Check Config.LSTM_NUM_LAYERS vs checkpoint. Error: {e}"
            )
            raise
        except FileNotFoundError:
            logger.error(f"Weights file not found at {filepath}")
            raise

    def predict_batch(self, batch_tensor: torch.Tensor) -> tuple:
        """
        Optimized inference for batch testing (DataLoader).
        Input: Tensor on device of shape (LSTM_BATCH_SIZE, SEQ_LENGTH, FEATURES).
        Returns: Tuple of 1D tensors (predicted_indices, confidences).
        """
        if not self.weights_loaded:
            logger.warning("Running batch inference with uninitialized weights.")

        self.eval()
        with torch.no_grad():
            outputs = self(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted_indices = torch.max(probabilities, 1)

        return predicted_indices, confidences

    def predict_stream(
        self, sequence_array, device: torch.device, threshold: Optional[float] = None
    ) -> tuple:
        """
        Fallback inference for real-time processing or FSM logic.
        Input: Numpy array or list of shape (SEQ_LENGTH, FEATURES).
        Returns: Tuple (predicted_class_index, confidence_value).
        """
        eval_threshold = threshold if threshold is not None else self.default_threshold
        if not self.weights_loaded:
            logger.warning("Running stream inference with uninitialized weights.")

        # as_tensor prevents memory copy if input is already a compatible ndarray
        tensor_input = torch.as_tensor(
            sequence_array, dtype=torch.float32, device=device
        ).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            output = self(tensor_input)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            # Export the full probability distribution for downstream tasks
            probs_array = probabilities.cpu().numpy()[0]

        conf_val = confidence.item()
        idx_val = predicted_idx.item()

        # Drop prediction if confidence is below safety threshold
        if conf_val < eval_threshold:
            return -1, conf_val, probs_array

        return idx_val, conf_val, probs_array


# CELL 5. BACKEND FSM & TRACKING
class TemporalSmoother:
    """Applies Confidence-Weighted Voting to smooth predictions over time."""

    def __init__(self, window_size: int):
        self.window = deque(maxlen=window_size)

    def update(self, current_prediction: int, confidence: float) -> int:
        self.window.append((current_prediction, confidence))
        votes = {}
        for pred, conf in self.window:
            votes[pred] = votes.get(pred, 0.0) + conf

        if not votes:
            return -1

        return max(votes.items(), key=lambda x: x[1])[0]


class ExerciseTracker(ABC):
    """Abstract Base Class for Finite State Machine kinematic tracking."""

    def __init__(self, rom_threshold: float):
        self.state = 0
        self.count = 0
        self.max_val = 0.0
        self.min_val = 180.0
        self.rom_threshold = rom_threshold

    @abstractmethod
    def update(self, angle: float) -> bool:
        pass

    def reset(self, angle: float):
        self.state = 0
        self.max_val = angle
        self.min_val = angle


class FlexionTracker(ExerciseTracker):
    """
    Kinematics: Starts extended.
    State 0 -> 1 triggered by angle decrease (eccentric).
    State 1 -> 0 triggered by angle increase (concentric) -> Repetition +1.
    """

    def update(self, angle: float) -> bool:
        self.max_val = max(self.max_val, angle)
        self.min_val = min(self.min_val, angle)

        if self.state == 0:
            if angle < self.max_val - self.rom_threshold:
                self.state = 1
                self.min_val = angle
        elif self.state == 1:
            if angle > self.min_val + self.rom_threshold:
                self.state = 0
                self.count += 1
                self.max_val = angle
                return True
        return False


class ExtensionTracker(ExerciseTracker):
    """
    Kinematics: Starts flexed.
    State 0 -> 1 triggered by angle increase (concentric).
    State 1 -> 0 triggered by angle decrease (eccentric) -> Repetition +1.
    """

    def update(self, angle: float) -> bool:
        self.max_val = max(self.max_val, angle)
        self.min_val = min(self.min_val, angle)

        if self.state == 0:
            if angle > self.min_val + self.rom_threshold:
                self.state = 1
                self.max_val = angle
        elif self.state == 1:
            if angle < self.max_val - self.rom_threshold:
                self.state = 0
                self.count += 1
                self.min_val = angle
                return True
        return False


class TrackerManager:
    """
    Orchestrates predictions, applies temporal smoothing, handles mutual exclusion (Mutex),
    and routes angular data to the correct specific ExerciseTracker.
    """

    def __init__(self, config):
        self.smoother = TemporalSmoother(config.VOTING_FRAMES)
        self.active_lock = -1  # -1 indicates NO active exercise
        self.lock_timeout_counter = 0
        self.timeout_limit = config.MUTEX_TIMEOUT_FRAMES
        self.rom_threshold = config.ROM_THRESHOLD

        self.trackers = {
            0: FlexionTracker(self.rom_threshold),  # Squat
            1: FlexionTracker(self.rom_threshold),  # Push-up
            2: FlexionTracker(self.rom_threshold),  # Bicep Curl
            3: ExtensionTracker(self.rom_threshold),  # Shoulder Press
        }

    def process_frame(
        self, raw_predicted_idx: int, confidence: float, angle: float
    ) -> tuple:
        """
        Processes a single frame's inference output.
        Returns: Tuple(Smoothed Prediction ID, Dictionary of current counts)
        """
        # 1. Temporal Smoothing
        smoothed_idx = self.smoother.update(raw_predicted_idx, confidence)

        # 2. Mutex Logic (Lock handling)
        if smoothed_idx != -1:
            if self.active_lock == -1:
                # Acquire lock
                self.active_lock = smoothed_idx
                self.lock_timeout_counter = 0
                self.trackers[self.active_lock].reset(angle)
            elif self.active_lock == smoothed_idx:
                # Maintain lock, reset timeout
                self.lock_timeout_counter = 0
            else:
                # Conflicting prediction while locked, increment timeout
                self.lock_timeout_counter += 1
                if self.lock_timeout_counter >= self.timeout_limit:
                    logger.debug(
                        f"Mutex timeout. Switching lock from {self.active_lock} to {smoothed_idx}"
                    )
                    self.active_lock = smoothed_idx
                    self.lock_timeout_counter = 0
                    self.trackers[self.active_lock].reset(angle)
        else:
            # Background/Noise detected
            if self.active_lock != -1:
                self.lock_timeout_counter += 1
                if self.lock_timeout_counter >= self.timeout_limit:
                    self.active_lock = -1
                    self.lock_timeout_counter = 0

        # 3. State Machine Update
        if self.active_lock != -1 and self.active_lock in self.trackers:
            self.trackers[self.active_lock].update(angle)

        counts = {k: v.count for k, v in self.trackers.items()}
        return smoothed_idx, counts


# CELL 6. BACKEND VISUAL ENGINE & UI
class DynamicUIRenderer:
    """
    Handles resolution-independent OpenCV rendering.
    Scales fonts, thicknesses, and paddings dynamically based on frame dimensions.
    """

    def __init__(self, config):
        self.config = config
        self.base_width = config.UI_BASE_WIDTH
        self.colors = {
            "bg": (40, 40, 40),
            "text_primary": (255, 255, 255),
            "text_accent": (0, 255, 0),
            "warning": (0, 0, 255),
            "skeleton_node": (0, 255, 255),
            "skeleton_edge": (255, 0, 255),
            "bbox": (0, 255, 0),
        }

    def _get_scale(self, frame_width: int) -> float:
        """Calculates scaling factor relative to standard 1080p width."""
        return max(0.5, frame_width / self.base_width)

    def draw_hud(
        self,
        frame: np.ndarray,
        exercise_name: str,
        count: int,
        confidence: float,
        state_info: str,
    ):
        """Renders the Heads-Up Display with metrics."""
        h, w = frame.shape[:2]
        scale = self._get_scale(w)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8 * scale
        thickness = max(1, int(2 * scale))
        padding = int(15 * scale)

        panel_w = int(400 * scale)
        panel_h = int(140 * scale)

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (padding, padding),
            (padding + panel_w, padding + panel_h),
            self.colors["bg"],
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        text_x = padding + int(10 * scale)
        text_y = padding + int(35 * scale)
        line_spacing = int(35 * scale)

        cv2.putText(
            frame,
            f"EX: {exercise_name}",
            (text_x, text_y),
            font,
            font_scale,
            self.colors["text_primary"],
            thickness,
        )
        cv2.putText(
            frame,
            f"REPS: {count}",
            (text_x, text_y + line_spacing),
            font,
            font_scale,
            self.colors["text_accent"],
            thickness,
        )

        conf_color = (
            self.colors["warning"] if confidence < 0.5 else self.colors["text_primary"]
        )
        cv2.putText(
            frame,
            f"CONF: {confidence:.2f} | ST: {state_info}",
            (text_x, text_y + line_spacing * 2),
            font,
            font_scale * 0.8,
            conf_color,
            thickness,
        )

    def draw_skeleton(
        self,
        frame: np.ndarray,
        keypoints: Optional[np.ndarray],
        scores: Optional[np.ndarray],
        edges: List[Tuple[int, int]],
        threshold: float = 0.5,
    ):
        """Draws COCO keypoints and connecting bones."""
        if keypoints is None or scores is None or len(keypoints) == 0:
            return

        scale = self._get_scale(frame.shape[1])
        radius = max(2, int(4 * scale))
        thickness = max(1, int(2 * scale))

        # Draw nodes
        for i, kpt in enumerate(keypoints):
            if i < len(scores) and scores[i] > threshold:
                # Spatial slicing: extracts only x and y, ignoring visibility
                x, y = kpt[:2]
                cv2.circle(
                    frame, (int(x), int(y)), radius, self.colors["skeleton_node"], -1
                )

        # Draw edges
        for p1, p2 in edges:
            if p1 < len(keypoints) and p2 < len(keypoints):
                if scores[p1] > threshold and scores[p2] > threshold:
                    # Spatial slicing for both edge coordinates
                    x1, y1 = keypoints[p1][:2]
                    x2, y2 = keypoints[p2][:2]
                    cv2.line(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        self.colors["skeleton_edge"],
                        thickness,
                    )

    def draw_bbox(self, frame: np.ndarray, bbox: Optional[np.ndarray]):
        """Draws bounding box around the subject."""
        if bbox is None or len(bbox) == 0:
            return

        scale = self._get_scale(frame.shape[1])
        thickness = max(1, int(2 * scale))
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors["bbox"], thickness)


class MetricsVisualizer:
    """
    Handles generation of purely matplotlib-based analytical plots.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Enforce a clean, standard styling
        plt.style.use("default")

    def plot_kalman_diagnostics(
        self, raw_signal: list, filtered_signal: list, filename: str
    ):

        plt.figure(figsize=(12, 4))
        plt.plot(
            raw_signal, label="Raw Angle", color="red", linewidth=2, alpha=0.6, zorder=1
        )
        plt.plot(
            filtered_signal,
            label="Filtered Angle (Kalman)",
            color="blue",
            linewidth=1,
            linestyle="--",
            zorder=2,
        )
        plt.title("Kalman Filter Calibration: Process vs Measure Variance")
        plt.xlabel("Frames")
        plt.ylabel("Angle (Degrees)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    def plot_training_curves(
        self,
        train_loss: List[float],
        val_loss: List[float],
        train_acc: List[float],
        val_acc: List[float],
        train_f1: List[float],
        val_f1: List[float],
        lrs: List[float],
        filename: str = "training_history.png",
    ):
        """Plots Loss, Accuracy, F1-Score, and LR over epochs in a 2x2 grid."""
        epochs = range(1, len(train_loss) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        axs[0, 0].plot(epochs, train_loss, "b-", label="Training", linewidth=2)
        axs[0, 0].plot(epochs, val_loss, "r--", label="Validation", linewidth=2)
        axs[0, 0].set_title("Cross-Entropy Loss", fontweight="bold")
        axs[0, 0].grid(True, linestyle=":", alpha=0.6)
        axs[0, 0].legend()

        # Accuracy
        axs[0, 1].plot(epochs, train_acc, "b-", label="Training", linewidth=2)
        axs[0, 1].plot(epochs, val_acc, "r--", label="Validation", linewidth=2)
        axs[0, 1].set_title("Classification Accuracy", fontweight="bold")
        axs[0, 1].grid(True, linestyle=":", alpha=0.6)
        axs[0, 1].legend()

        # F1-Score
        axs[1, 0].plot(epochs, train_f1, "b-", label="Training", linewidth=2)
        axs[1, 0].plot(epochs, val_f1, "r--", label="Validation", linewidth=2)
        axs[1, 0].set_title("Macro F1-Score", fontweight="bold")
        axs[1, 0].grid(True, linestyle=":", alpha=0.6)
        axs[1, 0].legend()

        # Learning Rate
        axs[1, 1].plot(epochs, lrs, "g-", label="Learning Rate", linewidth=2)
        axs[1, 1].set_title("LR Schedule (Log Scale)", fontweight="bold")
        axs[1, 1].set_yscale("log")
        axs[1, 1].grid(True, linestyle=":", alpha=0.6)
        axs[1, 1].legend()

        for ax in axs.flat:
            ax.set_xlabel("Epochs")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Advanced training curves saved to {filepath}")

    def plot_precision_recall_curves(
        self,
        y_true_bin: np.ndarray,
        y_scores: np.ndarray,
        classes: List[str],
        filename: str = "pr_curves.png",
    ):
        plt.figure(figsize=(10, 8))

        for i, cls_name in enumerate(classes):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], y_scores[:, i]
            )
            ap = average_precision_score(y_true_bin[:, i], y_scores[:, i])
            plt.plot(
                recall, precision, lw=2, label=f"{cls_name.title()} (AP = {ap:.2f})"
            )

        plt.xlabel("Recall", fontweight="bold")
        plt.ylabel("Precision", fontweight="bold")
        plt.title("Precision-Recall Curve (Per-Class Validation)", fontsize=14)
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.6)

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        classes: List[str],
        filename: str = "confusion_matrix.png",
    ):
        """Generates a styled confusion matrix using only matplotlib."""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(cm, cmap=plt.get_cmap("Blues"), alpha=0.8)
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text_color = "white" if cm[i, j] > (cm.max() / 2) else "black"
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    va="center",
                    ha="center",
                    color=text_color,
                    fontweight="bold",
                )

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="left")
        ax.set_yticklabels(classes)

        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.xaxis.set_label_position("top")
        ax.set_title("Confusion Matrix", fontsize=14, pad=20)

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved to {filepath}")

    def generate_classification_report(
        self,
        y_true: List[int],
        y_pred: List[int],
        classes: List[str],
        filename: str = "class_report.txt",
    ):
        """Generates and saves the F1-Score/Precision/Recall report."""
        report = classification_report(
            y_true,
            y_pred,
            labels=range(len(classes)),
            target_names=classes,
            zero_division=0,
        )
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            f.write(str(report))
        logger.info(f"Classification report saved to {filepath}")

    def plot_temporal_stability(
        self,
        probs_history: List[np.ndarray],
        lock_history: List[int],
        classes: List[str],
        rep_frames: Optional[List[int]] = None,
        filename: str = "temporal_stability.png",
    ):
        """Plots Softmax probabilities over time with Mutex Lock activation overlay."""
        frames = range(len(probs_history))
        probs_arr = np.array(probs_history)

        fig, ax = plt.subplots(figsize=(15, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))

        for i, cls_name in enumerate(classes):
            ax.plot(
                frames,
                probs_arr[:, i],
                label=cls_name,
                color=colors[i],
                linewidth=1.5,
                alpha=0.8,
            )

            class_locked = np.array(lock_history) == i
            if np.any(class_locked):
                ax.fill_between(
                    frames, 0, 1, where=class_locked, color=colors[i], alpha=0.15
                )

        if rep_frames:
            for idx, r_frame in enumerate(rep_frames):
                label = "Rep Triggered" if idx == 0 else ""
                ax.axvline(
                    x=r_frame,
                    color="black",
                    linestyle="-.",
                    linewidth=2,
                    alpha=0.9,
                    label=label,
                )

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Frame Timeline", fontweight="bold")
        ax.set_ylabel("Softmax Probability", fontweight="bold")
        ax.set_title(
            "FSM Mutex Lock Override vs Raw Softmax", fontsize=12, fontweight="bold"
        )
        ax.grid(True, linestyle="--", alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Temporal diagnostic plot saved to {filepath}")


# CELL 7. TEST METRICS BACKEND
class TestSetEvaluator:
    """
    Evaluates the model on the Test Set using Video-Level Hard Voting.
    """

    def __init__(self, model, config, visualizer):
        self.model = model
        self.config = config
        self.visualizer = visualizer

    def extract_video_tensors(self, test_dir: str) -> Dict[str, Dict]:
        """
        Scans the test directory and groups .pt sequence files by their source video ID.
        Assumes files are named like: classIdx_videoId_seqIdx.pt
        Example: 0_squatVideo1_seq0.pt
        """
        pt_files = glob.glob(os.path.join(test_dir, "*.pt"))
        video_groups = {}

        for filepath in pt_files:
            filename = os.path.basename(filepath)
            # Exctracts class and id from video name
            video_id = filename.split("_seq")[0]

            if video_id not in video_groups:
                tensor_data, class_idx = torch.load(filepath, weights_only=True)

                video_groups[video_id] = {
                    "true_class": class_idx,
                    "sequences": [tensor_data],
                }
            else:
                # Load tensor from sequence
                tensor_data, _ = torch.load(filepath, weights_only=True)
                video_groups[video_id]["sequences"].append(tensor_data)

        return video_groups

    def run_hard_voting_evaluation(self, test_features_dir: str):
        """
        Executes inference on all windows of each video and applies Hard Voting.
        """
        logger.info(
            f"Starting Test Set Evaluation using Hard Voting from {test_features_dir}"
        )
        video_groups = self.extract_video_tensors(test_features_dir)

        if not video_groups:
            logger.error("No test features found. Cannot evaluate.")
            return

        y_true_video = []
        y_pred_video = []

        for video_id, data in video_groups.items():
            true_class = data["true_class"]
            sequences = data["sequences"]

            # 1. Concats all video sequences in a single batch
            # Shape: (Num_Sequences, SEQ_LENGTH, FEATURES)
            video_batch = torch.stack(sequences).to(self.model.device)

            # 2. Inference on full video
            predicted_indices, _ = self.model.predict_batch(video_batch)
            predictions_list = predicted_indices.cpu().numpy().tolist()

            # 3. HARD VOTING
            # Selects majority class
            final_vote = Counter(predictions_list).most_common(1)[0][0]

            y_true_video.append(true_class)
            y_pred_video.append(final_vote)

            logger.debug(
                f"Video {video_id}: True [{true_class}] | Voted [{final_vote}] | Raw Votes: {predictions_list}"
            )

        # 4. Generates metrics and plots (Passando i risultati video-level al Visualizer)
        classes_names = list(self.config.TARGET_CLASSES.keys())

        self.visualizer.plot_confusion_matrix(
            y_true=y_true_video,
            y_pred=y_pred_video,
            classes=classes_names,
            filename="test_confusion_matrix.png",
        )

        self.visualizer.generate_classification_report(
            y_true=y_true_video,
            y_pred=y_pred_video,
            classes=classes_names,
            filename="test_classification_report.txt",
        )

        # Calculates total video accuracy
        correct = sum(1 for t, p in zip(y_true_video, y_pred_video) if t == p)
        total = len(y_true_video)
        video_accuracy = correct / total if total > 0 else 0

        logger.info(
            f"Test Set Evaluation Complete. Video-Level Accuracy: {video_accuracy:.2%} ({correct}/{total} videos)"
        )
        return video_accuracy


# CELL 8. UNIFIED TESTING
class DiagnosticsRunner:
    """Unified test suite for backend validation with clean Colab output."""

    @classmethod
    def print_header(cls, text):
        logger.info(f"=== {text.upper()} ===")

    @classmethod
    def print_pass(cls, text):
        logger.info(f"PASS: {text}")

    @classmethod
    def print_fail(cls, text):
        logger.error(f"FAIL: {text}")

    @classmethod
    def test_geometry(cls):
        cls.print_header("TEST 1: Geometry (PoseUtils)")
        angle = PoseUtils.calculate_angle(
            np.array([0, 1]), np.array([0, 0]), np.array([1, 0])
        )
        assert 89.9 < angle < 90.1, f"Geometry failure: calculated {angle}"
        cls.print_pass("90.0 degrees calculated correctly.")

    @classmethod
    def test_kalman(cls):
        cls.print_header("TEST 2: Spatial Smoothing (KalmanFilter)")
        kf = KalmanFilter()
        true_val = 100.0
        noisy_measurements = [true_val + np.random.normal(0, 15) for _ in range(50)]
        filtered = [kf.update(z) for z in noisy_measurements]

        mse_raw = np.mean((np.array(noisy_measurements) - true_val) ** 2)
        mse_filtered = np.mean((np.array(filtered) - true_val) ** 2)

        assert mse_filtered < mse_raw, "Kalman Filter increased variance."
        cls.print_pass(
            f"Noise reduced. Raw MSE: {mse_raw:.2f} -> Filtered MSE: {mse_filtered:.2f}"
        )

    @classmethod
    def test_data_coordinator(cls):
        cls.print_header("TEST 3: Anti-Leakage Split (DataCoordinator)")

        class MockDataset(FitnessSequenceDataset):
            def __init__(self):
                self.files = [
                    "/squat_org_vid1_seq0.pt",
                    "/squat_org_vid1_seq1.pt",
                    "/squat_org_vid1_seq2.pt",
                    "/push-up_syn_vid2_seq0.pt",
                    "/push-up_syn_vid2_seq1.pt",
                ]

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                return None

        train_sub, val_sub = DataCoordinator.group_split(MockDataset(), train_ratio=0.5)
        train_idx = set(train_sub.indices)
        val_idx = set(val_sub.indices)

        vid1_indices = [0, 1, 2]
        v1_in_train = all(i in train_idx for i in vid1_indices)
        v1_in_val = all(i in val_idx for i in vid1_indices)

        assert v1_in_train or v1_in_val, (
            "Data leakage: Sequences from the same video were split."
        )
        cls.print_pass("Video groups isolated successfully.")

    @classmethod
    def test_ai_topology(cls):
        cls.print_header("TEST 4: AI Inference Pipeline (FitnessClassifier)")
        model = FitnessClassifier(Config()).to("cpu")

        # Mute warnings during test by simulating loaded weights
        model.weights_loaded = True

        dummy_batch = torch.randn(
            Config.LSTM_BATCH_SIZE, Config.SEQ_LENGTH, Config.LSTM_INPUT_SIZE
        )
        batch_idx, batch_conf = model.predict_batch(dummy_batch)
        assert batch_idx.shape == (Config.LSTM_BATCH_SIZE,), (
            "Batch inference shape mismatch."
        )

        dummy_stream = np.random.rand(Config.SEQ_LENGTH, Config.LSTM_INPUT_SIZE)
        stream_idx, conf, probs = model.predict_stream(
            dummy_stream, device=torch.device("cpu")
        )
        assert stream_idx == -1 or 0 <= stream_idx < Config.LSTM_NUM_CLASSES, (
            "Invalid stream prediction."
        )
        assert probs.shape == (Config.LSTM_NUM_CLASSES,), (
            "Invalid probability array shape."
        )

        model.weights_loaded = False  # Restore state
        cls.print_pass("Batch and Stream tensor operations nominal.")

    @classmethod
    def test_fsm(cls):
        cls.print_header("TEST 5: Finite State Machine (TrackerManager)")

        class MockConfig:
            VOTING_FRAMES = 3
            ROM_THRESHOLD = 35.0
            MUTEX_TIMEOUT_FRAMES = 5

        manager = TrackerManager(MockConfig())

        predictions = [0, 0, 0, 0, 0, 0, 0, 0]
        angles = [180.0, 180.0, 180.0, 80.0, 80.0, 180.0, 180.0, 180.0]
        confidence = 0.95

        for p, a in zip(predictions, angles):
            _, counts = manager.process_frame(p, confidence, a)
        assert counts[0] == 1, "Repetition not counted on ideal trajectory."

        manager = TrackerManager(MockConfig())
        predictions = [1, 1, 1, -1, 1, 1]
        for p in predictions:
            manager.process_frame(p, confidence, 180.0)
        assert manager.active_lock == 1, "Lock lost due to transient noise."

        manager = TrackerManager(MockConfig())
        predictions = [2, 2, 2] + [-1] * 6
        for p in predictions:
            manager.process_frame(p, confidence, 180.0)
        assert manager.active_lock == -1, "Watchdog failed to release lock."

        cls.print_pass("Kinematics, Soft-Voting Smoothing, and Watchdog nominal.")

    @classmethod
    def test_vision(cls):
        cls.print_header("TEST 6: Vision Pipeline (PoseEstimator)")
        estimator = PoseEstimator(Config())
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        keypoints, scores, boxes, confidence = estimator.predict(dummy_frame)

        assert confidence < Config.MIN_DETECTION_CONFIDENCE, (
            "Ghost detection on black frame."
        )
        assert keypoints is None, "Keypoints generated for empty frame."
        cls.print_pass("Blank frame rejected correctly.")

    @classmethod
    def test_ui_engine(cls):
        cls.print_header("TEST 7: Visual Engine (DynamicUIRenderer)")
        renderer = DynamicUIRenderer(Config())
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        renderer.draw_hud(dummy_frame, "Test-Exercise", 12, 0.98, "Lock: 1")

        assert np.any(dummy_frame > 0), "UI Renderer failed to inject pixels."
        cls.print_pass("Resolution-agnostic HUD rendered without exceptions.")

    @classmethod
    def test_metrics_visualizer(cls):
        cls.print_header("TEST 8: Analytical Plotting (MetricsVisualizer)")

        with tempfile.TemporaryDirectory() as tmpdir:
            vis = MetricsVisualizer(tmpdir)
            classes = ["A", "B", "C", "D"]

            # Confusion Matrix
            y_true = [0, 0, 1, 1, 2, 2, 3, 3]
            y_pred = [0, 0, 1, 0, 2, 2, 3, 3]
            vis.plot_confusion_matrix(y_true, y_pred, classes, filename="test_cm.png")
            assert os.path.exists(os.path.join(tmpdir, "test_cm.png")), (
                "CM plot failed."
            )

            # Phase 1: Kalman
            vis.plot_kalman_diagnostics(
                [90] * 10, [90] * 10, filename="test_kalman.png"
            )
            assert os.path.exists(os.path.join(tmpdir, "test_kalman.png")), (
                "Kalman plot failed."
            )

            # Phase 2: Advanced Training Curves & PR Curves
            vis.plot_training_curves(
                [0.5],
                [0.6],
                [0.8],
                [0.7],
                [0.75],
                [0.65],
                [1e-3],
                filename="test_tc.png",
            )
            assert os.path.exists(os.path.join(tmpdir, "test_tc.png")), (
                "Training curves failed."
            )

            y_true_bin = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            y_scores = np.array([[0.9, 0.1, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]])
            vis.plot_precision_recall_curves(
                y_true_bin, y_scores, classes, filename="test_pr.png"
            )
            assert os.path.exists(os.path.join(tmpdir, "test_pr.png")), (
                "PR curves failed."
            )

            # Phase 3: Temporal Stability
            probs_hist = [np.array([0.1, 0.8, 0.05, 0.05])] * 10
            locks_hist = [1] * 10
            vis.plot_temporal_stability(
                probs_hist, locks_hist, classes, filename="test_temp.png"
            )
            assert os.path.exists(os.path.join(tmpdir, "test_temp.png")), (
                "Temporal plot failed."
            )

            cls.print_pass(
                "All analytical visualizations rendered and saved successfully."
            )

    @classmethod
    def test_hard_voting(cls):
        cls.print_header("TEST 9: Aggregation Logic (TestSetEvaluator)")

        class MockModel:
            device = "cpu"

            def predict_batch(self, batch):
                return torch.tensor([2, 2, 1]), None

        class MockVisualizer:
            def plot_confusion_matrix(self, *args, **kwargs):
                pass

            def generate_classification_report(self, *args, **kwargs):
                pass

        class MockConfig:
            TARGET_CLASSES = {"squat": 0, "pushup": 1, "curl": 2, "press": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                tensor_dummy = torch.zeros(45, 6)
                torch.save(
                    (tensor_dummy, 2), os.path.join(tmpdir, f"curl_org_vid1_seq{i}.pt")
                )

            evaluator = TestSetEvaluator(MockModel(), MockConfig(), MockVisualizer())
            accuracy = evaluator.run_hard_voting_evaluation(tmpdir)

            assert accuracy == 1.0, f"Aggregation failed. Expected 1.0, got {accuracy}"
            cls.print_pass("Video-level sequential aggregation optimal.")

    @classmethod
    def execute(cls):
        logger.info(">>> INITIATING UNIFIED TESTING <<<")
        try:
            cls.test_geometry()
            cls.test_kalman()
            cls.test_data_coordinator()
            cls.test_ai_topology()
            cls.test_fsm()
            cls.test_vision()
            cls.test_ui_engine()
            cls.test_metrics_visualizer()
            cls.test_hard_voting()
            logger.info(">>> ALL MODULES VERIFIED. BACKEND IS READY. <<<")
        except AssertionError as e:
            logger.critical(f"CRITICAL FAILURE: {e}")
        except Exception as e:
            logger.critical(f"SYSTEM EXCEPTION: {e}")


# CELL 10. WORKFLOW 2: DATA INGESTION & FEATURE EXTRACTION
class FeatureExtractor:
    """
    Extracts skeletal features from raw videos and saves them as PyTorch tensors.
    """

    def __init__(self, config, pose_estimator):
        self.config = config
        self.pose_estimator = pose_estimator
        self.seq_length = config.SEQ_LENGTH

    def extract_from_directory(
        self,
        input_dir: str,
        output_dir: str,
        class_mapping: Dict[str, int],
    ):
        """
        Processes all videos in input_dir and saves tensors locally.
        Expects filenames containing the class name (e.g., 'squat_001.mp4').
        """
        os.makedirs(output_dir, exist_ok=True)
        video_paths = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith((".mp4", ".mov", ".avi", ".m4v")):
                    video_paths.append(os.path.join(root, file))

        if not video_paths:
            logger.warning(f"No videos found in {input_dir} hierarchy.")
            return

        logger.info(
            f"Starting extraction for {len(video_paths)} videos in {input_dir}."
        )

        processed_count = 0
        for vid_path in video_paths:
            parent_dir = os.path.basename(os.path.dirname(vid_path)).lower()
            mapped_class = next(
                (k for k in class_mapping.keys() if k in parent_dir), None
            )

            if mapped_class is None:
                logger.warning(f"Skipping {vid_path}: Class not found from path.")
                continue

            class_idx = class_mapping[mapped_class]
            dataset_type = "syn" if "synthetic" in input_dir.lower() else "org"

            raw_id = os.path.splitext(os.path.basename(vid_path))[0]
            clean_id = raw_id.replace(mapped_class, "").strip("_")
            file_prefix = f"{mapped_class.replace(' ', '_')}_{dataset_type}_{clean_id}"
            # Checkpointing
            if os.path.exists(os.path.join(output_dir, f"{file_prefix}.done")):
                continue

            self._process_single_video(vid_path, file_prefix, class_idx, output_dir)
            processed_count += 1

        logger.info(f"Extraction loop finished. Processed {processed_count} files.")

    def _process_single_video(
        self, video_path: str, file_prefix: str, class_idx: int, output_dir: str
    ):
        cap = cv2.VideoCapture(video_path)
        sequence_buffer = []
        frames_batch = []
        seq_count = 0
        frame_idx = 0
        k_filters = [KalmanFilter() for _ in range(self.config.NUM_KALMAN_FILTERS)]

        is_synthetic = self.config.SYNTHETIC_IDENTIFIER in file_prefix
        if is_synthetic:
            current_stride = self.config.SYNTHETIC_STRIDE
            max_seq_allowed = self.config.SYNTHETIC_MAX_SEQ
        else:
            stride_divisor = self.config.ORGANIC_STRIDE_DIVISORS.get(class_idx, 1)
            current_stride = max(1, self.config.SEQ_STRIDE // stride_divisor)
            max_seq_allowed = float("inf")

        def _flush_batch():
            nonlocal seq_count
            if not frames_batch or seq_count >= max_seq_allowed:
                return

            results = self.pose_estimator.predict_batch(frames_batch)

            for res in results:
                keypoints, scores, _, _ = res

                if keypoints is None:
                    # Temporal integrity protection for R-CNN
                    features = [0.0] * self.config.LSTM_INPUT_SIZE
                else:
                    features = PoseUtils.extract_key_angles(keypoints, k_filters)

                if (
                    features is not None
                    and len(features) == self.config.LSTM_INPUT_SIZE
                ):
                    sequence_buffer.append(features)

                    if len(sequence_buffer) == self.seq_length:
                        # --- DROP-OUT REJECTION (Anti-Ghosting) ---
                        empty_frames = sum(1 for f in sequence_buffer if sum(f) == 0.0)
                        max_allowed_empty = int(
                            self.seq_length * self.config.MAX_DROPOUT_RATIO
                        )

                        if empty_frames <= max_allowed_empty:
                            if seq_count >= max_seq_allowed:
                                break

                            tensor_data = torch.tensor(
                                sequence_buffer, dtype=torch.float32
                            )
                            save_name = f"{file_prefix}_seq{seq_count}.pt"
                            save_path = os.path.join(output_dir, save_name)
                            torch.save((tensor_data, class_idx), save_path)
                            seq_count += 1
                        del sequence_buffer[:current_stride]

            frames_batch.clear()

        while cap.isOpened() and seq_count < max_seq_allowed:
            ret, frame = cap.read()
            if not ret:
                break
            # --- TEMPORAL DOWNSAMPLING ---
            frame_idx += 1
            if frame_idx % self.config.FRAME_SKIP != 0:
                continue
            # --- SPATIAL DOWNSAMPLING ---
            h, w = frame.shape[:2]
            scale = self.config.MAX_DIM / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            frames_batch.append(frame)

            if len(frames_batch) == self.config.BATCH_SIZE:
                _flush_batch()

        # Process residual frames
        _flush_batch()
        cap.release()
        # creating lock file
        open(os.path.join(output_dir, f"{file_prefix}.done"), "w").close()
        logger.info(
            f"Extracted {seq_count} sequences from {file_prefix} (Stride: {current_stride})"
        )


# CELL 11. WORKFLOW 3: MODEL TRAINING
class ModelTrainer:
    """
    Orchestrates the training lifecycle for the fitness classification model.
    Handles device allocation, metric tracking, visualization dispatch, and checkpointing.
    """

    def __init__(self, config, model: nn.Module, train_loader, val_loader, visualizer):
        self.config = config
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.visualizer = visualizer
        self.device = config.DEVICE
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> nn.Module:
        """
        Executes the training loop with early stopping/LR reduction mechanisms.
        """
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.config.SCHEDULER_PATIENCE,
            factor=self.config.SCHEDULER_FACTOR,
        )

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        lrs = []

        best_val_loss = float("inf")
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_val_probs = np.array([])
        best_val_labels = np.array([])

        logger.info(
            f"Initiating training phase on {self.device} for {self.config.NUM_EPOCHS} epochs."
        )

        for epoch in range(self.config.NUM_EPOCHS):
            current_lr = optimizer.param_groups[0]["lr"]
            lrs.append(current_lr)
            # --- TRAINING PHASE ---
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            all_t_labels, all_t_preds = [], []

            for features, labels in self.train_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * features.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_t_labels.extend(labels.cpu().numpy())
                all_t_preds.extend(preds.cpu().numpy())

            train_losses.append(running_loss / total)
            train_accs.append(correct / total)
            train_f1s.append(f1_score(all_t_labels, all_t_preds, average="macro"))

            # --- VALIDATION PHASE ---
            self.model.eval()
            val_loss, correct, total = 0.0, 0, 0
            all_v_labels, all_v_preds = [], []
            all_v_probs_batch = []

            with torch.no_grad():
                for features, labels in self.val_loader:
                    features = features.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    outputs = self.model(features)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * features.size(0)
                    probs = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    all_v_labels.extend(labels.cpu().numpy())
                    all_v_preds.extend(preds.cpu().numpy())
                    all_v_probs_batch.append(probs.cpu())

            val_losses.append(val_loss / total)
            val_accs.append(correct / total)
            val_f1s.append(f1_score(all_v_labels, all_v_preds, average="macro"))

            scheduler.step(val_losses[-1])

            # --- CHECKPOINTING ---
            status_msg = f"Epoch {epoch + 1:02d}/{self.config.NUM_EPOCHS} | LR: {current_lr:.1e} | T-Loss: {train_losses[-1]:.4f} T-Acc: {train_accs[-1]:.4f} T-F1: {train_f1s[-1]:.4f} | V-Loss: {val_losses[-1]:.4f} V-Acc: {val_accs[-1]:.4f} V-F1: {val_f1s[-1]:.4f}"

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_val_probs = torch.cat(all_v_probs_batch).numpy()
                best_val_labels = np.array(all_v_labels)
                status_msg += " -> [WEIGHTS SAVED]"

            logger.info(status_msg)

        # Restore best weights
        self.model.load_state_dict(best_model_wts)
        self._export_artifacts(
            train_losses,
            val_losses,
            train_accs,
            val_accs,
            train_f1s,
            val_f1s,
            lrs,
            best_val_labels,
            best_val_probs,
        )

        return self.model

    def _export_artifacts(
        self,
        t_loss: List[float],
        v_loss: List[float],
        t_acc: List[float],
        v_acc: List[float],
        t_f1: List[float],
        v_f1: List[float],
        lrs: List[float],
        best_val_labels: np.ndarray,
        best_val_probs: np.ndarray,
    ):
        """Delegates plotting and handles local serialization."""
        logger.info("Generating training metric visualizations...")
        self.visualizer.plot_training_curves(
            t_loss, v_loss, t_acc, v_acc, t_f1, v_f1, lrs
        )

        y_true_bin = label_binarize(
            best_val_labels, classes=range(self.config.LSTM_NUM_CLASSES)
        )
        self.visualizer.plot_precision_recall_curves(
            y_true_bin, best_val_probs, self.config.CLASSES_UI
        )

        logger.info(
            f"Serializing best model weights to {self.config.get_weights_path()}..."
        )
        torch.save(self.model.state_dict(), self.config.get_weights_path())
        if hasattr(self.model, "weights_loaded"):
            setattr(self.model, "weights_loaded", True)

        logger.info(
            "Artifact export complete. Slurm post-processing will handle persistence."
        )


# CELL 12. WORKFLOW 4: INFERENCE & VIDEO PIPELINE
class VideoPipeline:
    """
    Production inference pipeline. Handles I/O, FSM orchestration, UI rendering,
    and automated syncing with Google Drive.
    """

    def __init__(
        self,
        config,
        pose_estimator,
        classifier,
        tracker_manager,
        ui_renderer,
        visualizer,
    ):
        self.config = config
        self.pose_estimator = pose_estimator
        self.classifier = classifier
        self.tracker_manager = tracker_manager
        self.ui_renderer = ui_renderer
        self.visualizer = visualizer
        self.k_filters = [KalmanFilter() for _ in range(self.config.NUM_KALMAN_FILTERS)]

    def process_all(self):
        """Main execution loop for directory-based batch inference."""
        video_files = [
            f
            for f in os.listdir(self.config.INPUT_DIR)
            if f.lower().endswith((".mp4", ".mov", ".avi"))
        ]

        if not video_files:
            logger.warning(f"No videos found in {self.config.INPUT_DIR}.")
            return

        logger.info(f"Starting production pipeline on {len(video_files)} videos.")
        log_data = []
        y_true: List[int] = []
        y_pred: List[int] = []

        for video_file in video_files:
            # --- GROUND TRUTH EXTRACTION ---
            filename_lower = video_file.lower()
            mapped_class = next(
                (k for k in self.config.TARGET_CLASSES.keys() if k in filename_lower),
                None,
            )

            if mapped_class is None:
                logger.warning(
                    f"Skipping metrics for {video_file}: No ground truth class detected in filename."
                )
                continue

            true_label_idx = self.config.TARGET_CLASSES[mapped_class]
            input_path = os.path.join(self.config.INPUT_DIR, video_file)
            output_path = os.path.join(self.config.OUTPUT_DIR, f"out_{video_file}")

            # process_video now returns both final counts and the history of predicted frames
            final_counts, video_predictions_history = self.process_video(
                input_path, output_path
            )

            # --- HARD VOTING FOR VIDEO-LEVEL PREDICTION ---
            # Filter out background/noise (-1) if present to compute metrics on actual classes
            valid_preds = [p for p in video_predictions_history if p != -1]
            if valid_preds:
                predicted_label_idx = max(set(valid_preds), key=valid_preds.count)
            else:
                predicted_label_idx = -1  # Fallback to noise if nothing was classified

            y_true.append(true_label_idx)
            y_pred.append(predicted_label_idx)

            # Formatting CSV log entry
            entry = {"Video": video_file}
            for i, cls_name in enumerate(self.config.CLASSES_UI):
                entry[cls_name.title()] = final_counts.get(i, 0)
            log_data.append(entry)

        # --- GENERATE EVALUATION ARTIFACTS ---
        if y_true and y_pred:
            logger.info("Computing test set classification metrics...")
            self.visualizer.plot_confusion_matrix(
                y_true,
                y_pred,
                self.config.CLASSES_UI,
                filename="test_confusion_matrix.png",
            )
            self.visualizer.generate_classification_report(
                y_true,
                y_pred,
                self.config.CLASSES_UI,
                filename="test_classification_report.txt",
            )

        self._write_report(log_data)
        logger.info("Batch inference completed. Outputs staged in scratch workspace.")

    def process_video(
        self, input_path: str, output_path: str
    ) -> Tuple[Dict[int, int], List[int]]:
        """Processes a single video stream with Multi-threaded I/O."""
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.config.FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        sequence_buffer = []
        video_predictions_history: List[int] = []

        # Metrics buffers
        raw_history = []
        filtered_history = []
        probs_history: List[np.ndarray] = []
        lock_history: List[int] = []
        rep_triggers = []
        previous_counts = {i: 0 for i in range(self.config.LSTM_NUM_CLASSES)}
        frame_count = 0

        logger.info(f"Processing: {os.path.basename(input_path)} ({w}x{h} @ {fps}fps)")

        # Reset Kalman filters for new video
        self.k_filters = [KalmanFilter() for _ in range(self.config.NUM_KALMAN_FILTERS)]

        self.tracker_manager.active_lock = -1
        self.tracker_manager.lock_timeout_counter = 0
        for trk in self.tracker_manager.trackers.values():
            trk.count = 0
        self.tracker_manager.smoother.window.clear()

        # Producer-Consumer
        frame_queue = queue.Queue(maxsize=128)
        write_queue = queue.Queue(maxsize=128)

        def frame_reader():
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                frame_queue.put(f)
            frame_queue.put(None)

        def frame_writer():
            while True:
                f = write_queue.get()
                if f is None:
                    break
                out.write(f)

        reader_thread = threading.Thread(target=frame_reader, daemon=True)
        writer_thread = threading.Thread(target=frame_writer, daemon=True)
        reader_thread.start()
        writer_thread.start()

        default_probs = np.zeros(self.config.LSTM_NUM_CLASSES)

        # Main compute thread
        while True:
            frame = frame_queue.get()
            if frame is None:
                write_queue.put(None)
                break

            frame_count += 1
            # 1. Pose Estimation
            prediction_result = self.pose_estimator.predict(frame)
            kpts = prediction_result[0]
            scores = prediction_result[1]
            bbox = (
                prediction_result[2]
                if prediction_result[2] is not None and len(prediction_result[2]) > 0
                else None
            )

            pred_idx = -1
            conf = 0.0
            probs = default_probs

            if kpts is not None and len(kpts) > 0:
                # 2. Geometry Extraction
                angles, raw_angles = PoseUtils.extract_key_angles(
                    kpts, self.k_filters, return_raw=True
                )
                sequence_buffer.append(angles)

                if len(sequence_buffer) == self.config.SEQ_LENGTH:
                    # 3. Stream Inference
                    pred_idx, conf, probs = self.classifier.predict_stream(
                        sequence_array=sequence_buffer,
                        device=self.config.DEVICE,
                        threshold=self.config.STREAM_CONFIDENCE_THRESHOLD,
                    )
                    sequence_buffer.pop(0)

                # 4. Kinematic Routing & FSM Update
                # Use lock target if active, otherwise use raw network output to determine which angle to track
                target_idx = (
                    self.tracker_manager.active_lock
                    if self.tracker_manager.active_lock != -1
                    else pred_idx
                )

                def _get_tracking_angle(ang_list, target):
                    if target == 0:
                        return (ang_list[6] + ang_list[7]) / 2.0  # Squat -> Knee
                    if target in [1, 2]:
                        return (
                            ang_list[0] + ang_list[1]
                        ) / 2.0  # Push-up, Bicep Curl -> Elbow
                    if target == 3:
                        return (
                            ang_list[2] + ang_list[3]
                        ) / 2.0  # Shoulder Press -> Shoulder
                    return 0.0

                tracking_angle = _get_tracking_angle(angles, target_idx)
                raw_tracking_angle = _get_tracking_angle(raw_angles, target_idx)

                filtered_history.append(tracking_angle)
                raw_history.append(raw_tracking_angle)

                smoothed_idx, counts = self.tracker_manager.process_frame(
                    pred_idx, conf, tracking_angle
                )

                # Counting rep triggers
                active_idx = self.tracker_manager.active_lock
                if active_idx != -1:
                    current_count = counts.get(active_idx, 0)
                    if current_count > previous_counts[active_idx]:
                        rep_triggers.append(frame_count)
                        previous_counts[active_idx] = current_count

                # Temporal logging
                video_predictions_history.append(smoothed_idx)
                probs_history.append(probs)
                lock_history.append(self.tracker_manager.active_lock)

                # 5. UI Rendering
                active_class_name = (
                    self.config.CLASSES_UI[self.tracker_manager.active_lock]
                    if self.tracker_manager.active_lock != -1
                    else "IDLE"
                )
                current_reps = counts.get(self.tracker_manager.active_lock, 0)

                state_str = "N/A"
                if self.tracker_manager.active_lock != -1:
                    state_str = (
                        "CONCENTRIC"
                        if self.tracker_manager.trackers[
                            self.tracker_manager.active_lock
                        ].state
                        == 1
                        else "ECCENTRIC"
                    )

                # Standard edges for COCO (Simplified)
                edges = [
                    (5, 7),
                    (7, 9),
                    (6, 8),
                    (8, 10),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16),
                ]

                self.ui_renderer.draw_bbox(frame, bbox)
                self.ui_renderer.draw_skeleton(
                    frame, kpts, scores, edges, threshold=0.5
                )
                self.ui_renderer.draw_hud(
                    frame, active_class_name, current_reps, conf, state_str
                )

            write_queue.put(frame)

        cap.release()
        writer_thread.join()
        out.release()

        # Plots export
        diag_kalman_name = (
            f"kalman_diag_{os.path.basename(input_path).split('.')[0]}.png"
        )
        self.visualizer.plot_kalman_diagnostics(
            raw_history, filtered_history, diag_kalman_name
        )

        if probs_history:
            self.visualizer.plot_temporal_stability(
                probs_history=probs_history,
                lock_history=lock_history,
                classes=self.config.CLASSES_UI,
                rep_frames=rep_triggers,
                filename=f"temporal_{os.path.basename(input_path).split('.')[0]}.png",
            )

        final_counts = {k: v.count for k, v in self.tracker_manager.trackers.items()}
        return final_counts, video_predictions_history

    def _write_report(self, log_data: List[Dict]):
        if not log_data:
            return

        csv_path = os.path.join(self.config.OUTPUT_DIR, "pipeline_report.csv")
        fieldnames = ["Video"] + [cls.title() for cls in self.config.CLASSES_UI]

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_data)
        logger.info(f"Report generated at {csv_path}")


# ENTRY POINT
def main():
    parser = argparse.ArgumentParser(
        description="Fitness Action Recognition HPC Pipeline"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["diagnostics", "extract", "train", "inference", "all"],
        help="Target workflow to execute on the compute node.",
    )
    parser.add_argument(
        "--feature-tag",
        type=str,
        default=Config.FEATURE_TAG,
        help="Identifier for the feature space mapping.",
    )
    parser.add_argument(
        "--model-tag",
        type=str,
        default=Config.MODEL_TAG,
        help="Identifier for the model weights checkpointing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for spatial (R-CNN).",
    )
    parser.add_argument(
        "--lstm-batch-size",
        type=int,
        default=16,
        help="Batch size for temporal (BiLSTM) inference/training.",
    )
    args = parser.parse_args()

    # 1. Runtime Config Override (SSOT Maintained)
    Config.FEATURE_TAG = args.feature_tag
    Config.MODEL_TAG = args.model_tag
    Config.BATCH_SIZE = args.batch_size
    Config.LSTM_BATCH_SIZE = args.lstm_batch_size

    # 2. Bootstrapping HPC Environment
    global logger
    logger = Config.setup_environment()
    set_deterministic_environment(42)
    logger.info(f"HPC Node initialized. Master Task: {args.task.upper()}")

    # 3. Workflow Routing
    if args.task in ["diagnostics", "all"]:
        DiagnosticsRunner.execute()

    if args.task in ["extract", "all"]:
        logger.info("=== STARTING WORKFLOW: FEATURE EXTRACTION ===")
        estimator = PoseEstimator(Config())
        extractor = FeatureExtractor(Config(), estimator)

        # Ensure directories are populated by Slurm before this step
        extractor.extract_from_directory(
            input_dir=Config.REAL_DIR,
            output_dir=Config.get_features_dir(),
            class_mapping=Config.TARGET_CLASSES,
        )
        extractor.extract_from_directory(
            input_dir=Config.SYNTH_DIR,
            output_dir=Config.get_features_dir(),
            class_mapping=Config.TARGET_CLASSES,
        )

    if args.task in ["train", "all"]:
        logger.info("=== STARTING WORKFLOW: MODEL TRAINING ===")
        train_loader, val_loader = DataCoordinator.get_loaders(
            features_dir=Config.get_features_dir(), batch_size=Config.LSTM_BATCH_SIZE
        )
        if train_loader is None or val_loader is None:
            logger.error("Missing features. Run --task extract first.")
            sys.exit(1)

        model = FitnessClassifier(Config())
        visualizer = MetricsVisualizer(output_dir=Config.OUTPUT_DIR)
        trainer = ModelTrainer(
            config=Config(),
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            visualizer=visualizer,
        )
        trainer.train()

    if args.task in ["inference", "all"]:
        logger.info("=== STARTING WORKFLOW: PRODUCTION INFERENCE ===")
        model = FitnessClassifier(Config())
        model.load_weights(
            Config.get_weights_path(), torch.device(Config.DEVICE), strict_match=True
        )

        estimator = PoseEstimator(Config())
        tracker_mgr = TrackerManager(Config())
        ui_renderer = DynamicUIRenderer(Config())
        visualizer = MetricsVisualizer(output_dir=Config.OUTPUT_DIR)

        pipeline = VideoPipeline(
            Config(), estimator, model, tracker_mgr, ui_renderer, visualizer
        )
        pipeline.process_all()

    logger.info(
        f"Task {args.task.upper()} completed. Yielding to Slurm for post-processing."
    )


if __name__ == "__main__":
    main()

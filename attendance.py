"""Core logic for Face-based Class Attendance: find, count, recognise and group faces.

Combines four GeeksforGeeks Python projects:
  * Face detection with OpenCV          -> YuNet face detector (cv2.FaceDetectorYN)
  * Count number of faces               -> len(faces) in every class photo
  * Unsupervised face clustering        -> SFace face embeddings + DBSCAN-style grouping
  * Simple attendance tracker           -> register.py keeps the dated register and CSV

Runs in the browser through Pyodide, and also on a normal Python install.
Images are RGB NumPy arrays throughout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# SFace's published cosine-similarity threshold for "same person" is 0.363.
# We ask for a little more to avoid marking the wrong student present.
MATCH_THRESHOLD = 0.40
# Faces closer than this (cosine similarity) end up in the same group when clustering.
CLUSTER_THRESHOLD = 0.40

GREEN = (46, 160, 67)
AMBER = (230, 145, 20)


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, w, h in the original photo
    score: float  # detector confidence
    embedding: np.ndarray  # 128 numbers, unit length: the face's "fingerprint"


class FaceEngine:
    def __init__(self, min_confidence: float = 0.75):
        self.detector = cv2.FaceDetectorYN.create(
            os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"),
            "",
            (320, 320),
            score_threshold=min_confidence,
            nms_threshold=0.3,
            top_k=200,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            os.path.join(MODELS, "face_recognition_sface_2021dec_int8.onnx"), ""
        )

    # ---- face detection + counting --------------------------------------
    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Returns YuNet rows (box, 5 landmarks, score) in the photo's own coordinates."""
        h, w = bgr.shape[:2]
        # Class photos have many small faces, so keep plenty of resolution.
        scale = min(1.0, 1600 / max(h, w))
        small = cv2.resize(bgr, (round(w * scale), round(h * scale))) if scale < 1 else bgr
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, dets = self.detector.detect(small)
        if dets is None:
            return np.zeros((0, 15), np.float32)
        dets = dets.copy()
        dets[:, :14] /= scale
        return dets

    # ---- face fingerprint -------------------------------------------------
    def analyse(self, rgb: np.ndarray) -> list[Face]:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = []
        for d in self.detect(bgr):
            aligned = self.recognizer.alignCrop(bgr, d)
            emb = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            x, y, w, h = (int(round(v)) for v in d[:4])
            faces.append(Face((x, y, w, h), float(d[-1]), emb))
        faces.sort(key=lambda f: (f.box[0], f.box[1]))  # left to right, like reading a row
        return faces


# ---- recognition ---------------------------------------------------------
def match(faces: list[Face], roster: dict[str, np.ndarray], threshold: float = MATCH_THRESHOLD):
    """Pairs each face with at most one student and each student with at most one face.

    roster maps student id -> array of that student's enrolled fingerprints (n x 128).
    Returns a list (one per face) of (student_id or None, similarity).
    """
    ids = list(roster)
    result: list[tuple[str | None, float]] = [(None, 0.0)] * len(faces)
    if not faces or not ids:
        return result
    E = np.stack([f.embedding for f in faces])
    # best similarity between each face and any enrolled photo of each student
    sims = np.stack([(E @ roster[i].T).max(axis=1) for i in ids], axis=1)  # faces x students
    best = [(None, float(sims[k].max())) for k in range(len(faces))]
    # greedy: most confident pairs first, so two faces can't both claim one student
    order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
    used_f, used_s = set(), set()
    for f, s in order:
        if sims[f, s] < threshold:
            break
        if f in used_f or s in used_s:
            continue
        used_f.add(f)
        used_s.add(s)
        result[f] = (ids[s], float(sims[f, s]))
    for f in range(len(faces)):
        if result[f][0] is None:
            result[f] = best[f]
    return result


# ---- unsupervised grouping -----------------------------------------------
def cluster(embeddings: np.ndarray, threshold: float = CLUSTER_THRESHOLD, min_size: int = 1) -> list[int]:
    """DBSCAN-style grouping of face fingerprints (cosine distance, eps = 1 - threshold).

    Faces linked by a chain of close matches share a group. Groups smaller than
    min_size get the label -1 ("noise" in DBSCAN). Groups are numbered biggest first.
    """
    n = len(embeddings)
    if n == 0:
        return []
    sims = embeddings @ embeddings.T
    labels = -np.ones(n, int)
    current = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        labels[start] = current
        queue = [start]
        while queue:
            i = queue.pop()
            for j in np.where((sims[i] >= threshold) & (labels == -1))[0]:
                labels[j] = current
                queue.append(j)
        current += 1
    sizes = np.bincount(labels)
    order = np.argsort(-sizes, kind="stable")
    rename = {int(old): new for new, old in enumerate(order)}
    out = [rename[int(l)] for l in labels]
    return [l if sizes[order[l]] >= min_size else -1 for l in out]


# ---- drawing ---------------------------------------------------------------
def draw(rgb: np.ndarray, faces: list[Face], labels: list[tuple[str, tuple[int, int, int]]]) -> np.ndarray:
    out = rgb.copy()
    t = max(2, round(max(out.shape[:2]) / 500))
    for f, (text, color) in zip(faces, labels):
        x, y, w, h = f.box
        cv2.rectangle(out, (x, y), (x + w, y + h), color, t)
        if not text:
            continue
        fs = max(0.45, min(1.2, w / 160))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t - 1))
        ty = y + h + th + 8 if y < th + 10 else y
        cv2.rectangle(out, (x, ty - th - 8), (x + tw + 8, ty), color, -1)
        cv2.putText(out, text, (x + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)
    return out

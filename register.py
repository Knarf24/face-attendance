"""The class register the web page calls. Runs in the browser through Pyodide.

Keeps the students (name + face fingerprints), the dated attendance register,
and the faces from the latest photo. The page saves `export_state()` in the
browser's own storage, so nothing ever leaves the device.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime

import numpy as np

from attendance import AMBER, GREEN, Face, FaceEngine, cluster, draw, match

engine = FaceEngine()

students: dict[str, dict] = {}  # id -> {"name", "thumb", "embs": [[128 floats], ...]}
register: dict[str, dict[str, str]] = {}  # "YYYY-MM-DD" -> {student id: "HH:MM"}
last_faces: list[Face] = []
pool: list[Face] = []  # faces gathered from class photos, for grouping
pool_photo: list[int] = []
pool_labels: list[int] = []
MAX_PHOTOS_PER_STUDENT = 8


# ---- saving and loading ----------------------------------------------------
def export_state() -> str:
    return json.dumps({"version": 1, "students": students, "register": register})


def import_state(text: str) -> str:
    global students, register
    data = json.loads(text) if text else {}
    students = data.get("students", {})
    register = data.get("register", {})
    return summary()


def _roster() -> dict[str, np.ndarray]:
    return {i: np.asarray(s["embs"], np.float32) for i, s in students.items() if s["embs"]}


def _pixels(rgba, width: int, height: int) -> np.ndarray:
    buf = rgba.to_py() if hasattr(rgba, "to_py") else rgba
    return np.frombuffer(buf, np.uint8).reshape(height, width, 4)[:, :, :3].copy()


def _rgba(rgb: np.ndarray) -> bytes:
    return np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)]).tobytes()


def _emb_list(f: Face) -> list[float]:
    return [round(float(v), 5) for v in f.embedding]


# ---- photo scanning --------------------------------------------------------
def scan(rgba, width: int, height: int):
    """Finds and recognises every face. Returns (annotated RGBA bytes, JSON)."""
    global last_faces
    rgb = _pixels(rgba, width, height)
    last_faces = engine.analyse(rgb)
    matches = match(last_faces, _roster())
    faces, labels = [], []
    for k, (f, (sid, sim)) in enumerate(zip(last_faces, matches)):
        name = students[sid]["name"] if sid else None
        faces.append({"index": k, "box": list(f.box), "id": sid, "name": name, "similarity": round(sim, 3)})
        labels.append((name, GREEN) if sid else (f"New #{k + 1}", AMBER))
    annotated = draw(rgb, last_faces, labels)
    return _rgba(annotated), json.dumps({"count": len(faces), "faces": faces})


def mark_present(date: str) -> str:
    """Marks every recognised face in the last photo present on `date`."""
    day = register.setdefault(date, {})
    now = datetime.now().strftime("%H:%M")
    newly = []
    for sid, _ in match(last_faces, _roster()):
        if sid and sid not in day:
            day[sid] = now
            newly.append(students[sid]["name"])
    return json.dumps({"newly": newly, "present": len(day), **json.loads(summary())})


def set_present(date: str, sid: str, present: bool) -> str:
    """Manual tick or untick in the register (teacher override)."""
    day = register.setdefault(date, {})
    if present:
        day.setdefault(sid, datetime.now().strftime("%H:%M") + " (manual)")
    else:
        day.pop(sid, None)
    return summary()


# ---- students ----------------------------------------------------------------
def enroll(face_index: int, name: str, thumb: str = "") -> str:
    """Adds the face from the last photo to a student (new, or existing with that name)."""
    name = " ".join(name.split())
    if not name:
        raise ValueError("Please type the student's name.")
    face = last_faces[int(face_index)]
    sid = next((i for i, s in students.items() if s["name"].lower() == name.lower()), None)
    if sid is None:
        sid = uuid.uuid4().hex[:8]
        students[sid] = {"name": name, "thumb": thumb, "embs": []}
    s = students[sid]
    s["embs"] = (s["embs"] + [_emb_list(face)])[-MAX_PHOTOS_PER_STUDENT:]
    if thumb and not s.get("thumb"):
        s["thumb"] = thumb
    return summary()


def rename_student(sid: str, name: str) -> str:
    name = " ".join(name.split())
    if name and sid in students:
        students[sid]["name"] = name
    return summary()


def remove_student(sid: str) -> str:
    students.pop(sid, None)
    for day in register.values():
        day.pop(sid, None)
    return summary()


def clear_all() -> str:
    global pool, pool_photo, pool_labels
    students.clear()
    register.clear()
    pool, pool_photo, pool_labels = [], [], []
    return summary()


# ---- grouping faces from several class photos -------------------------------
def pool_add(rgba, width: int, height: int, photo_no: int) -> int:
    """Finds the faces in one class photo and adds them to the grouping pool."""
    faces = engine.analyse(_pixels(rgba, width, height))
    pool.extend(faces)
    pool_photo.extend([int(photo_no)] * len(faces))
    return len(faces)


def pool_clear() -> int:
    global pool, pool_photo, pool_labels
    pool, pool_photo, pool_labels = [], [], []
    return 0


def pool_group() -> str:
    """Groups the pooled faces by person (no names needed). Returns groups biggest first."""
    global pool_labels
    if not pool:
        return json.dumps({"groups": []})
    E = np.stack([f.embedding for f in pool])
    pool_labels = cluster(E)
    roster = _roster()
    groups = {}
    for k, (f, label) in enumerate(zip(pool, pool_labels)):
        groups.setdefault(label, []).append({"face": k, "photo": pool_photo[k], "box": list(f.box)})
    out = []
    for label in sorted(groups):
        members = groups[label]
        # does this group already look like an enrolled student?
        centre = E[[m["face"] for m in members]].mean(axis=0)
        centre /= np.linalg.norm(centre) + 1e-9
        guess = None
        if roster:
            fake = Face((0, 0, 0, 0), 1.0, centre)
            sid, sim = match([fake], roster)[0]
            guess = students[sid]["name"] if sid else None
        out.append({"group": int(label), "size": len(members), "faces": members, "known": guess})
    return json.dumps({"groups": out, "faces": len(pool)})


def pool_enroll_group(group: int, name: str, thumb: str = "") -> str:
    """Gives a name to a whole group: all its faces become that student's fingerprints."""
    global last_faces
    keep = last_faces
    members = [f for f, l in zip(pool, pool_labels) if l == int(group)]
    try:
        for k in range(len(members)):
            last_faces = members
            enroll(k, name, thumb if k == 0 else "")
    finally:
        last_faces = keep
    return summary()


# ---- reports -------------------------------------------------------------------
def summary() -> str:
    dates = sorted(register)
    rows = []
    for sid, s in sorted(students.items(), key=lambda kv: kv[1]["name"].lower()):
        attended = sum(sid in register[d] for d in dates)
        rows.append(
            {
                "id": sid,
                "name": s["name"],
                "thumb": s.get("thumb", ""),
                "photos": len(s["embs"]),
                "attended": attended,
                "rate": round(100 * attended / len(dates)) if dates else None,
                "days": {d: register[d].get(sid) for d in dates},
            }
        )
    return json.dumps({"students": rows, "dates": dates})


def export_csv() -> str:
    dates = sorted(register)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Student", *dates, "Days present", "Attendance %"])
    for sid, s in sorted(students.items(), key=lambda kv: kv[1]["name"].lower()):
        marks = ["Present" if sid in register[d] else "Absent" for d in dates]
        n = marks.count("Present")
        w.writerow([s["name"], *marks, n, round(100 * n / len(dates)) if dates else ""])
    w.writerow([])
    w.writerow(["Present (total)", *[len(register[d]) for d in dates]])
    return buf.getvalue()

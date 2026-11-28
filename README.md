# 🎓 Face-based Class Attendance

**Live demo:** https://knarf24.github.io/face-attendance/

**The problem:** roll call wastes class time and is easy to fake.

**The app:** snap the classroom. It counts the faces, recognises each enrolled student and marks
the register for the day. Faces it doesn't know yet can be named on the spot, and a whole class
list can be built from a few group photos without typing a name twice.

It is written in **Python** (OpenCV + NumPy) and runs **inside your browser** with
[Pyodide](https://pyodide.org), so it is hosted free on GitHub Pages.

> 🔒 **Privacy:** photos and face data never leave the device. The class list (names, small face
> pictures and face fingerprints) and the register are kept only in the browser's local storage.
> The page has *Download backup*, *Restore backup* and *Delete everything* buttons. Ask for consent
> before enrolling anyone's face.

Built by combining four GeeksforGeeks Python projects:

| GeeksforGeeks project | What it does here |
|---|---|
| [Face Detection with Webcam](https://www.geeksforgeeks.org/python/face-detection-using-python-and-opencv-with-webcam/) | Finds faces in the webcam or uploaded class photo with OpenCV |
| [Count Number of Faces](https://www.geeksforgeeks.org/python/count-number-of-faces-using-python-opencv/) | Counts everyone in the photo (headcount vs. recognised) |
| [Unsupervised Face Clustering Pipeline](https://www.geeksforgeeks.org/machine-learning/ml-unsupervised-face-clustering-pipeline/) | Groups faces from several photos by person, so the class list can be set up quickly |
| [Simple Attendance Tracker](https://www.geeksforgeeks.org/python/simple-attendance-tracker-using-python/) | Dated register, attendance rate per student and CSV export |

## How it works

1. **Face detection and counting:** OpenCV's YuNet detector (`cv2.FaceDetectorYN`) finds every face.
2. **Face fingerprint:** each face is aligned by its landmarks and turned into 128 numbers by the SFace
   recognition network (`cv2.FaceRecognizerSF`, int8 model).
3. **Recognition:** cosine similarity against each student's enrolled fingerprints (up to 8 per
   student). Each student can only be matched once per photo; below the threshold the face is "new".
4. **Grouping:** DBSCAN-style clustering on cosine distance groups faces from several photos by person.
5. **Register:** present / absent per date, manual corrections, attendance %, CSV export.

## Files

- `attendance.py`: face detection, fingerprints, matching, clustering and drawing (plain Python, also runs on a normal install)
- `register.py`: the class list and register that the web page calls
- `web/`: the page (`index.html`, `main.js` for the camera and storage, `style.css`) and the Pyodide runtime with OpenCV
- `models/`: YuNet face detector and SFace face recognizer from the [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo)
- `build_site.sh` and `.github/workflows/pages.yml`: every push to `main` publishes to GitHub Pages

## Run it on your computer

```bash
sh build_site.sh
cd _site && python -m http.server 8000
```

Then open http://localhost:8000.

## Fast loading

The page shows straight away and Python starts in the background, so nothing waits on a
"Starting Python" screen. Anything that needs Python simply waits for it if you are quicker.

- **Saved on the device** (`web/sw.js`): after the first visit, Python (Pyodide) and its packages are kept in
  the browser, so a refresh or a later visit doesn't download them again. Every app on the same site shares
  that one saved copy. The app's own files are still fetched fresh, so updates show up right away.
- **Quick restart** (`web/pyfast.js`): the first visit also saves a snapshot of a freshly started Python.
  Later visits restore it instead of starting Python from scratch (about 1.2 s down to 0.3 s on a laptop).
  If the snapshot can't be used (for example after a Pyodide upgrade), Python starts normally and a new one
  is saved.
- `build_site.sh` writes the Pyodide version into `sw.js`, so upgrading `web/pyodide/` needs no other change.


## Credits

- Face models: YuNet and SFace from the OpenCV Model Zoo (Apache 2.0 / MIT)
- Python in the browser: Pyodide (MPL 2.0)
- Project ideas: GeeksforGeeks

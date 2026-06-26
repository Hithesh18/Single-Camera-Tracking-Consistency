"""Cross-scene single-camera consistency runner.

One class drives the whole experiment for BOTH the Colab and the local notebooks,
so the notebooks stay tiny (config + a few calls). It:
  1. downloads a scene (videos + ground truth) from HuggingFace,
  2. generates single-camera tracks (detect -> embed -> track+fix),
  3. trains the learned ReID matcher on a scene,
  4. benchmarks raw vs conservative vs learned against ground truth and writes a
     clean combined record.

Outputs are cached so finished steps are skipped on re-run. On Colab, embeddings /
matcher / results are mirrored to Drive; locally everything stays on disk.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

REID_CMD = ("cd {repo}/deep-person-reid && "
            "PYTHONPATH={repo}/deep-person-reid:{repo}/deep-person-reid/torchreid "
            "{py} torchreid/aic25_extract.py -s {scene} --dataset {dataset} ../")

_KEYS = [("idf1", "IDF1"), ("mota", "MOTA"),
         ("num_switches", "IDSW"), ("num_fragmentations", "Frag")]


class CrossSceneRunner:
    def __init__(self, repo, py, dataset="Val", cameras=None, maxf=1500,
                 track_params=None, hf_token="", results_dir=None,
                 on_colab=False, drive=None):
        self.repo = repo
        self.py = py
        self.dataset = dataset
        self.cameras = cameras
        self.maxf = maxf
        self.track_params = track_params or {}
        self.hf_token = hf_token
        self.on_colab = on_colab
        self.drive = drive
        self.results_dir = results_dir or f"{repo}/results_local"

    # ------------------------------------------------------------------ shell
    def _sh(self, cmd):
        return os.system(cmd)

    def _run(self, cmd):
        r = subprocess.run(cmd, shell=True, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print("\n".join(r.stdout.strip().splitlines()[-30:]))
        return r.returncode

    # ------------------------------------------------------------------ paths
    def _scene_store(self, scene):
        """Where videos live: Drive datasets on Colab, repo tree locally."""
        if self.on_colab:
            return f"{self.drive}/datasets/{self.dataset}/{scene}"
        return f"{self.repo}/AIC25_Track1/{self.dataset}/{scene}"

    def _repo_scene(self, scene):
        return f"{self.repo}/AIC25_Track1/{self.dataset}/{scene}"

    def _emb_cache(self, scene):
        return f"{self.drive}/cache/{self.dataset}/{scene}/EmbedFeature"

    def cams_for(self, scene):
        src = f"{self._scene_store(scene)}/videos"
        allc = (sorted(os.path.splitext(f)[0] for f in os.listdir(src) if f.endswith(".mp4"))
                if os.path.isdir(src) else [])
        return ([c for c in self.cameras if c in allc] if self.cameras else allc), src

    # ------------------------------------------------------------------ colab Drive linking
    def link_outputs(self):
        """Detection/Tracking -> Drive (persist); EmbedFeature stays local. Colab only."""
        if not self.on_colab:
            return
        for folder in ["Detection", "Tracking"]:
            df = f"{self.drive}/outputs/{folder}"
            rf = f"{self.repo}/{folder}"
            os.makedirs(df, exist_ok=True)
            if os.path.islink(rf):
                continue
            if os.path.isdir(rf):
                for it in os.listdir(rf):
                    d = f"{df}/{it}"
                    if not os.path.exists(d):
                        shutil.move(f"{rf}/{it}", d)
                shutil.rmtree(rf, ignore_errors=True)
            os.symlink(df, rf)
        ef = f"{self.repo}/EmbedFeature"
        if os.path.islink(ef):
            os.unlink(ef)
        os.makedirs(ef, exist_ok=True)

    # ------------------------------------------------------------------ download
    def download_scene(self, scene):
        from huggingface_hub import snapshot_download, login
        store = self._scene_store(scene)
        repo_scene = self._repo_scene(scene)
        dv = f"{store}/videos"
        os.makedirs(store, exist_ok=True)
        os.makedirs(repo_scene, exist_ok=True)
        if not (os.path.isdir(dv) and os.listdir(dv)):
            assert self.hf_token, "Set hf_token (HF_TOKEN) before downloading."
            login(token=self.hf_token)
            sp = self.dataset.lower()
            print(f"[{scene}] downloading videos + GT from HuggingFace...", flush=True)
            tmp = f"{self.repo}/hf_tmp"
            snapshot_download(
                "nvidia/PhysicalAI-SmartSpaces", repo_type="dataset", local_dir=tmp,
                allow_patterns=[f"MTMC_Tracking_2025/{sp}/{scene}/videos/**",
                                f"MTMC_Tracking_2025/{sp}/{scene}/calibration.json",
                                f"MTMC_Tracking_2025/{sp}/{scene}/ground_truth.json"])
            src = f"{tmp}/MTMC_Tracking_2025/{sp}/{scene}"
            if not os.path.isdir(f"{src}/videos"):
                shutil.rmtree(tmp, ignore_errors=True)
                raise RuntimeError(
                    f"No files downloaded for '{scene}' in split '{self.dataset}'. "
                    f"That scene probably isn't in this split on HuggingFace — "
                    f"list available scenes with HfApi().list_repo_files('nvidia/PhysicalAI-SmartSpaces', "
                    f"repo_type='dataset') and pick a TEST_SCENE that exists in '{self.dataset}'.")
            if not os.path.isdir(dv):
                shutil.copytree(f"{src}/videos", dv)
            for fn in ("calibration.json", "ground_truth.json"):
                if os.path.exists(f"{src}/{fn}"):
                    shutil.copy(f"{src}/{fn}", f"{store}/{fn}")
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"[{scene}] videos already present")
        # scripts read GT/calib from the repo tree — mirror them there
        for fn in ("calibration.json", "ground_truth.json"):
            s, d = f"{store}/{fn}", f"{repo_scene}/{fn}"
            if os.path.exists(s) and not os.path.exists(d):
                shutil.copy(s, d)

    # ------------------------------------------------------------------ generate single-camera
    def _scene_done(self, scene):
        cams, _ = self.cams_for(scene)
        if not cams:
            return False
        for cam in cams:
            if not os.path.exists(f"{self.repo}/Tracking/Singlecamera/{scene}/{cam}/{cam}.json"):
                return False
            ok_local = os.path.isdir(f"{self.repo}/EmbedFeature/{scene}/{cam}")
            ok_cache = self.on_colab and os.path.isdir(f"{self._emb_cache(scene)}/{scene}/{cam}")
            if not (ok_local or ok_cache):
                return False
        return True

    def generate_scene(self, scene, force=False):
        self.download_scene(scene)
        os.chdir(self.repo)
        if not force and self._scene_done(scene):
            dst = f"{self.repo}/EmbedFeature/{scene}"
            if self.on_colab and not os.path.isdir(dst):
                shutil.copytree(f"{self._emb_cache(scene)}/{scene}", dst)
            print(f"[{scene}] CACHE HIT — skipped GPU work")
            return
        cams, cam_src = self.cams_for(scene)
        print(f"[{scene}] generating ({len(cams)} cams):", cams)
        vid_dir = f"{self._repo_scene(scene)}/videos"
        os.makedirs(vid_dir, exist_ok=True)
        for cam in cams:
            if os.path.isdir(f"{vid_dir}/{cam}/Frame"):
                continue
            srcmp4 = next((p for p in [f"{cam_src}/{cam}.mp4", f"{cam_src}/{cam}/{cam}.mp4"]
                           if os.path.exists(p)), None)
            if srcmp4 and not os.path.exists(f"{vid_dir}/{cam}.mp4") and not os.path.isdir(f"{vid_dir}/{cam}"):
                os.symlink(srcmp4, f"{vid_dir}/{cam}.mp4")
        self._sh(f"{self.py} tools/extract_frames_25.py AIC25_Track1/{self.dataset} -s {scene}")

        if os.path.exists(f"{self.repo}/BoT-SORT/ai_city_ckpt.pth.tar"):
            ckpt = "BoT-SORT/ai_city_ckpt.pth.tar"
            exp = "BoT-SORT/yolox/exps/example/mot/yolox_x_AI_City_25.py"
        else:
            ckpt = "BoT-SORT/pretrained/bytetrack_x_mot17.pth.tar"
            exp = "BoT-SORT/yolox/exps/example/mot/yolox_x_mix_det.py"
        det_cap = f"--max_frames {self.maxf}" if self.maxf else ""
        trk_cap = f"--limit_frames {self.maxf}" if self.maxf else ""
        track_args = " ".join(f"--{k} {v}" for k, v in self.track_params.items())
        shutil.rmtree(f"{self.repo}/Detection/{scene}", ignore_errors=True)
        os.makedirs(f"{self.repo}/Detection/{scene}", exist_ok=True)
        shutil.rmtree(f"{self.repo}/EmbedFeature/{scene}", ignore_errors=True)

        for cam in cams:
            print(f"  detect {cam}")
            self._run(f"{self.py} BoT-SORT/tools/aic25_get_detection.py --scene {scene} "
                      f"--dataset {self.dataset} --camera {cam} -f {exp} -c {ckpt} {det_cap} ./")
        print("  embeddings")
        if self._run(REID_CMD.format(repo=self.repo, py=self.py, scene=scene, dataset=self.dataset)) != 0:
            os.chdir(self.repo)
            raise RuntimeError(f"{scene}: embeddings failed (see above)")
        os.chdir(self.repo)
        for cam in cams:
            print(f"  track+fix {cam}")
            self._run(f"{self.py} BoT-SORT/single_camera_tracking.py -s {scene} -c {cam} "
                      f"--dataset {self.dataset} {track_args} {trk_cap}")
            self._run(f"{self.py} BoT-SORT/single_camera_fix.py -s {scene} -c {cam} --dataset {self.dataset}")

        if self.on_colab:
            ec = self._emb_cache(scene)
            os.makedirs(ec, exist_ok=True)
            if os.path.isdir(f"{ec}/{scene}"):
                shutil.rmtree(f"{ec}/{scene}", ignore_errors=True)
            shutil.copytree(f"{self.repo}/EmbedFeature/{scene}", f"{ec}/{scene}")
        print(f"[{scene}] single-camera ready" + (" + cached to Drive." if self.on_colab else " (local)."))

    # ------------------------------------------------------------------ train
    def train_on(self, scene, force=False):
        os.chdir(self.repo)
        local = f"{self.repo}/tracklet_repair/models/{scene}_matcher"
        drive = f"{self.drive}/models/{scene}_matcher" if self.on_colab else None
        if not force and self.on_colab and os.path.exists(drive + ".pt"):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            for ext in (".pt", ".json"):
                shutil.copy(drive + ext, local + ext)
            print(f"[{scene}] matcher CACHE HIT — restored from Drive")
            return
        if not force and not self.on_colab and os.path.exists(local + ".pt"):
            print(f"[{scene}] matcher already trained — skipped")
            return
        gt = f"{self._repo_scene(scene)}/ground_truth.json"
        self._sh(f"{self.py} -m tracklet_repair.src.matcher.train_matcher "
                 f"--gt-json {gt} --scene {scene} --out tracklet_repair/models/{scene}_matcher")
        if self.on_colab:
            os.makedirs(f"{self.drive}/models", exist_ok=True)
            for ext in (".pt", ".json"):
                if os.path.exists(local + ext):
                    shutil.copy(local + ext, drive + ext)
            print(f"[{scene}] matcher trained + saved to Drive.")
        else:
            print(f"[{scene}] matcher trained (local).")

    # ------------------------------------------------------------------ benchmark
    def _matcher_path(self, scene):
        local = f"{self.repo}/tracklet_repair/models/{scene}_matcher"
        if self.on_colab and not os.path.exists(local + ".pt"):
            for ext in (".pt", ".json"):
                d = f"{self.drive}/models/{scene}_matcher{ext}"
                if os.path.exists(d):
                    shutil.copy(d, local + ext)
        return local

    def benchmark(self, test_scene, matcher_scene):
        os.chdir(self.repo)
        gt = f"{self._repo_scene(test_scene)}/ground_truth.json"
        matcher = self._matcher_path(matcher_scene)
        assert os.path.exists(matcher + ".pt"), f"No {matcher_scene} matcher — run train_on first."
        tag = f"{matcher_scene}_to_{test_scene}"
        base = f"tracklet_repair/results/{tag}"
        bench = "tracklet_repair.src.evaluation.benchmark_scene"
        self._sh(f"{self.py} -m {bench} --gt-json {gt} --scene {test_scene} "
                 f"--merge-mode conservative --output-dir {base}_conservative")
        self._sh(f"{self.py} -m {bench} --gt-json {gt} --scene {test_scene} --merge-mode learned "
                 f"--matcher-path {matcher} --embed-root EmbedFeature --output-dir {base}_learned")
        txt, summary = self._combined_record(matcher_scene, test_scene, base)
        rd = (f"{self.drive}/results/{tag}" if self.on_colab else f"{self.results_dir}/{tag}")
        os.makedirs(rd, exist_ok=True)
        with open(f"{rd}/comparison.md", "w") as f:
            f.write(txt)
        with open(f"{rd}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        for v in ("conservative", "learned"):
            for ext in ("md", "json"):
                s = f"{self.repo}/{base}_{v}/scene_benchmark.{ext}"
                if os.path.exists(s):
                    shutil.copy(s, f"{rd}/{v}.{ext}")
        print(f"\n===== TRAIN {matcher_scene} -> TEST {test_scene} (unseen) =====")
        print(txt)
        print(f"[records saved] {rd}/  (comparison.md, summary.json, conservative.*, learned.*)")
        return summary

    def _combined_record(self, train, test, base):
        cons = json.load(open(f"{self.repo}/{base}_conservative/scene_benchmark.json"))
        lrn = json.load(open(f"{self.repo}/{base}_learned/scene_benchmark.json"))

        def f(v):
            return f"{v:.3f}" if isinstance(v, float) and abs(v) < 1 else f"{v:.0f}"

        L = [f"# {train} (train) -> {test} (test) : raw vs conservative vs learned", "",
             "| camera | metric | raw | conservative | learned |",
             "| --- | --- | ---: | ---: | ---: |"]
        for cam in cons["per_camera"]:
            for k, lbl in _KEYS:
                raw = cons["per_camera"][cam]["metrics"]["baseline"][k]
                c = cons["per_camera"][cam]["metrics"]["repaired"][k]
                l = lrn["per_camera"][cam]["metrics"]["repaired"][k] if cam in lrn["per_camera"] else float("nan")
                L.append(f"| {cam} | {lbl} | {f(raw)} | {f(c)} | {f(l)} |")
        ab, ac, al = cons["aggregate"]["baseline"], cons["aggregate"]["repaired"], lrn["aggregate"]["repaired"]
        L += ["", "## Aggregate", "", "| metric | raw | conservative | learned |", "| --- | ---: | ---: | ---: |"]
        for k, lbl in _KEYS:
            L.append(f"| {lbl} | {f(ab[k])} | {f(ac[k])} | {f(al[k])} |")
        gc = cons["aggregate"].get("gap_counts", {})
        if gc:
            tot = sum(gc.values()) or 1
            L += ["", "## Internal-gap root cause (raw)", "", "| cause | count | % |", "| --- | ---: | ---: |"]
            for cause, cnt in gc.items():
                L.append(f"| {cause} | {cnt} | {100 * cnt / tot:.1f}% |")
        txt = "\n".join(L) + "\n"
        summary = {"train": train, "test": test,
                   "aggregate": {"raw": ab, "conservative": ac, "learned": al},
                   "gap_root_cause": gc}
        return txt, summary

"""Generate easy_sm flow diagrams as .excalidraw files, laid out with Graphviz (dot -Tjson).

Usage: python3 make_diagrams.py OUTPUT_DIR
"""

import json
import random
import subprocess
import sys
from pathlib import Path

STYLE = {  # kind: (shape, stroke, background, stroke style)
    "fn": ("rectangle", "#0f6e5c", "#e4f1ed", "solid"),
    "check": ("diamond", "#b7791f", "#fff3d6", "solid"),
    "fail": ("rectangle", "#b42318", "#fde3df", "solid"),
    "ok": ("rectangle", "#2f7d32", "#e2f3df", "solid"),
    "box": ("rectangle", "#6b3fa0", "#ece7f5", "solid"),
    "aws": ("rectangle", "#3b5bab", "#e6ebf6", "solid"),
    "user": ("rectangle", "#18211d", "#ffffff", "dashed"),
    "note": ("rectangle", "#5b6661", "#f5f7f6", "dotted"),
}
FONT = 16
rand = random.Random(7)


def _id() -> str:
    return "".join(rand.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(16))


def _base(kind: str, x: float, y: float, w: float, h: float, stroke: str, bg: str, style: str = "solid") -> dict:
    return {
        "id": _id(), "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
        "strokeColor": stroke, "backgroundColor": bg, "fillStyle": "solid", "strokeWidth": 2,
        "strokeStyle": style, "roughness": 0, "opacity": 100, "groupIds": [], "frameId": None,
        "roundness": {"type": 3} if kind == "rectangle" else ({"type": 2} if kind in ("diamond", "arrow") else None),
        "seed": rand.randint(1, 2**31), "version": 1, "versionNonce": rand.randint(1, 2**31),
        "isDeleted": False, "boundElements": [], "updated": 1, "link": None, "locked": False,
    }


def _text(text: str, x: float, y: float, size: int = FONT, container: str | None = None,
          align: str = "center", color: str = "#18211d", w: float | None = None, h: float | None = None) -> dict:
    lines = text.split("\n")
    tw = w or max(len(line) for line in lines) * size * 0.56
    th = h or len(lines) * size * 1.25
    el = _base("text", x, y, tw, th, color, "transparent")
    el.update({"text": text, "originalText": text, "fontSize": size, "fontFamily": 2, "textAlign": align,
               "verticalAlign": "middle" if container else "top", "containerId": container,
               "autoResize": True, "lineHeight": 1.25, "roundness": None})
    return el


def _dot_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render(d: dict) -> dict:
    """d = {title, caption, rankdir, nodes: [(id, label, kind)], edges: [(a, b, label, dashed)], clusters}"""
    lines = [f'digraph G {{ rankdir={d.get("rankdir", "TB")}; nodesep=0.45; ranksep=0.55; pad=0.3;',
             'node [fontname="Helvetica", fontsize=15, margin="0.22,0.12"];',
             'edge [fontname="Helvetica", fontsize=13];']
    kinds = {}
    for nid, label, kind in d["nodes"]:
        kinds[nid] = kind
        shape = {"check": "diamond", "fail": "box", "ok": "box"}.get(kind, "box")
        lines.append(f'"{nid}" [label="{_dot_label(label)}", shape={shape}];')
    for i, (cid, (clabel, members)) in enumerate(d.get("clusters", {}).items()):
        lines.append(f'subgraph "cluster_{cid}" {{ label="{_dot_label(clabel)}"; fontname="Helvetica"; fontsize=17; margin=18;')
        lines += [f'"{m}";' for m in members]
        lines.append("}")
    for a, b, label, dashed in d["edges"]:
        attrs = []
        if label:
            attrs.append(f'label="{_dot_label(label)}"')
        if dashed:
            attrs.append("style=dashed")
        lines.append(f'"{a}" -> "{b}" [{", ".join(attrs)}];')
    lines.append("}")
    layout = json.loads(subprocess.run(["dot", "-Tjson"], input="\n".join(lines), text=True,
                                       capture_output=True, check=True).stdout)

    _, _, gw, gh = (float(v) for v in layout["bb"].split(","))
    top = 90.0  # room for the title
    flip = lambda y: gh - y + top  # noqa: E731  Graphviz y grows upwards

    elements: list[dict] = []
    elements.append(_text(d["title"], 20, 20, size=28, align="left"))
    if d.get("subtitle"):
        elements.append(_text(d["subtitle"], 20, 58, size=15, align="left", color="#56635d"))

    by_gvid: dict[int, dict] = {}
    for obj in layout.get("objects", []):
        if "nodes" in obj and "bb" in obj:  # a cluster
            x1, y1, x2, y2 = (float(v) for v in obj["bb"].split(","))
            rect = _base("rectangle", x1, flip(y2), x2 - x1, y2 - y1, "#8a9a92", "#f5f7f6", "dashed")
            rect["roundness"] = None
            elements.append(rect)
            elements.append(_text(obj["label"], x1 + 10, flip(y2) + 6, size=17, align="left", color="#3a4741"))
    for obj in layout.get("objects", []):
        if "nodes" in obj:
            continue
        nid = obj["name"]
        kind = kinds[nid]
        shape, stroke, bg, style = STYLE[kind]
        cx, cy = (float(v) for v in obj["pos"].split(","))
        w, h = float(obj["width"]) * 72, float(obj["height"]) * 72
        el = _base(shape, cx - w / 2, flip(cy) - h / 2, w, h, stroke, bg, style)
        label = next(n[1] for n in d["nodes"] if n[0] == nid)
        txt = _text(label, el["x"], el["y"], container=el["id"], w=w * (0.7 if shape == "diamond" else 0.92),
                    h=len(label.split("\n")) * FONT * 1.25)
        txt["x"] = cx - txt["width"] / 2
        txt["y"] = flip(cy) - txt["height"] / 2
        el["boundElements"].append({"id": txt["id"], "type": "text"})
        elements += [el, txt]
        by_gvid[obj["_gvid"]] = el

    for e, (a, b, label, dashed) in zip(layout.get("edges", []), d["edges"]):
        parts = e["pos"].split()
        end = None
        if parts[0].startswith("e,"):
            end = tuple(float(v) for v in parts[0][2:].split(","))
            parts = parts[1:]
        pts = [tuple(float(v) for v in p.split(",")) for p in parts]
        path = [pts[0]]
        for i in range(0, len(pts) - 3, 3):
            p0, p1, p2, p3 = pts[i:i + 4]
            for t in (0.25, 0.5, 0.75, 1.0):
                u = 1 - t
                path.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                             u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
        if end:
            path.append(end)
        path = [(x, flip(y)) for x, y in path]
        x0, y0 = path[0]
        arrow = _base("arrow", x0, y0, 1, 1, "#4b5751", "transparent", "dashed" if dashed else "solid")
        arrow.update({
            "points": [[x - x0, y - y0] for x, y in path], "lastCommittedPoint": None,
            "startArrowhead": None, "endArrowhead": "arrow", "elbowed": False,
            "startBinding": {"elementId": by_gvid[e["tail"]]["id"], "focus": 0, "gap": 4},
            "endBinding": {"elementId": by_gvid[e["head"]]["id"], "focus": 0, "gap": 4},
        })
        xs, ys = [p[0] for p in arrow["points"]], [p[1] for p in arrow["points"]]
        arrow["width"], arrow["height"] = max(xs) - min(xs) or 1, max(ys) - min(ys) or 1
        by_gvid[e["tail"]]["boundElements"].append({"id": arrow["id"], "type": "arrow"})
        by_gvid[e["head"]]["boundElements"].append({"id": arrow["id"], "type": "arrow"})
        elements.append(arrow)
        if label and "lp" in e:
            lx, ly = (float(v) for v in e["lp"].split(","))
            t = _text(label, 0, 0, size=13, color="#3a4741")
            t["x"], t["y"] = lx - t["width"] / 2, flip(ly) - t["height"] / 2
            t["backgroundColor"] = "#ffffff"
            elements.append(t)

    if d.get("caption"):
        elements.append(_text(d["caption"], 20, gh + top + 30, size=15, align="left", color="#56635d"))
    return {"type": "excalidraw", "version": 2, "source": "easy_sm make_diagrams.py",
            "elements": elements, "appState": {"viewBackgroundColor": "#fcfdfc", "gridSize": None}, "files": {}}


def N(nid: str, label: str, kind: str = "fn") -> tuple:
    return (nid, label, kind)


def E(a: str, b: str, label: str | None = None, dashed: bool = False) -> tuple:
    return (a, b, label, dashed)


DIAGRAMS: dict[str, dict] = {}

DIAGRAMS["01-overview"] = dict(
    title="Overview: where the code runs", rankdir="LR",
    subtitle="__main__.py · config.py · init.py · image.py · local.py · cloud.py · runtime/",
    nodes=[N("cli", "__main__.py\ncommands, errors to stderr"), N("cfg", "config.py\nload, resolve"),
           N("ini", "init.py\ncreate_project"), N("img", "image.py\nhash, build"), N("loc", "local.py\ndocker run argv"),
           N("cld", "cloud.py\npush, jobs, logs"), N("scf", "scaffold/\ntemplate files"),
           N("rt", "runtime/\ntrain, serve, run", "box"), N("usr", "your src/\ntrain(), Handler", "user"),
           N("ecr", "ECR\nimages", "aws"), N("sm", "SageMaker\njobs", "aws"), N("s3", "S3\ndata, models", "aws"),
           N("cw", "CloudWatch\nlogs", "aws")],
    clusters={"host": ("Your machine", ["cli", "cfg", "ini", "img", "loc", "cld", "scf"]),
              "image": ("Docker image (local Docker or SageMaker)", ["rt", "usr"]),
              "aws": ("AWS", ["ecr", "sm", "s3", "cw"])},
    edges=[E("cli", "cfg", "reads settings"), E("cli", "ini"), E("ini", "scf", "copies", True), E("cli", "img", "build_image"),
           E("img", "rt", "copied into image", True), E("cli", "loc"), E("loc", "rt", "docker run"), E("cli", "cld"),
           E("cld", "ecr", "push"), E("cld", "sm", "Create*Job"), E("cld", "cw", "polls logs"), E("sm", "ecr", "pulls image"),
           E("sm", "rt", "starts"), E("sm", "s3", "input in, model.tar.gz out"), E("rt", "usr", "imports")],
    caption="The same image runs your code in local Docker and on SageMaker. The runtime reaches the image only by being\ncopied in at build time, which is why upgrading easy_sm changes the image tag.")

DIAGRAMS["02-workflow"] = dict(
    title="Workflow: one project from start to cloud", rankdir="TB",
    subtitle="the order you normally run commands, and what each talks to",
    nodes=[N("i", "1  easy_sm init churn\nwrites easy_sm.toml, pyproject.toml, src/, data/"),
           N("tl", "2  easy_sm train --local\nbuild if code changed → docker run … train"),
           N("ml", ".easy_sm/model/model.json", "ok"),
           N("sv", "3  easy_sm endpoints serve --local\ndocker run … serve on 127.0.0.1"),
           N("iv", "4  easy_sm endpoints invoke --local -d …\nPOST /invocations → JSON"),
           N("tc", "5  easy_sm train\ncheck S3 input → push if tag missing → CreateTrainingJob"),
           N("jn", "stdout: job name", "ok"),
           N("smr", "SageMaker pulls image from ECR,\ncopies S3 input, runs train()", "aws"),
           N("fol", "easy_sm follows: status + logs\nevery 5 s, to stderr"),
           N("mod", "model.tar.gz in S3\nexit 0 (or 1 + FailureReason)", "ok"),
           N("rc", "6  easy_sm run clean.py --date …\nprocessing job, same image"),
           N("dep", "next (step 6): endpoints deploy --latest\nuses this job's model AND image", "note")],
    clusters={"local": ("Local: Docker only, no AWS", ["i", "tl", "ml", "sv", "iv"]),
              "cloud": ("Cloud: SageMaker", ["tc", "jn", "smr", "fol", "mod", "rc", "dep"])},
    edges=[E("i", "tl"), E("tl", "ml"), E("ml", "sv"), E("sv", "iv", "second terminal"), E("iv", "tc", "happy with it"),
           E("tc", "jn"), E("tc", "smr"), E("jn", "fol"), E("smr", "fol", "CloudWatch"), E("fol", "mod"),
           E("mod", "rc"), E("mod", "dep", None, True)],
    caption="The cloud half reuses the image built locally: ECR receives it once per tag, SageMaker pulls it for every job.")

DIAGRAMS["03-calls-setup"] = dict(
    title="Call map: init, env, build", rankdir="TB", subtitle="every function these commands reach, grouped by file",
    nodes=[N("cb", "main\n--config → ctx.obj"), N("lc", "load_config"), N("fl", "fail\nstderr + exit", "fail"),
           N("ci", "init"), N("ce", "env"), N("cbu", "build"), N("bi", "build_image"),
           N("ld", "load"), N("fla", "_flatten"), N("cap", "check_app_name"), N("rs", "Config.resolve"), N("gt", "Config.get"),
           N("cp", "create_project"), N("af", "app_name_from_folder"), N("sf", "scaffold_files\n(walk)"),
           N("b", "build"), N("imf", "image_files"), N("fo", "_folder"), N("pv", "python_version"), N("rd", "_read"),
           N("dfi", "dockerfile"), N("ih", "image_hash"), N("it", "image_tag"), N("wc", "write_context"),
           N("bc", "build_command"), N("rn", "_run\nuv lock, docker buildx"), N("sc", "_succeeds\ndocker image inspect"),
           N("se", "session", "aws"), N("sk", "session_kwargs", "aws"), N("ai", "account_id", "aws")],
    clusters={"m": ("__main__.py", ["cb", "lc", "fl", "ci", "ce", "cbu", "bi"]),
              "c": ("config.py", ["ld", "fla", "cap", "rs", "gt"]),
              "i": ("init.py", ["cp", "af", "sf"]),
              "g": ("image.py", ["b", "imf", "fo", "pv", "rd", "dfi", "ih", "it", "wc", "bc", "rn", "sc"]),
              "k": ("cloud.py", ["se", "sk", "ai"])},
    edges=[E("cb", "lc"), E("lc", "ld"), E("ld", "fla"), E("ld", "cap"), E("lc", "fl", "ConfigError", True),
           E("ci", "cp"), E("cp", "af"), E("cp", "cap"), E("cp", "sf"),
           E("ce", "lc"), E("ce", "it"), E("ce", "se"), E("ce", "ai"), E("ce", "rs"),
           E("it", "imf"), E("it", "ih"), E("se", "sk"), E("se", "gt"), E("gt", "rs"),
           E("cbu", "lc"), E("cbu", "bi"), E("bi", "b"), E("bi", "fl", "ImageError", True),
           E("b", "rn", "no uv.lock / build"), E("b", "imf"), E("b", "ih"), E("b", "sc", "tag exists?"),
           E("b", "wc"), E("b", "bc"), E("imf", "fo"), E("imf", "pv"), E("pv", "rd"), E("imf", "dfi")],
    caption="image_tag and build share image_files and image_hash, so `easy_sm env` shows exactly the tag the next build produces.\nLibrary errors become a stderr message and exit 1 in one place: fail.")

DIAGRAMS["04-calls-local"] = dict(
    title="Call map: local commands", rankdir="TB",
    subtitle="train --local, run --local, endpoints serve --local, endpoints invoke --local",
    nodes=[N("tr", "train --local"), N("ru", "run --local"), N("sv", "serve"), N("iv", "invoke"),
           N("rco", "reject_cloud_options"), N("bi", "build_image"), N("doe", "docker_or_exit"), N("fl", "fail", "fail"),
           N("ldr", "local_dirs\n.easy_sm + .gitignore"), N("em", "empty"), N("dr", "docker_run\ndocker run --rm --platform\n-e AWS_… names, no values"),
           N("tc", "train_command"), N("rc", "run_command"), N("scm", "serve_command"), N("pe", "parse_env"),
           N("cs", "check_script"), N("rb", "request_body\ntext or @file"), N("li", "invoke\nurllib POST"),
           N("dk", "docker\nsubprocess, stdout → stderr\nsecrets only in its environment"), N("g", "image.build"),
           N("cc", "cloud.container_credentials\nboto3 chain, like the AWS CLI", "aws")],
    clusters={"m": ("__main__.py", ["tr", "ru", "sv", "iv", "rco", "bi", "doe", "fl"]),
              "l": ("local.py", ["ldr", "em", "dr", "tc", "rc", "scm", "pe", "cs", "rb", "li", "dk"])},
    edges=[E("tr", "rco"), E("tr", "bi"), E("tr", "ldr"), E("tr", "em"), E("tr", "tc"), E("tc", "dr"), E("tr", "doe"),
           E("ru", "cs"), E("ru", "pe"), E("ru", "rco"), E("ru", "bi"), E("ru", "ldr"), E("ru", "rc"), E("rc", "dr"),
           E("ru", "doe"), E("sv", "ldr"), E("sv", "bi"), E("sv", "scm"), E("scm", "dr"), E("sv", "doe"),
           E("iv", "rb"), E("iv", "li"), E("bi", "g"), E("doe", "dk"), E("doe", "fl", "exit code ≠ 0", True),
           E("rco", "fl", "exit 2", True), E("li", "fl", "nothing listening", True),
           E("tr", "cc"), E("ru", "cc"), E("sv", "cc"), E("cc", "dr", "-e NAME only", True)],
    caption="The *_command functions only return lists; docker is the one function that runs anything.\nThat split is why the docker commands are tested without Docker.")

DIAGRAMS["05-calls-cloud"] = dict(
    title="Call map: cloud commands", rankdir="TB", subtitle="push, train and run without --local",
    nodes=[N("pu", "push"), N("tr", "train"), N("ru", "run"), N("s3d", "s3_default"), N("rcj", "run_cloud_job"),
           N("bi", "build_image"), N("fl", "fail", "fail"), N("lcl", "local.check_script\nlocal.parse_env"),
           N("se", "session", "aws"), N("sk", "session_kwargs", "aws"), N("ai", "account_id\nSTS", "aws"),
           N("db", "default_bucket", "aws"), N("eb", "ensure_bucket", "aws"), N("chk", "check_s3_has_data", "aws"),
           N("sp", "split_s3", "aws"), N("pt", "parse_timeout", "aws"), N("tjr", "training_job_request", "aws"),
           N("pjr", "processing_job_request", "aws"), N("cpu", "push\nECR", "aws"), N("eu", "ecr_uri", "aws"),
           N("dd", "_docker\nlogin, tag, push", "aws"), N("jn", "job_name", "aws"), N("nw", "now", "aws"),
           N("st", "start\nCreate*Job", "aws"), N("fo", "follow", "aws"), N("de", "describe", "aws"),
           N("nl", "new_log_lines\nCloudWatch", "aws"), N("ho", "host_of", "aws")],
    clusters={"m": ("__main__.py", ["pu", "tr", "ru", "s3d", "rcj", "bi", "fl"]),
              "k1": ("cloud.py · session", ["se", "sk", "ai"]),
              "k2": ("cloud.py · S3", ["db", "eb", "chk", "sp"]),
              "k3": ("cloud.py · ECR", ["cpu", "eu", "dd"]),
              "k4": ("cloud.py · requests (pure)", ["pt", "tjr", "pjr", "jn", "nw"]),
              "k5": ("cloud.py · run and watch", ["st", "fo", "de", "nl", "ho"])},
    edges=[E("pu", "bi"), E("pu", "se"), E("pu", "ai"), E("pu", "cpu"), E("tr", "s3d"), E("s3d", "se"),
           E("s3d", "ai"), E("s3d", "db"), E("s3d", "eb"), E("tr", "chk"), E("chk", "sp"), E("tr", "pt"),
           E("tr", "tjr"), E("tr", "rcj"), E("ru", "lcl"), E("ru", "pt"), E("ru", "pjr"), E("ru", "rcj"),
           E("rcj", "bi"), E("rcj", "se"), E("se", "sk"), E("rcj", "ai"), E("rcj", "cpu"), E("cpu", "eu"),
           E("cpu", "dd"), E("rcj", "jn"), E("jn", "nw"), E("rcj", "st"), E("rcj", "fo"), E("fo", "de"),
           E("fo", "nl"), E("nl", "ho"), E("rcj", "fl", "failed; 130 on Ctrl-C", True)],
    caption="train and run differ only in the request they build; run_cloud_job receives that builder as a function and does the rest for both.")

DIAGRAMS["06-calls-runtime"] = dict(
    title="Call map: runtime inside the image", rankdir="LR",
    subtitle="runtime/__main__.py · __init__.py · train.py · serve.py · run.py (standard library only)",
    nodes=[N("ep", "ENTRYPOINT\npython -m runtime ARGS", "box"), N("mn", "main(argv)", "box"),
           N("rtr", "run_training\nSRC, INPUT, MODEL, FAILURE paths", "box"), N("sv", "serve", "box"),
           N("rc", "run_command", "box"), N("us", "usage, exit 2", "fail"), N("lf", "load_from\nsrc/train.py → train", "box"),
           N("ut", "your train()", "user"), N("gc", "gunicorn_command", "box"),
           N("gu", "gunicorn\nreplaces the process", "box"), N("cr", "create()\nonce per worker", "box"),
           N("lf2", "load_from\nsrc/predict.py → Handler", "box"), N("uh", "your Handler(model_dir)", "user"),
           N("ma", "make_app(handler)", "box"), N("ap", "app(environ, start_response)", "box"),
           N("dc", "decode\nJSON or CSV", "box"), N("uh2", "handler(data)", "user"), N("re", "respond\nJSON + status", "box"),
           N("bc", "build_command\n.py → python", "box"), N("ex", "chdir src/, execvp\nreplaces the process", "box")],
    edges=[E("ep", "mn"), E("mn", "rtr", "train"), E("mn", "sv", "serve"), E("mn", "rc", "run …"), E("mn", "us", "other"),
           E("rtr", "lf"), E("lf", "ut"), E("sv", "gc"), E("gc", "gu"), E("gu", "cr"), E("cr", "lf2"), E("lf2", "uh"),
           E("cr", "ma"), E("ma", "ap"), E("ap", "dc"), E("ap", "uh2"), E("ap", "re"), E("rc", "bc"), E("bc", "ex")],
    caption="Dashed white boxes are your code. Two places replace the running process (gunicorn for serve, execvp for run),\nso signals and exit codes go straight to your program.")

DIAGRAMS["07-settings"] = dict(
    title="Settings: loading and resolving", rankdir="TB",
    subtitle="__main__.py: main, load_config · config.py: load, _flatten, check_app_name, Config.resolve",
    nodes=[N("a", "any command\n--config PATH (default ./easy_sm.toml)"), N("l", "config.load(path)"),
           N("c1", "file exists?", "check"), N("e1", "exit 1: run 'easy_sm init'", "fail"),
           N("c2", "valid TOML?", "check"), N("e2", "exit 1: parse error", "fail"),
           N("f", "_flatten\n[train] instance → train.instance"), N("c3", "every key\nin SETTINGS?", "check"),
           N("e3", "exit 1: unknown setting", "fail"), N("c4", "right type?", "check"),
           N("e4", "exit 1: should be int, got str", "fail"),
           N("c5", "app_name valid?\na-z 0-9 hyphens, ≤32", "check"), N("e5", "exit 1: invalid app_name", "fail"),
           N("c6", "run.env values\nstrings?", "check"), N("e6", "exit 1: must be strings", "fail"),
           N("ok", "Config(path, values)"), N("r", "Config.resolve(key, flag)"),
           N("r1", "flag given?", "check"), N("v1", "value, 'flag'", "ok"),
           N("r2", "env var set?\nAWS_REGION, SAGEMAKER_ROLE", "check"), N("v2", "value, env var name", "ok"),
           N("r3", "key in file?", "check"), N("v3", "value, 'easy_sm.toml'", "ok"), N("v4", "SETTINGS default", "ok")],
    edges=[E("a", "l"), E("l", "c1"), E("c1", "e1", "no"), E("c1", "c2", "yes"), E("c2", "e2", "no"), E("c2", "f", "yes"),
           E("f", "c3"), E("c3", "e3", "no"), E("c3", "c4", "yes"), E("c4", "e4", "no"), E("c4", "c5", "yes"),
           E("c5", "e5", "no"), E("c5", "c6", "yes"), E("c6", "e6", "no"), E("c6", "ok", "yes"), E("ok", "r"),
           E("r", "r1"), E("r1", "v1", "yes"), E("r1", "r2", "no"), E("r2", "v2", "yes"), E("r2", "r3", "no"),
           E("r3", "v3", "yes"), E("r3", "v4", "no")],
    caption="The file is checked once, strictly; then each setting is looked up flag → environment variable → file → default.\nAWS credentials never pass through here: boto3 finds them like the AWS CLI.")

DIAGRAMS["08-env"] = dict(
    title="env: what will actually be used", rankdir="TB", subtitle="__main__.py: env",
    nodes=[N("a", "easy_sm env"), N("lc", "load_config\n(settings flow)"), N("t", "image.image_tag\nimage_files + image_hash"),
           N("c1", "uv.lock and src/\npresent?", "check"), N("t1", "image  churn:<hash>  computed", "ok"),
           N("t2", "image  -  run 'easy_sm build'", "note"), N("s", "cloud.session + account_id\n(boto3 default chain)", "aws"),
           N("c2", "credentials and\nregion work?", "check"),
           N("a1", "aws.account  <id>  region, credentials: source\n(profile shown if from ~/.aws)", "ok"),
           N("a2", "aws.account  -  not available (error type)", "note"),
           N("r", "for every key in SETTINGS:\nConfig.resolve → value, source"), N("o", "one line per setting on stdout", "ok")],
    edges=[E("a", "lc"), E("lc", "t"), E("t", "c1"), E("c1", "t1", "yes"), E("c1", "t2", "no"), E("t1", "s"), E("t2", "s"),
           E("s", "c2"), E("c2", "a1", "yes"), E("c2", "a2", "no"), E("a1", "r"), E("a2", "r"), E("r", "o")],
    caption="env never fails because of AWS: it shows '-' instead, so it is safe to run first in any shell or CI job.")

DIAGRAMS["09-init"] = dict(
    title="init: creating a project", rankdir="LR",
    subtitle="__main__.py: init · init.py: app_name_from_folder, scaffold_files, create_project",
    nodes=[N("a", "easy_sm init PATH\n--name optional"), N("n", "--name given?", "check"),
           N("d", "app_name_from_folder\n'My Churn_Model' → my-churn-model"), N("v", "check_app_name\nvalid?", "check"),
           N("e1", "exit 1: invalid app_name", "fail"), N("s", "scaffold_files\nreplace __APP_NAME__"),
           N("x", "any target file\nexists?", "check"), N("e2", "exit 1: not overwriting\n(nothing written)", "fail"),
           N("w", "write all files,\nprint their paths", "ok")],
    edges=[E("a", "n"), E("n", "d", "no"), E("n", "v", "yes"), E("d", "v"), E("v", "e1", "no"), E("v", "s", "yes"),
           E("s", "x"), E("x", "e2", "yes"), E("x", "w", "no")],
    caption="Every target is checked before anything is written, so a failed init never leaves half a project.")

DIAGRAMS["10-build"] = dict(
    title="build: the image and its tag", rankdir="TB",
    subtitle="__main__.py: build_image · image.py: build, image_files, _folder, python_version, dockerfile, image_hash, write_context, build_command",
    nodes=[N("a", "build_image(cfg)\nused by build, push, train, run, serve"), N("u", "uv.lock exists?", "check"),
           N("ul", "uv lock\n(output to stderr)"), N("f", "image_files(project)"), N("s", "src/ exists?", "check"),
           N("e1", "exit 1: src not found", "fail"),
           N("g", "collect pyproject.toml, uv.lock,\nsrc/** and runtime/**\nskip __pycache__, .pyc, hidden files"),
           N("df", "own\nDockerfile?", "check"), N("own", "use it"),
           N("pv", "python version\nfound?", "check"), N("e2", "exit 1: set requires-python", "fail"),
           N("gen", "dockerfile(python)\nuv sync before COPY src/"),
           N("h", "image_hash\nsha256 of sorted names + contents → 12 chars"), N("t", "tag = app_name:hash"),
           N("i", "docker image inspect\ntag succeeds?", "check"), N("r", "reuse, return tag (~0.1 s)", "ok"),
           N("w", "write_context into temp folder"), N("b", "docker buildx build\n--platform linux/amd64 --load"),
           N("bok", "exit code 0?", "check"), N("e3", "exit 1: build failed", "fail"), N("r2", "return tag", "ok")],
    edges=[E("a", "u"), E("u", "ul", "no"), E("u", "f", "yes"), E("ul", "f"), E("f", "s"), E("s", "e1", "no"),
           E("s", "g", "yes"), E("g", "df"), E("df", "own", "yes"), E("df", "pv", "no"), E("pv", "e2", "no"),
           E("pv", "gen", "yes"), E("own", "h"), E("gen", "h"), E("h", "t"), E("t", "i"), E("i", "r", "yes"),
           E("i", "w", "no"), E("w", "b"), E("b", "bok"), E("bok", "e3", "no"), E("bok", "r2", "yes")],
    caption="One dictionary of files produces both the tag and the build folder, so the tag changes exactly when the image would.")

DIAGRAMS["11-what-rebuilds"] = dict(
    title="What rebuilds when you change something", rankdir="LR", subtitle="image.py: image_files decides; Docker's layer cache decides how much",
    nodes=[N("c1", "edit README, data/,\neasy_sm.toml"), N("c2", "edit src/train.py"), N("c3", "uv add pandas\n(uv.lock changes)"),
           N("c4", "upgrade easy_sm\n(runtime, uv, gunicorn pins)"), N("c5", "edit your own Dockerfile"),
           N("h1", "same tag", "check"), N("h2", "new tag", "check"),
           N("n1", "nothing rebuilds,\nnothing pushed", "ok"), N("l1", "only COPY src/ layer\n(seconds); push 1 small layer"),
           N("l2", "uv sync layer and after\n(minutes); push new layers"), N("l3", "from the first changed\nDockerfile line onwards"),
           N("j", "next job / local run uses it", "ok")],
    edges=[E("c1", "h1"), E("c2", "h2"), E("c3", "h2"), E("c4", "h2"), E("c5", "h2"), E("h1", "n1"),
           E("h2", "l1", "src/ only"), E("h2", "l2", "dependencies"), E("h2", "l3", "runtime or Dockerfile"),
           E("l1", "j"), E("l2", "j"), E("l3", "j")],
    caption="The tag is a hash of what goes into the image; Docker reuses every layer before the first changed input.")

DIAGRAMS["12-train-local"] = dict(
    title="train --local: training in local Docker", rankdir="TB",
    subtitle="__main__.py: train · local.py: local_dirs, empty, train_command, docker · runtime: run_training, load_from",
    nodes=[N("a", "easy_sm train --local\n--input DIR (default data/)"),
           N("c1", "cloud-only\noptions given?", "check"), N("e1", "exit 2: only apply to SageMaker jobs", "fail"),
           N("c2", "input folder\nexists?", "check"), N("e2", "exit 1: not found", "fail"), N("b", "build_image"),
           N("d", "local_dirs: .easy_sm/model, output\n.gitignore '*'; empty the model folder"),
           N("cmd", "train_command\ndata → /opt/ml/input/data/training (read-only)\nmodel → /opt/ml/model, output → /opt/ml/output\n-e AWS_ACCESS_KEY_ID … (names only; no ~/.aws)"),
           N("dk", "docker run (output to stderr)"), N("rt", "python -m runtime train", "box"),
           N("ld", "run_training: load_from(src, train.py)\ntrain(input_dir, model_dir)", "box"),
           N("x", "train()\nraised?", "check"),
           N("ff", "write /opt/ml/output/failure\n'Type: message' first, then traceback", "box"),
           N("e3", "container exits 1 → easy_sm exits 1", "fail"), N("ok", "exit 0, stdout: model folder", "ok")],
    clusters={"c": ("inside the container", ["rt", "ld", "x", "ff"])},
    edges=[E("a", "c1"), E("c1", "e1", "yes"), E("c1", "c2", "no"), E("c2", "e2", "no"), E("c2", "b", "yes"),
           E("b", "d"), E("d", "cmd"), E("cmd", "dk"), E("dk", "rt"), E("rt", "ld"), E("ld", "x"),
           E("x", "ff", "yes"), E("ff", "e3"), E("x", "ok", "no")],
    caption="Folders are mounted where SageMaker puts them, so train.py runs unchanged in both places.")

DIAGRAMS["13-run"] = dict(
    title="run: any command, locally or as a processing job", rankdir="TB",
    subtitle="__main__.py: run · local.py: check_script, parse_env, run_command · runtime/run.py · cloud.py: processing_job_request",
    nodes=[N("a", "easy_sm run [options] COMMAND...\noptions stop at the command's first word"),
           N("c1", ".py file missing\nfrom src/?", "check"), N("e1", "exit 1: src/clen.py not found", "fail"),
           N("c2", "every --env\nKEY=VALUE?", "check"), N("e2", "exit 1: must look like KEY=VALUE", "fail"),
           N("env", "env = [run] env overlaid by --env"), N("l", "--local?", "check"),
           N("c3", "cloud-only\noptions?", "check"), N("e3", "exit 2", "fail"), N("b", "build_image"),
           N("rc", "run_command: each var as its own -e K=V\ndata → /opt/ml/processing/input (ro)\n.easy_sm/output → /opt/ml/processing/output\nAWS credentials as -e NAME only"),
           N("dk", "docker run"), N("t", "timeout valid?\n3600, 90m, 2h, 1d", "check"), N("e4", "exit 1: invalid timeout", "fail"),
           N("cj", "run_cloud_job + processing_job_request\nContainerArguments: run COMMAND...\nenv values ≤ 256 chars"),
           N("sm", "SageMaker processing job", "aws"), N("rt", "python -m runtime run COMMAND...", "box"),
           N("bc", "first word\nends in .py?", "check"), N("py", "prepend python", "box"), N("as", "keep as is", "box"),
           N("ex", "chdir src/, os.execvp\nprocess becomes the command", "box"),
           N("out", "command's exit code = easy_sm's\n(local) or the job's (cloud)", "ok")],
    clusters={"c": ("inside the container", ["rt", "bc", "py", "as", "ex"])},
    edges=[E("a", "c1"), E("c1", "e1", "yes"), E("c1", "c2", "no"), E("c2", "e2", "no"), E("c2", "env", "yes"),
           E("env", "l"), E("l", "c3", "yes"), E("c3", "e3", "yes"), E("c3", "b", "no"), E("b", "rc"), E("rc", "dk"),
           E("l", "t", "no"), E("t", "e4", "no"), E("t", "cj", "yes"), E("cj", "sm", "see cloud jobs", True),
           E("dk", "rt"), E("sm", "rt"), E("rt", "bc"), E("bc", "py", "yes"), E("bc", "as", "no"),
           E("py", "ex"), E("as", "ex"), E("ex", "out")],
    caption="Cheap checks run before any build or machine. Because the runtime replaces itself with the command,\nsys.exit(3) in your script becomes exit 3 / a Failed job; nothing in between can swallow it.")

DIAGRAMS["14-serve-invoke"] = dict(
    title="serve and invoke: a local endpoint", rankdir="TB",
    subtitle="__main__.py: serve, invoke · local.py: serve_command, request_body, invoke · runtime/serve.py",
    nodes=[N("s", "easy_sm endpoints serve --local --port P"), N("s1", "--local?", "check"),
           N("se1", "exit 2: serve only runs locally", "fail"), N("s2", ".easy_sm/model\nempty?", "check"),
           N("se2", "exit 1: run 'train --local' first", "fail"),
           N("sb", "build_image, serve_command\n-p 127.0.0.1:P:8080, model read-only\nAWS credentials as -e NAME only"),
           N("g", "gunicorn: port 8080,\none worker per CPU, 60 s timeout", "box"),
           N("cr", "create() per worker\nHandler(/opt/ml/model) loaded once", "box"), N("app", "make_app(handler)\nwaits for requests", "box"),
           N("i", "easy_sm endpoints invoke --local\n-d DATA or @file"), N("rb", "request_body, POST\n127.0.0.1:P/invocations"),
           N("i1", "something\nanswering?", "check"), N("ie1", "exit 1: start it with serve --local", "fail"),
           N("rt", "route?", "check"), N("p200", "200 status ok", "ok"), N("p404", "404 no route", "fail"),
           N("dc", "JSON parses\nor CSV?", "check"), N("p400", "400 with the reason", "fail"),
           N("hd", "handler(data) and\njson.dumps succeed?", "check"), N("p500", "500, traceback in container logs", "fail"),
           N("p2", "200 JSON result", "ok"), N("io", "invoke prints body, exit 0", "ok"), N("ie2", "invoke: exit 1, 'HTTP 4xx/5xx …'", "fail")],
    clusters={"c": ("inside the container", ["g", "cr", "app", "rt", "dc", "hd"])},
    edges=[E("s", "s1"), E("s1", "se1", "no"), E("s1", "s2", "yes"), E("s2", "se2", "yes"), E("s2", "sb", "no"),
           E("sb", "g"), E("g", "cr"), E("cr", "app"), E("i", "rb"), E("rb", "i1"), E("i1", "ie1", "no"),
           E("i1", "app", "yes"), E("app", "rt"), E("rt", "p200", "GET /ping"), E("rt", "p404", "other"),
           E("rt", "dc", "POST /invocations"), E("dc", "p400", "no"), E("dc", "hd", "yes"), E("hd", "p500", "no"),
           E("hd", "p2", "yes"), E("p2", "io"), E("p400", "ie2"), E("p500", "ie2")],
    caption="Two terminals: serve runs in the foreground until Ctrl-C (which stops the container). CSV rows arrive as lists of strings; responses are JSON.")

DIAGRAMS["15-push"] = dict(
    title="push: image to ECR", rankdir="TB", subtitle="__main__.py: push · cloud.py: session, account_id, push, ecr_uri, _docker",
    nodes=[N("a", "easy_sm push\n(also step 1 of every cloud job)"), N("b", "build_image → churn:<hash>"),
           N("s", "session + account_id\nboto3 default chain", "aws"), N("r1", "ECR repository\n'churn' exists?", "check"),
           N("cr", "create_repository\nimmutable tags", "aws"), N("r2", "this tag already\nin ECR?", "check"),
           N("sk", "skip upload", "ok"), N("tok", "get_authorization_token", "aws"),
           N("lg", "docker login --password-stdin\n(password never on the command line)"), N("tg", "docker tag churn:<hash> <uri>"),
           N("ps", "docker push <uri>\n(only layers ECR lacks)"), N("c", "exit code 0?", "check"),
           N("e", "exit 1: 'docker push' failed", "fail"), N("ok", "stdout: <account>.dkr.ecr.<region>.amazonaws.com/churn:<hash>", "ok")],
    edges=[E("a", "b"), E("b", "s"), E("s", "r1"), E("r1", "cr", "no"), E("r1", "r2", "yes"), E("cr", "r2"),
           E("r2", "sk", "yes"), E("r2", "tok", "no"), E("tok", "lg"), E("lg", "tg"), E("tg", "ps"), E("ps", "c"),
           E("c", "e", "no"), E("c", "ok", "yes"), E("sk", "ok")],
    caption="Tags are content hashes, so ECR's immutable tags guarantee a tag always means the same code. Nothing is written to ~/.aws/config.")

DIAGRAMS["16-cloud-jobs"] = dict(
    title="Cloud jobs: train and run on SageMaker", rankdir="TB",
    subtitle="__main__.py: train, s3_default, run_cloud_job · cloud.py",
    nodes=[N("a", "easy_sm train (no --local)"),
           N("in", "input = --input → train.input →\ns3://sagemaker-<region>-<account>/<app>/data\noutput likewise …/models"),
           N("d1", "input folder has\nany object?", "check"), N("e1", "exit 1: no data, upload hint\n(no job started)", "fail"),
           N("to", "timeout\nvalid?", "check"), N("e2", "exit 1: invalid timeout", "fail"), N("j", "run_cloud_job"),
           N("run", "easy_sm run COMMAND\n(see run flow)"), N("b", "build_image"),
           N("se", "region known?\nflag, AWS_REGION, toml, boto3", "check"), N("e3", "exit 1: no AWS region", "fail"),
           N("ro", "role set?\nSAGEMAKER_ROLE / [aws] role", "check"), N("e4", "exit 1: no IAM role", "fail"),
           N("pu", "account_id → push\n(see push flow)", "aws"), N("nm", "job_name\napp-YYYYMMDD-HHMMSS-hash"),
           N("st", "create_training_job /\ncreate_processing_job", "aws"), N("out", "stdout: job name"),
           N("nw", "--no-wait?", "check"), N("ok1", "exit 0 now", "ok"),
           N("fl", "follow: every 5 s describe the job,\nstatus + new log lines to stderr\nas [algo-1] …  (silent with -q)"),
           N("cc", "Ctrl-C?", "check"), N("e5", "exit 130: stopped following, job keeps running\nhint: easy_sm jobs logs JOB -f / jobs cancel JOB", "fail"),
           N("ts", "Completed, Failed\nor Stopped?", "check"), N("ok2", "exit 0", "ok"),
           N("e6", "exit 1: name, status, FailureReason", "fail"),
           N("any", "any AWS error on the way\nClientError, BotoCoreError, CloudError", "note"), N("e7", "exit 1 with the message", "fail")],
    edges=[E("a", "in"), E("in", "d1"), E("d1", "e1", "no"), E("d1", "to", "yes"), E("to", "e2", "no"), E("to", "j", "yes"),
           E("run", "j"), E("j", "b"), E("b", "se"), E("se", "e3", "no"), E("se", "ro", "yes"), E("ro", "e4", "no"),
           E("ro", "pu", "yes"), E("pu", "nm"), E("nm", "st"), E("st", "out"), E("out", "nw"), E("nw", "ok1", "yes"),
           E("nw", "fl", "no"), E("fl", "cc"), E("cc", "e5", "yes"), E("cc", "ts", "no"), E("ts", "fl", "not yet"),
           E("ts", "ok2", "Completed"), E("ts", "e6", "Failed / Stopped"), E("any", "e7", None, True)],
    caption="Everything that can fail cheaply fails before a machine starts. Only the job name goes to stdout;\nwatching and running are separate, so losing the terminal never stops the job.")

DIAGRAMS["17-job-states"] = dict(
    title="Job states as easy_sm sees them", rankdir="LR", subtitle="cloud.py: describe, follow — the [status] lines on stderr",
    nodes=[N("cr", "CreateTrainingJob /\nCreateProcessingJob", "aws"), N("st", "InProgress\nStarting", "aws"),
           N("dl", "InProgress\nDownloading", "aws"), N("tr", "InProgress\nTraining", "aws"), N("up", "InProgress\nUploading", "aws"),
           N("sp", "Stopping", "aws"), N("co", "Completed\n→ exit 0", "ok"), N("fa", "Failed\n→ exit 1 + FailureReason", "fail"),
           N("so", "Stopped\n→ exit 1", "fail")],
    edges=[E("cr", "st"), E("st", "dl", "instance up, image pulled"), E("dl", "tr", "S3 input copied"),
           E("tr", "up", "train() returned"), E("up", "co", "model.tar.gz in S3"), E("tr", "fa", "train() raised / exit ≠ 0"),
           E("st", "fa", "bad image, role, quota"), E("tr", "sp", "stop-training-job / timeout"), E("sp", "so")],
    caption="Training jobs report the detailed InProgress phases; processing jobs report only InProgress, then a final status.")

DIAGRAMS["18-endpoints-deploy"] = dict(
    title="endpoints deploy: from a training job to a live endpoint", rankdir="TB",
    subtitle="__main__.py: deploy, endpoint_name · cloud.py: latest_job, job_source, resource_name, model_request, endpoint_config_request, deploy, wait_for_endpoint, deploy_outcome",
    nodes=[N("a", "easy_sm endpoints deploy [NAME]\n--latest | --job JOB | --model S3  [--dry-run]"),
           N("c1", "exactly one\nsource given?", "check"), N("e1", "exit 2: choose exactly one", "fail"),
           N("nm", "endpoint_name: NAME → [endpoint] name → app_name"), N("c2", "valid name?\nletters, digits, hyphens, ≤63", "check"),
           N("e2", "exit 1: invalid endpoint name", "fail"), N("c3", "role set?", "check"), N("e3", "exit 1: no IAM role", "fail"),
           N("src", "which source?", "check"),
           N("lj", "latest_job: newest Completed job\nwhose name matches <app>-…-<hash>", "aws"),
           N("js", "job_source: the job's model.tar.gz\nAND the image it trained with", "aws"),
           N("c4", "job Completed?", "check"), N("e4", "exit 1: job is Failed / not Completed", "fail"),
           N("md", "--model: S3 has the file?\nthen build + push CURRENT code", "aws"),
           N("rn", "resource_name: <endpoint>-<time>-<hash>\nnew model + config name every deploy"),
           N("rq", "model_request + endpoint_config_request\ninstance, or serverless memory/concurrency"),
           N("dr", "--dry-run?", "check"), N("dro", "JSON: endpoint, create|update,\nsource, model, endpoint_config", "ok"),
           N("ex", "endpoint exists?", "check"),
           N("cf", "its current config\nstill exists?", "check"), N("e5", "exit 1: config was deleted,\nSageMaker can't update it", "fail"),
           N("cr", "create_model, create_endpoint_config,\ncreate_endpoint", "aws"),
           N("up", "create_model, create_endpoint_config,\nupdate_endpoint (old ones kept)", "aws"),
           N("w", "wait_for_endpoint: every 15 s,\nstatus changes to stderr"),
           N("o", "deploy_outcome: InService\nON THE NEW CONFIG?", "check"),
           N("ok", "stdout: endpoint name, exit 0", "ok"),
           N("rb", "exit 1: rolled back (still on old config)\nor Failed with reason", "fail")],
    edges=[E("a", "c1"), E("c1", "e1", "no"), E("c1", "nm", "yes"), E("nm", "c2"), E("c2", "e2", "no"), E("c2", "c3", "yes"),
           E("c3", "e3", "no"), E("c3", "src", "yes"), E("src", "lj", "--latest"), E("lj", "js"), E("src", "js", "--job"),
           E("js", "c4"), E("c4", "e4", "no"), E("c4", "rn", "yes"), E("src", "md", "--model"), E("md", "rn"),
           E("rn", "rq"), E("rq", "dr"), E("dr", "dro", "yes"), E("dr", "ex", "no"), E("ex", "cr", "no"), E("ex", "cf", "yes"),
           E("cf", "e5", "no"), E("cf", "up", "yes"), E("cr", "w"), E("up", "w"), E("w", "o"), E("o", "ok", "yes"), E("o", "rb", "no")],
    caption="Deploying from a job serves the model with the exact code that trained it. Old models and configs are never\ndeleted: SageMaker needs the old config to roll back a failed update.")

DIAGRAMS["19-calls-endpoints-jobs"] = dict(
    title="Call map: endpoints, jobs, data, skill", rankdir="TB", subtitle="every function these commands reach",
    nodes=[N("dep", "endpoints deploy"), N("inv", "endpoints invoke"), N("els", "endpoints ls"), N("ed", "endpoints describe"),
           N("edl", "endpoints delete"), N("jls", "jobs ls"), N("jin", "jobs inspect"), N("jlo", "jobs logs"),
           N("jca", "jobs cancel"), N("jar", "jobs artifacts"), N("du", "data upload"), N("sk", "skill"),
           N("en", "endpoint_name"), N("ip", "image_plan\n(--dry-run)"), N("fl", "fail / emit_json", "note"),
           N("lj", "latest_job", "aws"), N("pl", "pick_latest", "aws"), N("io", "is_own", "aws"), N("js", "job_source", "aws"),
           N("rsn", "resource_name", "aws"), N("mr", "model_request", "aws"), N("ecr", "endpoint_config_request", "aws"),
           N("dp", "deploy", "aws"), N("fe", "find_endpoint", "aws"), N("we", "wait_for_endpoint", "aws"),
           N("do", "deploy_outcome", "aws"), N("ie", "invoke_endpoint", "aws"), N("le", "list_endpoints", "aws"),
           N("edt", "endpoint_details", "aws"), N("dl", "describe_lines", "aws"), N("de", "delete_endpoint", "aws"),
           N("lij", "list_jobs", "aws"), N("jr", "job_rows", "aws"), N("fd", "fmt_duration", "aws"), N("fj", "find_job", "aws"),
           N("il", "inspect_lines", "aws"), N("nl", "new_log_lines / follow", "aws"), N("sj", "stop_job", "aws"),
           N("up", "upload_plan", "aws"), N("upl", "upload", "aws"), N("lr", "local.request_body\nlocal.invoke"),
           N("smd", "skill.md\n(packaged)", "note")],
    clusters={"m": ("__main__.py", ["dep", "inv", "els", "ed", "edl", "jls", "jin", "jlo", "jca", "jar", "du", "sk", "en", "ip", "fl"]),
              "c1": ("cloud.py · endpoints", ["lj", "pl", "io", "js", "rsn", "mr", "ecr", "dp", "fe", "we", "do", "ie", "le", "edt", "dl", "de"]),
              "c2": ("cloud.py · jobs and data", ["lij", "jr", "fd", "fj", "il", "nl", "sj", "up", "upl"])},
    edges=[E("dep", "en"), E("dep", "lj"), E("lj", "pl"), E("pl", "io"), E("dep", "js"), E("dep", "ip", "--model --dry-run", True),
           E("dep", "rsn"), E("dep", "mr"), E("dep", "ecr"), E("dep", "fe", "--dry-run", True), E("dep", "dp"), E("dp", "fe"),
           E("dep", "we"), E("dep", "do"), E("inv", "en"), E("inv", "ie"), E("inv", "lr", "--local"), E("els", "le"),
           E("ed", "en"), E("ed", "edt"), E("ed", "dl"), E("edl", "de"), E("jls", "lij"), E("lij", "jr"), E("jr", "io"),
           E("jr", "fd"), E("jin", "fj"), E("jin", "il"), E("il", "fd"), E("jlo", "fj"), E("jlo", "nl"), E("jca", "fj"),
           E("jca", "sj"), E("jar", "fj"), E("du", "up"), E("du", "upl"), E("sk", "smd"),
           E("els", "fl", "--format json", True), E("jls", "fl", "--format json", True)],
    caption="is_own is the single rule for 'this job belongs to this app', shared by jobs ls and deploy --latest.")

DIAGRAMS["20-jobs-data"] = dict(
    title="jobs and data: the checks", rankdir="LR", subtitle="__main__.py: jobs ls, inspect, logs, cancel, artifacts · data upload",
    nodes=[N("fj", "find_job(name)"), N("t", "training job\nwith this name?", "check"), N("p", "processing job\nwith this name?", "check"),
           N("nf", "exit 1: no job named …", "fail"), N("k", "kind + description"),
           N("ins", "inspect: inspect_lines\nenv NAMES only", "ok"), N("lo", "logs: [host] lines on STDOUT\n-f: follow until it ends", "ok"),
           N("ca", "cancel: InProgress?", "check"), N("cae", "exit 1: only InProgress\njobs can be stopped", "fail"),
           N("cao", "stop_job, print name", "ok"), N("ar", "artifacts: training\nAND Completed?", "check"),
           N("are", "exit 1: no usable model\n(failed jobs still list a path)", "fail"), N("aro", "print model.tar.gz path", "ok"),
           N("du", "data upload [DIR] [S3]"), N("dd", "DIR exists?", "check"), N("dde", "exit 1: not found", "fail"),
           N("tg", "target: S3 arg → train.input →\ndefault bucket/<app>/data"), N("pl", "upload_plan: every file,\nhidden files skipped"),
           N("pe", "any files?", "check"), N("pee", "exit 1: nothing to upload", "fail"), N("ul", "upload each (progress on stderr),\nprint the S3 folder", "ok")],
    edges=[E("fj", "t"), E("t", "k", "yes"), E("t", "p", "no"), E("p", "k", "yes"), E("p", "nf", "no"),
           E("k", "ins"), E("k", "lo"), E("k", "ca"), E("ca", "cae", "no"), E("ca", "cao", "yes"), E("k", "ar"),
           E("ar", "are", "no"), E("ar", "aro", "yes"),
           E("du", "dd"), E("dd", "dde", "no"), E("dd", "tg", "yes"), E("tg", "pl"), E("pl", "pe"), E("pe", "pee", "no"), E("pe", "ul", "yes")],
    caption="Jobs are named the same way whatever their kind, so you never pass the kind: easy_sm asks for a training job first.")

DIAGRAMS["21-dry-run"] = dict(
    title="--dry-run: what it reads and what it never does", rankdir="LR",
    subtitle="train, run, push, endpoints deploy (cloud and --local) · __main__.py: image_plan, run_cloud_job(dry_run)",
    nodes=[N("a", "COMMAND --dry-run"),
           N("r1", "read easy_sm.toml, settings", "ok"), N("r2", "image_tag from files\n(fails if no uv.lock)", "ok"),
           N("r3", "docker image inspect: build needed?", "ok"), N("r4", "ECR describe_images: push needed?", "ok"),
           N("r5", "S3 list: input has data?", "ok"), N("r6", "SageMaker: latest job, endpoint exists?", "ok"),
           N("r7", "STS: account; boto3: credential NAMES", "ok"),
           N("n1", "uv lock (writes uv.lock)", "fail"), N("n2", "docker build / push", "fail"),
           N("n3", "create the default bucket", "fail"), N("n4", "create .easy_sm/ folders", "fail"),
           N("n5", "create jobs, models, configs, endpoints", "fail"), N("n6", "print secret values", "fail"),
           N("o", "stdout JSON: command, dry_run, image {tag, build, uri, push},\nrequest | docker argv | model + endpoint_config", "ok")],
    clusters={"reads": ("looks: allowed", ["r1", "r2", "r3", "r4", "r5", "r6", "r7"]),
              "never": ("changes: never", ["n1", "n2", "n3", "n4", "n5", "n6"])},
    edges=[E("a", "r1"), E("a", "r2"), E("a", "r3"), E("a", "r4"), E("a", "r5"), E("a", "r6"), E("a", "r7"),
           E("a", "n1", "never", True), E("a", "n2", "never", True), E("a", "n3", "never", True), E("a", "n4", "never", True),
           E("a", "n5", "never", True), E("a", "n6", "never", True), E("r7", "o")],
    caption="The test for any step: afterwards, is anything in your account, your Docker or your files different? If so, a dry run must not do it.")

DIAGRAMS["22-credentials"] = dict(
    title="How AWS credentials reach a local container", rankdir="LR",
    subtitle="cloud.py: container_credentials, credential_env · local.py: docker_run(passed), docker(command, secrets)",
    nodes=[N("sh", "your shell\nAWS_PROFILE / keys / SSO / OIDC"), N("bc", "boto3 default chain\n(same as the AWS CLI)", "aws"),
           N("c", "credentials found?", "check"), N("no", "pass nothing\n(warning if a login is broken)", "note"),
           N("fz", "get_frozen_credentials\nrefreshes SSO / assumed role", "aws"),
           N("ce", "credential_env\nAWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,\nAWS_SESSION_TOKEN (if temporary), AWS_REGION", "aws"),
           N("argv", "docker argv: -e AWS_ACCESS_KEY_ID …\nNAMES only (visible in ps, --dry-run)"),
           N("penv", "docker process environment:\nthe VALUES (never printed)"),
           N("ct", "container: SDKs read AWS_* vars\nno ~/.aws, no AWS_PROFILE", "box"),
           N("aws", "~/.aws (all profiles, config, SSO cache)\nnot mounted, not writable", "fail")],
    edges=[E("sh", "bc"), E("bc", "c"), E("c", "no", "no"), E("c", "fz", "yes"), E("fz", "ce"), E("ce", "argv", "names"),
           E("ce", "penv", "values"), E("argv", "ct"), E("penv", "ct", "docker copies NAME's value"), E("aws", "ct", "never", True)],
    caption="The container gets exactly the one set of credentials you are using, read-only, and nothing else from ~/.aws (least privilege).")


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, d in DIAGRAMS.items():
        (out / f"{name}.excalidraw").write_text(json.dumps(render(d), indent=1))
        print(out / f"{name}.excalidraw")


if __name__ == "__main__":
    main()

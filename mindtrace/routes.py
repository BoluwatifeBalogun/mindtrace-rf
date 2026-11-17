"""HTTP routes: authentication, upload, analysis, results, history, model report."""
from __future__ import annotations

import csv
import io
import json
import re
import secrets
import uuid
from functools import wraps
from pathlib import Path

import cv2
from flask import (Blueprint, Response, abort, current_app, flash, g, jsonify, redirect, render_template,
                   request, send_from_directory, session, url_for)
from sqlalchemy import func

from . import nigeria
from .ml import features as feat
from .ml import preprocess
from .models import MRIImage, Prediction, User, db, now

bp = Blueprint("main", __name__)
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,40}$")
PER_PAGE = 9


# ---------------------------------------------------------------- helpers ----
@bp.before_app_request
def load_user():
    g.user = db.session.get(User, session["uid"]) if "uid" in session else None
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    if request.method == "POST" and request.form.get("csrf_token") != session["csrf"]:
        abort(400, "Your session expired. Go back, refresh the page and try again.")


@bp.app_context_processor
def inject():
    clf = current_app.classifier
    return {"user": g.get("user"), "csrf_token": session.get("csrf", ""), "M": clf.metrics,
            "pct": lambda v, d=1: f"{v * 100:.{d}f}"}


@bp.app_template_filter("num")
def num(v):
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.4f}"


@bp.app_template_filter("when")
def when(dt):
    return dt.strftime("%d %b %Y, %H:%M")


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if g.user is None:
            flash("Sign in to continue.", "info")
            return redirect(url_for("main.signin", next=request.path))
        return view(*a, **kw)
    return wrapped


def own_prediction(pid: int) -> Prediction:
    p = db.session.get(Prediction, pid)
    if p is None or p.user_id != g.user.id:
        abort(404)
    return p


# ----------------------------------------------------------------- public ----
@bp.get("/")
def landing():
    return render_template("landing.html", zones=nigeria.ZONES, hero=hero_sample())


_hero = {}


def hero_sample() -> dict:
    """Runs the real pipeline once on the bundled sample slice shown in the hero."""
    if not _hero:
        path = Path(current_app.static_folder) / "img" / "hero-scan.png"
        r = current_app.classifier.analyse(path.read_bytes())
        f = r["features"]
        _hero.update(short=r["short"], prob=r["probability"], ms=r["timing_ms"]["total"],
                     contrast=f["glcm_contrast_d1"], homog=f["glcm_homogeneity_d1"],
                     energy=f["glcm_energy_d1"], corr=f["glcm_correlation_d1"], mean=f["int_mean"],
                     entropy=f["glcm_entropy_d1"], std=f["int_std"], prob_ad=r["prob_ad"],
                     top=current_app.classifier.top_features(f, 4))
    return _hero


@bp.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user:
        return redirect(url_for("main.overview"))
    form, errors = request.form, {}
    if request.method == "POST":
        username = form.get("username", "").strip()
        full_name = " ".join(form.get("full_name", "").split())
        password = form.get("password", "")
        if len(full_name) < 3:
            errors["full_name"] = "Enter your full name."
        if not USERNAME_RE.match(username):
            errors["username"] = "Use 3 to 40 letters, numbers, dots or underscores."
        elif User.query.filter(func.lower(User.username) == username.lower()).first():
            errors["username"] = "That username is taken."
        if len(password) < 8:
            errors["password"] = "Use at least 8 characters."
        elif password != form.get("confirm", ""):
            errors["confirm"] = "The two passwords do not match."
        if form.get("role") not in nigeria.ROLES:
            errors["role"] = "Choose a role."
        if form.get("state") not in nigeria.STATES:
            errors["state"] = "Choose a state."
        if len(form.get("facility", "").strip()) < 3:
            errors["facility"] = "Enter your hospital, clinic or institution."
        if not errors:
            u = User(username=username, full_name=full_name, role=form["role"],
                     facility=form["facility"].strip()[:160], state=form["state"])
            u.set_password(password)
            db.session.add(u)
            db.session.commit()
            flash("Account created. Sign in to start.", "success")
            return redirect(url_for("main.signin"))
    return render_template("signup.html", errors=errors, form=form, states=nigeria.STATES,
                           roles=nigeria.ROLES), (400 if errors else 200)


@bp.route("/signin", methods=["GET", "POST"])
def signin():
    if g.user:
        return redirect(url_for("main.overview"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        u = User.query.filter(func.lower(User.username) == username.lower()).first()
        if u and u.check_password(request.form.get("password", "")):
            session.clear()
            session.permanent = True
            session["uid"], session["csrf"] = u.id, secrets.token_hex(16)
            u.last_login = now()
            db.session.commit()
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("main.overview"))
        error = "Incorrect username or password."
    return render_template("signin.html", error=error), (401 if error else 200)


@bp.post("/signout")
def signout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("main.landing"))


# ---------------------------------------------------------- authenticated ----
@bp.get("/overview")
@login_required
def overview():
    mine = Prediction.query.filter_by(user_id=g.user.id)
    total = mine.count()
    ad = mine.filter_by(predicted_class="AD").count()
    avg_ms = db.session.query(func.avg(Prediction.processing_ms)).filter_by(user_id=g.user.id).scalar()
    recent = mine.order_by(Prediction.created_at.desc()).limit(5).all()

    rows = (db.session.query(User.state, func.count(Prediction.id))
            .join(Prediction, Prediction.user_id == User.id).group_by(User.state).all())
    by_zone = {z: 0 for z in nigeria.ZONES}
    for state, n in rows:
        by_zone[nigeria.ZONE_OF.get(state, "North Central")] += n
    return render_template("overview.html", total=total, ad=ad, no_ad=total - ad, avg_ms=avg_ms, recent=recent,
                           by_zone=by_zone, zone_max=max(by_zone.values()) or 1,
                           national=sum(by_zone.values()), my_zone=nigeria.ZONE_OF.get(g.user.state))


@bp.route("/scan", methods=["GET", "POST"])
@login_required
def scan():
    if request.method == "GET":
        return render_template("scan.html", error=None)

    def fail(msg):
        return render_template("scan.html", error=msg), 400

    file = request.files.get("mri")
    if not file or not file.filename:
        return fail("Choose an MRI image to upload.")
    if not preprocess.allowed_file(file.filename):
        return fail("Unsupported file type. Upload a PNG, JPG or BMP image.")
    data = file.read()
    try:
        result = current_app.classifier.analyse(data)
    except preprocess.InvalidImage as e:
        return fail(str(e))

    h, w = preprocess.decode(data).shape[:2]
    stored = f"{uuid.uuid4().hex}.png"
    cv2.imwrite(str(Path(current_app.config["UPLOAD_FOLDER"]) / stored), result["gray"])
    image = MRIImage(user_id=g.user.id, original_name=Path(file.filename).name[:255], stored_name=stored,
                     scan_ref=(request.form.get("scan_ref", "").strip()[:60] or None),
                     width=w, height=h, size_bytes=len(data))
    image.prediction = Prediction(
        user_id=g.user.id, predicted_class=result["short"], probability=result["probability"],
        prob_ad=result["prob_ad"], features_json=json.dumps(result["features"]),
        processing_ms=result["timing_ms"]["total"], timing_json=json.dumps(result["timing_ms"]))
    db.session.add(image)
    db.session.commit()
    return redirect(url_for("main.report", pid=image.prediction.id))


@bp.get("/report/<int:pid>")
@login_required
def report(pid):
    p = own_prediction(pid)
    feats = json.loads(p.features_json)
    table = [{"name": n, **feat.describe(n), "value": feats[n],
              "importance": current_app.classifier.importance[n]} for n in feat.FEATURE_NAMES]
    return render_template("report.html", p=p, top=current_app.classifier.top_features(feats),
                           table=table, timing=json.loads(p.timing_json))


@bp.get("/records")
@login_required
def records():
    show, q = request.args.get("show", "all"), request.args.get("q", "").strip()
    page = max(request.args.get("page", 1, type=int), 1)
    query = Prediction.query.filter_by(user_id=g.user.id).join(MRIImage)
    if show in ("ad", "no-ad"):
        query = query.filter(Prediction.predicted_class == ("AD" if show == "ad" else "No AD"))
    if q:
        like = f"%{q}%"
        query = query.filter(db.or_(MRIImage.scan_ref.ilike(like), MRIImage.original_name.ilike(like)))
    total = query.count()
    rows = query.order_by(Prediction.created_at.desc()).offset((page - 1) * PER_PAGE).limit(PER_PAGE).all()
    return render_template("records.html", rows=rows, total=total, show=show, q=q, page=page,
                           pages=max((total - 1) // PER_PAGE + 1, 1))


@bp.post("/records/<int:pid>/delete")
@login_required
def delete(pid):
    p = own_prediction(pid)
    (Path(current_app.config["UPLOAD_FOLDER"]) / p.image.stored_name).unlink(missing_ok=True)
    db.session.delete(p.image)
    db.session.commit()
    flash("Record deleted.", "success")
    return redirect(url_for("main.records"))


@bp.get("/slice/<int:pid>.png")
@login_required
def scan_image(pid):
    p = own_prediction(pid)
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], p.image.stored_name, max_age=3600)


@bp.get("/records.csv")
@login_required
def records_csv():
    """Download the signed-in user's prediction history as a spreadsheet file."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["record_id", "scan_reference", "file_name", "predicted_class", "class_probability",
                "probability_ad", "processing_ms", "analysed_at"])
    for p in Prediction.query.filter_by(user_id=g.user.id).order_by(Prediction.created_at.desc()):
        w.writerow([p.id, p.image.scan_ref or "", p.image.original_name, p.predicted_class,
                    f"{p.probability:.4f}", f"{p.prob_ad:.4f}", f"{p.processing_ms:.1f}",
                    p.created_at.strftime("%Y-%m-%d %H:%M:%S")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=mindtrace-records.csv"})


@bp.get("/performance")
def performance():
    m = current_app.classifier.metrics
    pts = " ".join(f"{30 + x * 260:.1f},{270 - y * 260:.1f}" for x, y in zip(m["roc"]["fpr"], m["roc"]["tpr"]))
    return render_template("performance.html", roc_points=pts)


@bp.get("/health")
def health():
    return jsonify(status="ok", model="random-forest", features=len(feat.FEATURE_NAMES))


# ----------------------------------------------------------------- errors ----
@bp.app_errorhandler(400)
@bp.app_errorhandler(404)
@bp.app_errorhandler(413)
@bp.app_errorhandler(500)
def error(e):
    code = getattr(e, "code", 500)
    msg = {404: "That page does not exist, or the record belongs to another account.",
           413: "The file is larger than 8 MB.",
           500: "Something went wrong on the server."}.get(code, getattr(e, "description", "Bad request."))
    if code == 413 and g.get("user"):
        return render_template("scan.html", error=msg), 413
    return render_template("error.html", code=code, msg=msg), code

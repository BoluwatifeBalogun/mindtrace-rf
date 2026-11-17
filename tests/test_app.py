"""Automated versions of the write-up's test cases TC01 to TC12 (Table 3.5),
plus unit tests for the feature extractor. Run:  python -m pytest -q"""
import io
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from mindtrace import create_app
from mindtrace.ml import features, preprocess
from mindtrace.models import Prediction, User, db

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


@pytest.fixture()
def app(tmp_path):
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 't.db'}",
                      "UPLOAD_FOLDER": str(tmp_path / "up")})
    with app.app_context():
        db.create_all()
        u = User(username="tester", full_name="Test User", role="Student", facility="YabaTech", state="Lagos")
        u.set_password("Password1")
        db.session.add(u)
        db.session.commit()
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def csrf(client, path="/signin"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([0-9a-f]+)"', html).group(1)


def login(client, password="Password1"):
    return client.post("/signin", data={"username": "tester", "password": password, "csrf_token": csrf(client)})


def upload(client, name, data=None):
    data = data if data is not None else (SAMPLES / name).read_bytes()
    return client.post("/scan", data={"csrf_token": csrf(client, "/scan"), "scan_ref": "CASE-01",
                                         "mri": (io.BytesIO(data), name)}, content_type="multipart/form-data")


# ---- unit: preprocessing and features ---------------------------------------
def test_tc05_preprocess_resizes_and_normalises():
    img = cv2.imread(str(SAMPLES / "AD_mild_1.png"))
    big = cv2.resize(img, (300, 260))
    g = preprocess.preprocess(big)
    assert g.shape == (128, 128) and g.dtype == np.uint8 and g.min() == 0 and g.max() == 255


def test_feature_vector_shape_and_determinism():
    g = preprocess.preprocess(cv2.imread(str(SAMPLES / "AD_mild_1.png")))
    a, b = features.extract(g), features.extract(g)
    assert list(a) == features.FEATURE_NAMES and len(a) == 69 and a == b
    assert all(np.isfinite(v) for v in a.values())


def test_glcm_is_normalised_symmetric_and_matches_known_case():
    q = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 2, 2, 2], [2, 2, 3, 3]])   # Haralick's example image
    p = features.glcm(q, 0, 1, levels=4)
    assert np.isclose(p.sum(), 1) and np.allclose(p, p.T)
    assert np.allclose(p * 24, [[4, 2, 1, 0], [2, 4, 0, 0], [1, 0, 6, 1], [0, 0, 1, 2]])


def test_glcm_matches_scikit_image():
    sk = pytest.importorskip("skimage.feature")
    g = preprocess.preprocess(cv2.imread(str(SAMPLES / "NoAD_nondemented_1.png")))
    q = preprocess.quantise(g)
    ref = sk.graycomatrix(q.astype(np.uint8), [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
    assert np.allclose(ref[:, :, 0, 0], features.glcm(q, 0, 1))
    assert np.allclose(ref[:, :, 0, 1], features.glcm(q, -1, 0))
    assert np.isclose(sk.graycoprops(ref, "contrast")[0, 0], features.glcm_props(features.glcm(q, 0, 1))["contrast"])


# ---- TC01 / TC02 authentication ---------------------------------------------
def test_tc01_valid_login_reaches_dashboard(client):
    r = login(client)
    assert r.status_code == 302 and r.headers["Location"].endswith("/overview")
    assert b"Overview" in client.get("/overview").data


def test_tc02_invalid_login_shows_error(client):
    r = login(client, "wrong")
    assert r.status_code == 401 and b"Incorrect username or password" in r.data


def test_protected_pages_require_login(client):
    for path in ("/overview", "/scan", "/records", "/report/1", "/records.csv"):
        assert client.get(path).status_code == 302


def test_post_without_csrf_token_is_rejected(client):
    assert client.post("/signin", data={"username": "tester", "password": "Password1"}).status_code == 400


def test_registration_validates_and_creates_account(client, app):
    base = {"full_name": "Ada Obi", "username": "ada", "password": "Secret123", "confirm": "Secret123",
            "role": "Researcher", "state": "Enugu", "facility": "UNTH"}
    bad = client.post("/signup", data={**base, "state": "Atlantis", "csrf_token": csrf(client, "/signup")})
    assert bad.status_code == 400
    ok = client.post("/signup", data={**base, "csrf_token": csrf(client, "/signup")})
    assert ok.status_code == 302
    with app.app_context():
        assert User.query.filter_by(username="ada").one().password_hash != "Secret123"


# ---- TC03 to TC10: upload, analysis, result, storage ------------------------
def test_tc03_06_07_09_10_valid_upload_is_analysed_stored_and_shown(client, app):
    login(client)
    r = upload(client, "AD_mild_1.png")
    assert r.status_code == 302 and "/report/" in r.headers["Location"]
    with app.app_context():
        p = Prediction.query.one()
        assert p.predicted_class == "AD" and 0.5 <= p.probability <= 1 and p.image.scan_ref == "CASE-01"
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Alzheimer's Disease" in page and "Class probability" in page and "Feature importance" in page


def test_tc07_non_demented_sample_is_classified_no_ad(client, app):
    login(client)
    upload(client, "NoAD_nondemented_1.png")
    with app.app_context():
        assert Prediction.query.one().predicted_class == "No AD"


@pytest.mark.parametrize("name,data,msg", [
    ("INVALID_not_an_image.txt", None, b"Unsupported file type"),
    ("INVALID_colour_photo.jpg", None, b"colour photograph"),
    ("fake.png", b"not really a png", b"could not be read"),
    ("blank.png", cv2.imencode(".png", np.zeros((128, 128), np.uint8))[1].tobytes(), b"blank"),
    ("tiny.png", cv2.imencode(".png", np.full((20, 20), 90, np.uint8))[1].tobytes(), b"too small"),
], ids=["text-file", "colour-photo", "corrupt-png", "blank-image", "tiny-image"])
def test_tc04_invalid_uploads_are_rejected_with_message(client, app, name, data, msg):
    login(client)
    r = upload(client, name, data)
    assert r.status_code == 400 and msg in r.data
    with app.app_context():
        assert Prediction.query.count() == 0


def test_tc08_feature_importance_is_reported(app):
    top = app.classifier.top_features(app.classifier.analyse((SAMPLES / "AD_mild_1.png").read_bytes())["features"])
    assert len(top) == 8 and top[0]["importance"] >= top[-1]["importance"] > 0
    assert abs(sum(app.classifier.importance.values()) - 1) < 1e-6


# ---- TC11 / TC12 history and logout -----------------------------------------
def test_tc11_history_lists_filters_and_deletes(client, app):
    login(client)
    upload(client, "AD_mild_1.png")
    upload(client, "NoAD_nondemented_1.png")
    assert client.get("/records").data.count(b'class="rec ') == 2
    assert client.get("/records?show=ad").data.count(b'class="rec ') == 1
    assert client.get("/records?q=nondemented").data.count(b'class="rec ') == 1
    rows = client.get("/records.csv").get_data(as_text=True).strip().splitlines()
    assert len(rows) == 3 and rows[0].startswith("record_id,")
    client.post("/records/1/delete", data={"csrf_token": csrf(client, "/records")})
    with app.app_context():
        assert Prediction.query.count() == 1
    assert len(list(Path(app.config["UPLOAD_FOLDER"]).iterdir())) == 1


def test_records_are_private_to_their_owner(client, app):
    login(client)
    upload(client, "AD_mild_1.png")
    client.post("/signout", data={"csrf_token": csrf(client, "/records")})
    with app.app_context():
        u = User(username="other", full_name="Other User", role="Student", facility="X", state="Kano")
        u.set_password("Password1")
        db.session.add(u)
        db.session.commit()
    client.post("/signin", data={"username": "other", "password": "Password1", "csrf_token": csrf(client)})
    assert client.get("/report/1").status_code == 404 and client.get("/slice/1.png").status_code == 404


def test_tc12_logout_ends_session(client):
    login(client)
    r = client.post("/signout", data={"csrf_token": csrf(client, "/overview")})
    assert r.status_code == 302 and client.get("/overview").status_code == 302


# ---- performance and model validation ---------------------------------------
def test_performance_single_image_under_one_second(app):
    t = app.classifier.analyse((SAMPLES / "AD_verymild_1.png").read_bytes())["timing_ms"]
    assert t["total"] < 1000


def test_model_metrics_are_consistent(app):
    m = app.classifier.metrics
    c = m["test"]["confusion"]
    n = sum(c.values())
    assert n == m["dataset"]["split"]["test"]
    assert abs(m["test"]["accuracy"] - (c["tp"] + c["tn"]) / n) < 1e-9
    assert abs(m["test"]["precision"] - c["tp"] / (c["tp"] + c["fp"])) < 1e-9
    assert abs(m["test"]["recall"] - c["tp"] / (c["tp"] + c["fn"])) < 1e-9
    assert m["test"]["accuracy"] > 0.75

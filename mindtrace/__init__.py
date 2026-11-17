"""MindTrace NG application factory."""
import os
import secrets
from pathlib import Path

import click
from flask import Flask

from .models import User, db

ROOT = Path(__file__).resolve().parent.parent


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"),
                instance_path=str(ROOT / "instance"))
    Path(app.instance_path).mkdir(exist_ok=True)

    key_file = Path(app.instance_path) / "secret.key"
    if not os.environ.get("SECRET_KEY") and not key_file.exists():
        key_file.write_text(secrets.token_hex(32))
    db_url = os.environ.get("DATABASE_URL") or f"sqlite:///{Path(app.instance_path) / 'mindtrace.db'}"
    if db_url.startswith("mysql://"):
        db_url = db_url.replace("mysql://", "mysql+pymysql://", 1)

    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY") or key_file.read_text().strip(),
        SQLALCHEMY_DATABASE_URI=db_url,
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        UPLOAD_FOLDER=str(Path(app.instance_path) / "uploads"),
        MODEL_DIR=os.environ.get("MODEL_DIR", str(ROOT / "models")),
        MAX_CONTENT_LENGTH=9 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("COOKIE_SECURE") == "1",
        PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
    )
    if test_config:
        app.config.update(test_config)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    from .ml.classifier import Classifier
    app.classifier = Classifier(app.config["MODEL_DIR"])

    from .routes import bp
    app.register_blueprint(bp)

    @app.cli.command("seed-demo")
    def seed_demo():
        """Create a demonstration account (username: demo, password: Demo@1234)."""
        if User.query.filter_by(username="demo").first():
            click.echo("Demo user already exists.")
            return
        u = User(username="demo", full_name="Demo Radiographer", role="Radiographer",
                 facility="Federal Neuropsychiatric Hospital, Yaba", state="Lagos")
        u.set_password("Demo@1234")
        db.session.add(u)
        db.session.commit()
        click.echo("Created demo / Demo@1234")

    return app

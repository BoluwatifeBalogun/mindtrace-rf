"""Database tables (write-up 3.12.6 and the class diagram, Fig. 3.8)."""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(40), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(40), nullable=False)
    facility = db.Column(db.String(160), nullable=False)
    state = db.Column(db.String(40), nullable=False)
    created_at = db.Column(db.DateTime, default=now, nullable=False)
    last_login = db.Column(db.DateTime)
    images = db.relationship("MRIImage", backref="user", cascade="all, delete-orphan")

    def set_password(self, raw):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw):
        return check_password_hash(self.password_hash, raw)

    @property
    def first_name(self):
        return self.full_name.split()[0]

    @property
    def initials(self):
        return "".join(p[0] for p in self.full_name.split()[:2]).upper()


class MRIImage(db.Model):
    __tablename__ = "mri_images"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    original_name = db.Column(db.String(255), nullable=False)
    stored_name = db.Column(db.String(80), nullable=False, unique=True)
    scan_ref = db.Column(db.String(60))
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    size_bytes = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default="analysed", nullable=False)
    uploaded_at = db.Column(db.DateTime, default=now, nullable=False)
    prediction = db.relationship("Prediction", backref="image", uselist=False, cascade="all, delete-orphan")


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey("mri_images.id", ondelete="CASCADE"), nullable=False, unique=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    predicted_class = db.Column(db.String(10), nullable=False)       # "AD" or "No AD"
    probability = db.Column(db.Float, nullable=False)                # probability of the predicted class
    prob_ad = db.Column(db.Float, nullable=False)
    features_json = db.Column(db.Text, nullable=False)               # the 69 extracted feature values
    processing_ms = db.Column(db.Float, nullable=False)
    timing_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=now, nullable=False, index=True)
    user = db.relationship("User")

    @property
    def is_ad(self):
        return self.predicted_class == "AD"

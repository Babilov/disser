from sqlalchemy import Table, Column, Integer, String, ForeignKey, DateTime, JSON
from database import metadata
import datetime

Client = Table(
    "client",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("client_id", String, unique=True, nullable=False),
    Column("connected_at", DateTime, default=datetime.datetime.now),
)

ROI = Table(
    "roi",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("client_id", Integer, ForeignKey("client.id"), nullable=False),
    Column("roi_index", Integer, nullable=False),
    Column("cords", JSON, nullable=False),
)

ROIStat = Table(
    "roi_stat",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("roi_id", Integer, ForeignKey("roi.id"), nullable=False),
    Column("density", Integer, nullable=False),
    Column("intensity", Integer, nullable=False),
    Column("timestamp", DateTime, default=datetime.datetime.utcnow),
)

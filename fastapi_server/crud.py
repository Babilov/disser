from database import database
from models import Client, ROI, ROIStat
from sqlalchemy import select
from datetime import datetime

async def create_client(client_id: str):
    now = datetime.now()
    query = Client.insert().values(client_id=client_id)
    client_pk = await database.execute(query)
    return {"id": client_pk, "client_id": client_id, "connected_at": now}

async def get_client(client_pk: int):
    query = select(Client).where(Client.c.id == client_pk)
    client = await database.fetch_one(query)
    return client

async def create_roi(client_id: int, roi_index: int, cords: dict):
    query = ROI.insert().values(client_id=client_id, roi_index=roi_index, cords=cords)
    roi_pk = await database.execute(query)
    return {"id": roi_pk, "client_id": client_id, "roi_index": roi_index, "cords": cords}

async def get_roi(client_id: int, roi_index: int):
    query = ROI.select().where(
        (ROI.c.client_id == client_id) & (ROI.c.roi_index == roi_index)
    )
    return await database.fetch_one(query)

async def update_roi(roi_id: int, cords: dict):
    query = ROI.update().where(ROI.c.id == roi_id).values(cords=cords)
    await database.execute(query)

async def create_roi_stat(roi_id: int, density: int, intensity: int, timestamp=None):
    values = {
        "roi_id": roi_id,
        "density": density,
        "intensity": intensity,
        "timestamp": timestamp,
    }
    query = ROIStat.insert().values(**values)
    roi_stat_pk = await database.execute(query)
    return {"id": roi_stat_pk, **values}

async def get_roi_stat(roi_stat_pk: int):
    query = select(ROIStat).where(ROIStat.c.id == roi_stat_pk)
    roi_stat = await database.fetch_one(query)
    return roi_stat

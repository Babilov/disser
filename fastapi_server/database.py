from databases import Database
from sqlalchemy import create_engine, MetaData

DATABASE_URL = "postgresql+asyncpg://root:root@localhost:5432/mydatabase"

database = Database(DATABASE_URL)
metadata = MetaData()

# Создание обычного sync engine для миграций и создания схемы
engine = create_engine(DATABASE_URL.replace("+asyncpg", ""))

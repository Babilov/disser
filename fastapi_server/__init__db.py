from database import engine, metadata
import models  # это нужно, чтобы таблицы зарегистрировались в metadata

metadata.create_all(bind=engine)
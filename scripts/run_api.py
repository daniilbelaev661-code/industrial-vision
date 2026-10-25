"""Запуск FastAPI сервера."""
import uvicorn

if __name__ == "__main__":
    # host="0.0.0.0" разрешает доступ из локальной сети
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
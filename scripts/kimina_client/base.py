from typing import Union
import os
from typing import Any, Type, TypeVar
from typing import Union, Dict, Tuple, List

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseKimina:
    def __init__(
        self,
        api_url: Union[str, None] = None,
        api_key: Union[str, None] = None,
        headers: Union[Dict[str, str], None] = None,
        http_timeout: int = 60,
        n_retries: int = 3,
    ):
        if not api_url:
            api_url = os.getenv("LEAN_SERVER_API_URL", "http://localhost:8000")
        self.api_url = api_url.rstrip("/")

        if not api_key:
            api_key = os.getenv("LEAN_SERVER_API_KEY") or os.getenv(
                "LEANSERVER_API_KEY"
            )

        self.api_key = api_key
        self.headers = headers or {}
        if self.api_key:
            self.headers.setdefault("Authorization", f"Bearer {self.api_key}")
        self.http_timeout = http_timeout
        self.n_retries = n_retries

    def build_url(self, path: str) -> str:
        return f"{self.api_url}/{path.lstrip('/')}"

    def handle(self, resp: Any, model: Type[T]) -> T:
        return model.model_validate(resp)

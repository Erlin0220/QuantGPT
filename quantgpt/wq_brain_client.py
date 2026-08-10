"""WorldQuant BRAIN API client.

Wraps the WQ BRAIN REST API for alpha simulation, quality checks, and
formal submission. Credentials are read from environment variables
WQ_BRAIN_EMAIL and WQ_BRAIN_PASSWORD.
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

API_BASE = "https://api.worldquantbrain.com"

SUBMIT_THRESHOLDS = {
    "sharpe": 1.25,
    "fitness": 1.0,
    "turnover_max": 0.7,
    "turnover_min": 0.01,
}

_POLL_INTERVAL = max(1, int(os.environ.get("WQ_SIM_POLL_INTERVAL", "10")))
_SIM_POLL_MAX_WAIT = max(_POLL_INTERVAL, int(os.environ.get("WQ_SIM_POLL_MAX_WAIT", "1800")))
_POLL_MAX_ATTEMPTS = max(1, _SIM_POLL_MAX_WAIT // _POLL_INTERVAL)
_RECOVERY_LOOKUP_EVERY = max(1, int(os.environ.get("WQ_SIM_RECOVERY_LOOKUP_EVERY", "3")))
_CONCURRENT_BACKOFF = 30
_MAX_RETRIES = 5


def _env_timeout(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


HTTP_TIMEOUT = (
    _env_timeout("WQ_HTTP_CONNECT_TIMEOUT", 10.0),
    _env_timeout("WQ_HTTP_READ_TIMEOUT", 60.0),
)


_ACCOUNT_ENV = {
    "primary": ("WQ_BRAIN_EMAIL", "WQ_BRAIN_PASSWORD"),
    "alt": ("WQ_BRAIN_ALT_EMAIL", "WQ_BRAIN_ALT_PASSWORD"),
}


def is_configured(account: str | None = None) -> bool:
    if account:
        env_email, env_pwd = _ACCOUNT_ENV.get(account, _ACCOUNT_ENV["primary"])
        return bool(os.environ.get(env_email) and os.environ.get(env_pwd))
    return any(bool(os.environ.get(e) and os.environ.get(p)) for e, p in _ACCOUNT_ENV.values())


def configured_accounts() -> list[str]:
    return [name for name, (e, p) in _ACCOUNT_ENV.items() if os.environ.get(e) and os.environ.get(p)]


def get_client(account: str = "primary") -> "WQBrainClient":
    env_email, env_pwd = _ACCOUNT_ENV.get(account, _ACCOUNT_ENV["primary"])
    return WQBrainClient(
        email=os.environ.get(env_email, ""),
        password=os.environ.get(env_pwd, ""),
    )


class WQBrainClient:
    def __init__(self, email: str | None = None, password: str | None = None):
        self.email = email or os.environ.get("WQ_BRAIN_EMAIL", "")
        self.password = password or os.environ.get("WQ_BRAIN_PASSWORD", "")
        self._session: requests.Session | None = None
        self._operators_cache: list[dict] | None = None
        self._data_fields_cache: dict[tuple, list[dict]] = {}
        self._datasets_cache: dict[tuple, list[dict]] = {}
        self._alpha_pnl_cache: dict[str, dict] = {}

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.trust_env = False
            retry = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        return self._session

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    def fork_authenticated(self) -> "WQBrainClient":
        """Create an independent worker client reusing this client's auth state.

        ``requests.Session`` is not shared between simulation workers.  Cookies,
        headers and the operator catalog are copied so parallel workers do not
        need to authenticate or refetch operators for every Alpha.
        """
        child = WQBrainClient(email=self.email, password=self.password)
        if self._session is not None:
            child_session = child._get_session()
            child_session.headers.update(self._session.headers)
            child_session.cookies.update(self._session.cookies)
        if self._operators_cache is not None:
            child._operators_cache = list(self._operators_cache)
        child._data_fields_cache = {key: list(value) for key, value in self._data_fields_cache.items()}
        child._datasets_cache = {key: list(value) for key, value in self._datasets_cache.items()}
        child._alpha_pnl_cache = {key: dict(value) for key, value in self._alpha_pnl_cache.items()}
        return child

    def authenticate(self, _max_retries: int = 5) -> bool:
        s = self._get_session()
        for attempt in range(_max_retries):
            r = s.post(
                f"{API_BASE}/authentication",
                auth=(self.email, self.password),
                timeout=HTTP_TIMEOUT,
            )
            if r.status_code == 429:
                retry = int(r.headers.get("Retry-After", "60"))
                logger.info(f"WQ auth rate-limited, waiting {retry}s (attempt {attempt + 1}/{_max_retries})")
                time.sleep(retry + 1)
                continue

            if r.status_code not in (200, 201):
                logger.error(f"WQ auth failed: HTTP {r.status_code}")
                return False

            data = r.json()
            if "inquiry" in data:
                logger.error("WQ auth requires biometric verification — log in via browser first")
                return False

            logger.info("WQ BRAIN authenticated")
            return True

        logger.error(f"WQ auth failed: rate-limited {_max_retries} times")
        return False

    def get_user_info(self) -> dict:
        r = self._get_session().get(f"{API_BASE}/users/self", timeout=HTTP_TIMEOUT)
        return r.json() if r.status_code == 200 else {}

    def get_user_competitions(self, user_id: str) -> dict:
        """Return competition/Challenge progress for the authenticated user."""
        r = self._get_session().get(f"{API_BASE}/users/{user_id}/competitions", timeout=HTTP_TIMEOUT)
        return r.json() if r.status_code == 200 else {}

    def get_user_alpha_summary(self) -> dict:
        """Return platform Alpha counts grouped by submission status."""
        r = self._get_session().get(f"{API_BASE}/users/self/alphas/summary", timeout=HTTP_TIMEOUT)
        return r.json() if r.status_code == 200 else {}

    def list_operators(self, refresh: bool = False) -> list[dict]:
        """Return the operator catalog currently exposed to this BRAIN account."""
        if self._operators_cache is not None and not refresh:
            return self._operators_cache
        r = self._get_session().get(f"{API_BASE}/operators", timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch WQ operators: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        if isinstance(data, list):
            operators = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            operators = [item for item in data.get("results", []) if isinstance(item, dict)]
        else:
            operators = []
        self._operators_cache = operators
        return operators

    def list_operator_names(self) -> set[str]:
        return {str(item.get("name", "")).strip().lower() for item in self.list_operators() if item.get("name")}

    @staticmethod
    def _catalog_results(data) -> list[dict]:
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            return [item for item in data.get("results", []) if isinstance(item, dict)]
        return []

    def list_data_fields(
        self,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
        dataset_id: str | None = None,
        search: str | None = None,
        limit: int = 200,
        refresh: bool = False,
    ) -> list[dict]:
        """Return BRAIN Data Explorer fields available to the authenticated account."""
        limit = max(1, min(int(limit), 1000))
        key = (instrument_type, region, universe, int(delay), dataset_id or "", search or "", limit)
        if key in self._data_fields_cache and not refresh:
            return list(self._data_fields_cache[key])

        fields: list[dict] = []
        offset = 0
        page_size = min(50, limit)
        while len(fields) < limit:
            params = {
                "instrumentType": instrument_type,
                "region": region,
                "delay": int(delay),
                "universe": universe,
                "limit": min(page_size, limit - len(fields)),
                "offset": offset,
            }
            if dataset_id:
                params["dataset.id"] = dataset_id
            if search:
                params["search"] = search
            r = self._get_session().get(f"{API_BASE}/data-fields", params=params, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"Failed to fetch WQ data fields: HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            page = self._catalog_results(data)
            fields.extend(page)
            total = int(data.get("count", len(fields))) if isinstance(data, dict) else len(fields)
            if not page or len(fields) >= total:
                break
            offset += len(page)

        result = fields[:limit]
        self._data_fields_cache[key] = list(result)
        return result

    def list_datasets(
        self,
        *,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        instrument_type: str = "EQUITY",
        limit: int = 100,
        refresh: bool = False,
    ) -> list[dict]:
        """Return BRAIN data sets visible to the account for the requested scope."""
        limit = max(1, min(int(limit), 500))
        key = (instrument_type, region, universe, int(delay), limit)
        if key in self._datasets_cache and not refresh:
            return list(self._datasets_cache[key])

        datasets: list[dict] = []
        offset = 0
        # BRAIN rejects oversized catalog pages on some accounts.  Keep the
        # request size aligned with the data-fields endpoint and paginate.
        page_size = min(50, limit)
        while len(datasets) < limit:
            params = {
                "instrumentType": instrument_type,
                "region": region,
                "delay": int(delay),
                "universe": universe,
                "limit": min(page_size, limit - len(datasets)),
                "offset": offset,
            }
            r = self._get_session().get(f"{API_BASE}/data-sets", params=params, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"Failed to fetch WQ datasets: HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            page = self._catalog_results(data)
            datasets.extend(page)
            total = int(data.get("count", len(datasets))) if isinstance(data, dict) else len(datasets)
            if not page or len(datasets) >= total:
                break
            offset += len(page)

        result = datasets[:limit]
        self._datasets_cache[key] = list(result)
        return result

    def simulate(
        self,
        expression: str,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        decay: int = 0,
        neutralization: str = "SUBINDUSTRY",
        truncation: float = 0.08,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> dict:
        """Start a BRAIN simulation and poll it in the worker, not in the MCP request.

        Polling can run for up to ``WQ_SIM_POLL_MAX_WAIT`` (30 minutes by
        default).  While polling, the client periodically reconciles against
        ``/users/self/alphas``.  This handles the platform behaviour where an
        Alpha is already created even though the simulation endpoint never
        transitions to DONE for this client session.
        """
        s = self._get_session()
        payload = {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region": region,
                "universe": universe,
                "delay": delay,
                "decay": decay,
                "neutralization": neutralization,
                "truncation": truncation,
                "pasteurization": "ON",
                "unitHandling": "VERIFY",
                "nanHandling": "OFF",
                "language": "FASTEXPR",
                "visualization": False,
            },
            "regular": expression,
        }
        baseline_alpha_ids = {
            str(alpha.get("id"))
            for alpha in self._list_matching_alphas(expression, payload["settings"])
            if alpha.get("id")
        }
        simulation_started_at = time.time()

        def cancelled() -> bool:
            try:
                return bool(cancel_callback and cancel_callback())
            except Exception:
                logger.exception("WQ cancel callback failed")
                return False

        def wait(seconds: int) -> bool:
            deadline = time.monotonic() + max(0, seconds)
            while time.monotonic() < deadline:
                if cancelled():
                    return False
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            return not cancelled()

        def recover_from_alpha_list(simulation_id: str) -> dict | None:
            alpha = self._find_recent_matching_alpha(
                expression,
                payload["settings"],
                created_after=simulation_started_at - 60,
                exclude_ids=baseline_alpha_ids,
            )
            if not alpha:
                return None
            alpha_id = alpha.get("id")
            if not alpha_id:
                return None
            logger.info(
                "Recovered WQ simulation from alpha list: simulation=%s alpha=%s",
                simulation_id,
                alpha_id,
            )
            if progress_callback:
                progress_callback(100, "平台已生成 Alpha，已从 Alpha 列表恢复结果")
            return {
                "ok": True,
                "expression": expression,
                "is": alpha.get("is", {}),
                "oos": alpha.get("oos", {}),
                "settings": alpha.get("settings", payload["settings"]),
                "alpha_id": alpha_id,
                "simulation_id": simulation_id,
                "recovered_from_platform": True,
            }

        for attempt in range(_MAX_RETRIES):
            if cancelled():
                return {"ok": False, "cancelled": True, "error": "WQ simulation cancelled"}
            try:
                r = s.post(f"{API_BASE}/simulations", json=payload, timeout=HTTP_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout) as e:
                wait_seconds = _CONCURRENT_BACKOFF * (attempt + 1)
                logger.warning(
                    f"WQ connection error (attempt {attempt + 1}/{_MAX_RETRIES}): {e}, retrying in {wait_seconds}s"
                )
                if progress_callback:
                    progress_callback(0, f"连接异常，等待 {wait_seconds}s（第 {attempt + 1} 次重试）")
                if not wait(wait_seconds):
                    return {"ok": False, "cancelled": True, "error": "WQ simulation cancelled"}
                continue

            if r.status_code in (200, 201, 202):
                break

            if r.status_code == 429:
                detail = ""
                try:
                    detail = r.json().get("detail", "")
                except Exception:
                    pass

                if "CONCURRENT_SIMULATION_LIMIT" in detail:
                    wait_seconds = _CONCURRENT_BACKOFF * (attempt + 1)
                    logger.info(f"WQ concurrent limit, waiting {wait_seconds}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                    if progress_callback:
                        progress_callback(0, f"并发限制，等待 {wait_seconds}s（第 {attempt + 1} 次重试）")
                    if not wait(wait_seconds):
                        return {"ok": False, "cancelled": True, "error": "WQ simulation cancelled"}
                    continue

                retry = int(r.headers.get("Retry-After", "60"))
                logger.info(f"WQ rate-limited, waiting {retry}s")
                if progress_callback:
                    progress_callback(0, f"速率限制，等待 {retry}s")
                if not wait(retry + 1):
                    return {"ok": False, "cancelled": True, "error": "WQ simulation cancelled"}
                continue

            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
        else:
            return {"ok": False, "error": "WQ concurrent retry limit exceeded"}

        location = r.headers.get("Location", "")
        if not location:
            return {"ok": False, "error": "No Location header in response"}

        url = location if location.startswith("http") else f"{API_BASE}{location}"
        simulation_id = url.rstrip("/").split("/")[-1]

        for i in range(_POLL_MAX_ATTEMPTS):
            if cancelled():
                return {
                    "ok": False,
                    "cancelled": True,
                    "error": "WQ simulation cancelled",
                    "simulation_id": simulation_id,
                }

            data: dict = {}
            try:
                poll_response = s.get(url, timeout=HTTP_TIMEOUT)
                if poll_response.status_code == 200:
                    try:
                        data = poll_response.json()
                    except Exception:
                        data = {}
            except (requests.ConnectionError, requests.Timeout):
                logger.warning(f"WQ poll connection error (attempt {i + 1}), retrying...")

            status = str(data.get("status", "")).upper()
            progress = data.get("progress", 0)

            if progress_callback and data:
                try:
                    pct = int(progress * 100) if isinstance(progress, float) and progress <= 1 else int(progress)
                except (TypeError, ValueError):
                    pct = 0
                progress_callback(min(max(pct, 0), 99), f"模拟进行中 ({pct}%)")

            if status in ("DONE", "COMPLETE"):
                alpha_raw = data.get("alpha", "")
                alpha_id = alpha_raw.split("/")[-1] if alpha_raw else None
                is_data = data.get("is", {})
                oos_data = data.get("oos", {})

                if alpha_id and not is_data:
                    alpha_detail = self._fetch_alpha(alpha_id)
                    is_data = alpha_detail.get("is", {})
                    oos_data = alpha_detail.get("oos", {})

                if progress_callback:
                    progress_callback(100, "模拟完成")

                return {
                    "ok": True,
                    "expression": expression,
                    "is": is_data,
                    "oos": oos_data,
                    "settings": data.get("settings", payload["settings"]),
                    "alpha_id": alpha_id,
                    "simulation_id": data.get("id") or simulation_id,
                }
            if status in ("ERROR", "FAILED"):
                return {
                    "ok": False,
                    "error": f"WQ simulation failed: {data.get('message', status)}",
                    "simulation_id": data.get("id") or simulation_id,
                }

            if (i + 1) % _RECOVERY_LOOKUP_EVERY == 0:
                recovered = recover_from_alpha_list(simulation_id)
                if recovered:
                    return recovered

            if not wait(_POLL_INTERVAL):
                return {
                    "ok": False,
                    "cancelled": True,
                    "error": "WQ simulation cancelled",
                    "simulation_id": simulation_id,
                }

        recovered = recover_from_alpha_list(simulation_id)
        if recovered:
            return recovered
        return {
            "ok": False,
            "error": f"WQ simulation polling timeout ({_SIM_POLL_MAX_WAIT}s)",
            "simulation_id": simulation_id,
        }

    def _fetch_alpha(self, alpha_id: str) -> dict:
        r = self._get_session().get(f"{API_BASE}/alphas/{alpha_id}", timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                logger.warning(f"Empty/invalid JSON from /alphas/{alpha_id}")
                return {}
        return {}

    def _list_matching_alphas(
        self,
        expression: str,
        settings: dict,
        *,
        created_after: float = 0,
        limit: int = 100,
    ) -> list[dict]:
        """List recent platform Alphas matching the exact expression/settings."""
        try:
            r = self._get_session().get(
                f"{API_BASE}/users/self/alphas",
                params={"limit": min(limit, 100), "offset": 0, "order": "-dateCreated"},
                timeout=HTTP_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout):
            return []
        if r.status_code != 200:
            return []
        try:
            data = r.json()
        except Exception:
            return []

        raw_alphas = data if isinstance(data, list) else data.get("results", [])
        target_expression = re.sub(r"\s+", "", expression).lower()
        keys = ("region", "universe", "delay", "decay", "neutralization", "truncation")

        def settings_match(candidate: dict) -> bool:
            for key in keys:
                expected = settings.get(key)
                actual = candidate.get(key)
                if key == "truncation":
                    if actual is None or expected is None:
                        if actual != expected:
                            return False
                        continue
                    try:
                        if abs(float(actual) - float(expected)) > 1e-9:
                            return False
                    except (TypeError, ValueError):
                        if actual != expected:
                            return False
                elif str(actual).upper() != str(expected).upper():
                    return False
            return True

        matches: list[dict] = []
        for alpha in raw_alphas:
            created = alpha.get("dateCreated")
            if created and created_after:
                try:
                    created_ts = datetime.fromisoformat(str(created).replace("Z", "+00:00")).timestamp()
                    if created_ts < created_after:
                        continue
                except (TypeError, ValueError):
                    continue

            regular = alpha.get("regular", {})
            code = regular.get("code", "") if isinstance(regular, dict) else str(regular)
            if re.sub(r"\s+", "", code).lower() != target_expression:
                continue
            if not settings_match(alpha.get("settings", {})):
                continue
            matches.append(alpha)
        return matches

    def _find_recent_matching_alpha(
        self,
        expression: str,
        settings: dict,
        *,
        created_after: float,
        exclude_ids: set[str] | None = None,
        limit: int = 100,
    ) -> dict:
        """Find a newly created Alpha when the simulation endpoint lags.

        ``exclude_ids`` is the pre-submit baseline, preventing a prior Alpha
        with the same expression/settings from being mistaken for this run.
        """
        excluded = exclude_ids or set()
        for alpha in self._list_matching_alphas(
            expression,
            settings,
            created_after=created_after,
            limit=limit,
        ):
            if str(alpha.get("id")) not in excluded:
                return alpha
        return {}

    def check_alpha_status(self, alpha_id: str) -> dict:
        """Fetch actual platform-side alpha status including submission state."""
        data = self._fetch_alpha(alpha_id)
        if not data:
            return {"ok": False, "error": f"Alpha {alpha_id} not found"}
        return {
            "ok": True,
            "alpha_id": alpha_id,
            "status": data.get("status"),
            "dateSubmitted": data.get("dateSubmitted"),
            "dateCreated": data.get("dateCreated"),
            "grade": data.get("grade"),
            "color": data.get("color"),
            "hidden": data.get("hidden"),
            "is": data.get("is", {}),
            "checks": data.get("checks", {}),
        }

    def fetch_alpha_pnl(self, alpha_id: str, *, refresh: bool = False) -> dict:
        """Fetch cumulative Alpha PnL for a local, non-official diversity proxy."""
        key = str(alpha_id or "").strip()
        if not key:
            return {}
        if key in self._alpha_pnl_cache and not refresh:
            return dict(self._alpha_pnl_cache[key])
        try:
            response = self._get_session().get(
                f"{API_BASE}/alphas/{key}/recordsets/pnl",
                timeout=HTTP_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout):
            return {}
        if response.status_code != 200:
            return {}
        try:
            payload = response.json()
        except Exception:
            return {}
        if isinstance(payload, dict):
            self._alpha_pnl_cache[key] = dict(payload)
            return dict(payload)
        if isinstance(payload, list):
            wrapped = {"records": payload}
            self._alpha_pnl_cache[key] = wrapped
            return dict(wrapped)
        return {}

    def submit_alpha(self, alpha_id: str) -> dict:
        s = self._get_session()

        for submit_try in range(3):
            try:
                r = s.post(
                    f"{API_BASE}/alphas/{alpha_id}/submit",
                    timeout=HTTP_TIMEOUT,
                )
                body = r.text[:500]
                logger.info(f"Submit {alpha_id}: HTTP {r.status_code}, body={body}")
            except (requests.ConnectionError, requests.Timeout) as e:
                logger.warning(f"Submit {alpha_id}: connection outcome unknown: {e}")
                # A POST timeout does not prove the platform rejected the request.
                # Never retry the formal POST blindly or release the local slot;
                # reconciliation must query BRAIN before any later submission.
                return {
                    "status_code": 0,
                    "ok": False,
                    "detail": f"submit request outcome unknown after connection error: {e}",
                    "platform_status": "UNKNOWN",
                    "submission_uncertain": True,
                }

            if r.status_code == 403:
                try:
                    resp = r.json()
                    checks = resp.get("is", {}).get("checks", [])
                    sc = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), None)
                    if sc and sc.get("result") == "FAIL":
                        logger.warning(f"Submit {alpha_id}: SC FAIL value={sc.get('value')} limit={sc.get('limit')}")
                        return {
                            "status_code": 403,
                            "ok": False,
                            "detail": f"SC FAIL: value={sc.get('value')} > limit={sc.get('limit')}",
                            "sc_value": sc.get("value"),
                            "sc_limit": sc.get("limit"),
                            "checks": checks,
                        }
                except Exception:
                    pass
                return {"status_code": 403, "ok": False, "detail": body}

            if r.status_code == 429:
                wait = 30 * (submit_try + 1)
                logger.warning(f"Submit {alpha_id}: rate limited (429), waiting {wait}s before retry")
                time.sleep(wait)
                continue

            if r.status_code not in (200, 201, 202):
                logger.warning(f"Submit {alpha_id}: unexpected HTTP {r.status_code}, waiting 15s before retry")
                time.sleep(15)
                continue

            poll_result = self._poll_alpha_submission(alpha_id)

            if poll_result.get("ok"):
                return poll_result

            if poll_result.get("platform_status") == "TIMEOUT":
                try:
                    alpha_data = self._fetch_alpha(alpha_id)
                except (requests.ConnectionError, requests.Timeout):
                    alpha_data = {}
                actual_status = (alpha_data.get("status") or "").upper()
                if actual_status == "ACTIVE":
                    return {
                        "status_code": 200,
                        "ok": True,
                        "detail": "poll timeout but platform now confirms ACTIVE",
                        "platform_status": "ACTIVE",
                    }
                if actual_status == "UNSUBMITTED":
                    logger.warning(
                        f"Submit {alpha_id}: platform confirms UNSUBMITTED after poll, retrying submit (try {submit_try + 1})"
                    )
                    time.sleep(10)
                    continue
                logger.warning(
                    f"Submit {alpha_id}: poll timeout with unresolved platform status={actual_status or 'UNKNOWN'}"
                )
                return {
                    **poll_result,
                    "ok": False,
                    "detail": f"submission outcome unresolved after poll timeout (status={actual_status or 'UNKNOWN'})",
                    "platform_status": actual_status or "UNKNOWN",
                    "submission_uncertain": True,
                }

            return poll_result

        try:
            alpha_data = self._fetch_alpha(alpha_id)
        except (requests.ConnectionError, requests.Timeout):
            alpha_data = {}
        actual_status = (alpha_data.get("status") or "").upper()
        if actual_status == "UNSUBMITTED":
            return {
                "status_code": 200,
                "ok": False,
                "detail": "platform confirms alpha remains UNSUBMITTED after submit retries",
                "platform_status": "UNSUBMITTED",
                "confirmed_not_submitted": True,
            }
        if actual_status == "ACTIVE":
            return {
                "status_code": 200,
                "ok": True,
                "detail": "platform confirms ACTIVE after submit retries",
                "platform_status": "ACTIVE",
            }
        return {
            "status_code": 0,
            "ok": False,
            "detail": f"submission outcome unresolved after retries (status={actual_status or 'UNKNOWN'})",
            "platform_status": actual_status or "UNKNOWN",
            "submission_uncertain": True,
        }

    def _poll_alpha_submission(self, alpha_id: str, max_polls: int = 12, interval: int = 10) -> dict:
        """Poll alpha status until platform confirms submission or SC check completes."""
        s = self._get_session()
        status = "UNKNOWN"
        sc_result = "MISSING"
        for i in range(max_polls):
            time.sleep(interval)
            try:
                r = s.get(f"{API_BASE}/alphas/{alpha_id}", timeout=HTTP_TIMEOUT)
            except (requests.ConnectionError, requests.Timeout):
                logger.warning(f"Submit poll {alpha_id}: connection error at poll #{i}")
                continue
            if r.status_code != 200:
                continue
            try:
                data = r.json()
            except Exception:
                continue

            status = data.get("status", "").upper()
            is_data = data.get("is", {})
            checks = is_data.get("checks", [])

            sc_check = next((c for c in checks if c.get("name") == "SELF_CORRELATION"), None)
            sc_result = sc_check.get("result", "PENDING") if sc_check else "MISSING"

            logger.info(f"Submit poll {alpha_id} #{i}: status={status}, SC={sc_result}")

            if status == "ACTIVE":
                logger.info(f"Submit {alpha_id}: confirmed ACTIVE on platform")
                return {
                    "status_code": 200,
                    "ok": True,
                    "detail": f"submitted and ACTIVE, SC={sc_result}",
                    "platform_status": status,
                }
            elif sc_result == "FAIL":
                assert sc_check is not None
                sc_value = sc_check.get("value", "?")
                sc_limit = sc_check.get("limit", "?")
                logger.warning(f"Submit {alpha_id}: SC FAIL (value={sc_value}, limit={sc_limit})")
                return {
                    "status_code": 200,
                    "ok": False,
                    "detail": f"SC FAIL: value={sc_value} > limit={sc_limit}",
                    "platform_status": status,
                    "sc_value": sc_value,
                    "sc_limit": sc_limit,
                }
            elif sc_result == "PASS" and status == "UNSUBMITTED":
                logger.info(f"Submit {alpha_id}: SC PASS but still UNSUBMITTED, retrying submit...")
                try:
                    s.post(
                        f"{API_BASE}/alphas/{alpha_id}/submit",
                        timeout=HTTP_TIMEOUT,
                    )
                except Exception:
                    pass

        return {
            "status_code": 200,
            "ok": False,
            "detail": f"submission polling timeout ({max_polls * interval}s), last status={status}, SC={sc_result}",
            "platform_status": "TIMEOUT",
        }

    def delete_alpha(self, alpha_id: str) -> dict:
        """Delete/retire an alpha from the platform."""
        s = self._get_session()
        r = s.delete(f"{API_BASE}/alphas/{alpha_id}", timeout=HTTP_TIMEOUT)
        if r.status_code in (200, 204):
            return {"ok": True, "detail": f"Alpha {alpha_id} deleted"}
        if r.status_code == 405:
            r2 = s.patch(
                f"{API_BASE}/alphas/{alpha_id}",
                json={"hidden": True},
                timeout=HTTP_TIMEOUT,
            )
            if r2.status_code in (200, 204):
                return {"ok": True, "detail": f"Alpha {alpha_id} hidden via PATCH"}
            return {"ok": False, "detail": f"DELETE 405, PATCH also failed: {r2.status_code} {r2.text[:200]}"}
        return {"ok": False, "detail": f"DELETE failed: {r.status_code} {r.text[:200]}"}

    def unhide_alpha(self, alpha_id: str) -> dict:
        """Restore a hidden alpha."""
        s = self._get_session()
        r = s.patch(
            f"{API_BASE}/alphas/{alpha_id}",
            json={"hidden": False},
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code in (200, 204):
            return {"ok": True, "detail": f"Alpha {alpha_id} restored"}
        return {"ok": False, "detail": f"Unhide failed: {r.status_code} {r.text[:200]}"}

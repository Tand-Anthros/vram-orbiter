#!/usr/bin/env python3
"""
vram_arbiter.py — следит, чтобы видеопамять занимал только один бэкенд:
ComfyUI или LM Studio.

Правило: если модель загружена в обоих, остаётся тот, чья задача началась
позже, а второй выгружается (ComfyUI через POST /free, LM Studio через
POST /api/v1/models/unload).

Как определяется "свежесть":
  * ComfyUI   — время старта последней задачи (очередь и история).
  * LM Studio — момент, когда скрипт увидел, что модель загрузилась.
    У LM Studio нет API "время последнего запроса", но при JIT-загрузке
    модель грузится именно по запросу, так что это и есть начало задачи.

Зависимостей нет, только стандартная библиотека. Python 3.9+.

Запуск:
    python vram_arbiter.py              # работать постоянно
    python vram_arbiter.py --once       # одна проверка и выход
    python vram_arbiter.py --dry-run    # ничего не выгружать, только писать
    python vram_arbiter.py --status     # показать состояние и выйти

Токен LM Studio (если включена авторизация) берётся из переменной
окружения LM_API_TOKEN или из --lms-token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

GB = 1024 ** 3


# ----------------------------------------------------------------- HTTP

def http_json(method: str, url: str, body: Optional[dict] = None,
              token: Optional[str] = None, timeout: float = 5.0) -> Any:
    """Запрос с JSON-телом. Возвращает разобранный JSON или None, если тело пустое."""
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with _OPENER.open(req, timeout=timeout) as resp:
        raw = resp.read()
    if not raw.strip():
        return None
    return json.loads(raw.decode("utf-8"))


# Бэкенды локальные, поэтому системный прокси намеренно не используется.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

NET_ERRORS = (urllib.error.URLError, OSError, ValueError)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------ состояние

@dataclass
class State:
    reachable: bool = False
    loaded: bool = False
    busy: bool = False                 # прямо сейчас выполняет задачу
    activity: Optional[float] = None   # время старта самой свежей задачи (epoch, сек)
    detail: str = ""


# -------------------------------------------------------------- ComfyUI

class Comfy:
    name = "ComfyUI"

    def __init__(self, url: str, threshold_gb: float, grace: float = 15.0):
        self.url = url.rstrip("/")
        self.threshold = threshold_gb * GB
        self.grace = grace              # сколько секунд после /free не верить счётчику VRAM
        self.started_at = time.time()
        self.freed_at = 0.0
        self.dirty = False              # после последнего /free выполнялась задача
        self.activity: Optional[float] = None
        self._running_seen: dict[str, float] = {}
        self._last_hist_id: Optional[str] = None
        self._first_poll = True

    # --- опрос

    def poll(self, now: float) -> State:
        try:
            queue = http_json("GET", f"{self.url}/queue") or {}
            history = http_json("GET", f"{self.url}/history?max_items=1") or {}
            stats = http_json("GET", f"{self.url}/system_stats") or {}
        except NET_ERRORS:
            self._first_poll = False
            return State(reachable=False, detail="недоступен")

        # 1) что выполняется сейчас
        running_ids = []
        for item in queue.get("queue_running") or []:
            # элемент очереди: [номер, prompt_id, граф, extra, outputs]
            if isinstance(item, (list, tuple)) and len(item) > 1:
                running_ids.append(str(item[1]))
        for pid in running_ids:
            if pid not in self._running_seen:
                # на первом опросе задача могла идти давно, но точнее "сейчас" мы не знаем
                self._running_seen[pid] = now
                self._bump(now)
                self.dirty = True
        busy = bool(running_ids)

        # 2) последняя завершённая задача
        for pid, entry in (history.items() if isinstance(history, dict) else []):
            if pid == self._last_hist_id:
                continue
            self._last_hist_id = pid
            started = self._start_time(entry)
            if started is None:
                # в истории нет меток времени: берём момент, когда сами увидели задачу
                started = self._running_seen.get(pid)
                if started is None and not self._first_poll:
                    started = now
            if started is not None:
                self._bump(started)
                if started > max(self.freed_at, self.started_at):
                    self.dirty = True

        for pid in list(self._running_seen):
            if pid not in running_ids:
                del self._running_seen[pid]

        # 3) счётчик видеопамяти PyTorch
        torch_used = 0.0
        for dev in stats.get("devices") or []:
            v = float(dev.get("torch_vram_total") or 0)
            if 0 < v < 10_000_000:      # на случай сборок, где значение в мегабайтах
                v *= 1024 ** 2
            torch_used = max(torch_used, v)
        vram_says_loaded = torch_used > self.threshold
        if now - self.freed_at < self.grace:
            vram_says_loaded = False    # после /free счётчик опускается не мгновенно

        self._first_poll = False
        loaded = self.dirty or vram_says_loaded or busy
        return State(True, loaded, busy, self.activity,
                     f"torch VRAM {torch_used / GB:.1f} ГБ" + (", идёт задача" if busy else ""))

    def _bump(self, t: float) -> None:
        if self.activity is None or t > self.activity:
            self.activity = t

    @staticmethod
    def _start_time(entry: Any) -> Optional[float]:
        """Время execution_start из записи истории (в секундах) или None."""
        try:
            messages = entry["status"]["messages"]
        except (KeyError, TypeError):
            return None
        first = None
        for m in messages or []:
            if isinstance(m, (list, tuple)) and len(m) > 1 and isinstance(m[1], dict):
                ts = m[1].get("timestamp")
                if isinstance(ts, (int, float)):
                    sec = ts / 1000.0 if ts > 1e11 else float(ts)
                    if m[0] == "execution_start":
                        return sec
                    first = sec if first is None else min(first, sec)
        return first

    # --- выгрузка

    def free(self, now: float) -> None:
        http_json("POST", f"{self.url}/free",
                  {"unload_models": True, "free_memory": True})
        self.freed_at = now
        self.dirty = False


# ------------------------------------------------------------ LM Studio

class LMStudio:
    name = "LM Studio"

    def __init__(self, url: str, token: Optional[str]):
        self.url = url.rstrip("/")
        self.token = token
        self.activity: Optional[float] = None
        self.instances: list[str] = []
        self._first_poll = True

    def poll(self, now: float) -> State:
        try:
            data = http_json("GET", f"{self.url}/api/v1/models", token=self.token) or {}
        except NET_ERRORS:
            self.instances = []
            self._first_poll = False
            return State(reachable=False, detail="недоступен")

        current = []
        for model in data.get("models") or []:
            for inst in model.get("loaded_instances") or []:
                if isinstance(inst, dict) and inst.get("id"):
                    current.append(str(inst["id"]))

        appeared = set(current) - set(self.instances)
        if appeared and not self._first_poll:
            # модель загрузилась на наших глазах: началась новая задача
            self.activity = now
        if not current:
            self.activity = None
        self.instances = current
        self._first_poll = False

        detail = ", ".join(current) if current else "моделей нет"
        return State(True, bool(current), False, self.activity, detail)

    def free(self, now: float) -> None:
        for inst in list(self.instances):
            http_json("POST", f"{self.url}/api/v1/models/unload",
                      {"instance_id": inst}, token=self.token, timeout=30)
        self.instances = []
        self.activity = None


# --------------------------------------------------------------- решение

def decide(comfy: State, lms: State, prefer: str) -> tuple[Optional[str], str]:
    """
    Кого выгружать: "comfy", "lmstudio" или None.
    Возвращает (решение, пояснение).
    """
    if not (comfy.loaded and lms.loaded):
        return None, ""

    if comfy.activity is not None and lms.activity is not None \
            and comfy.activity != lms.activity:
        loser = "comfy" if comfy.activity < lms.activity else "lmstudio"
        why = "его задача старше"
    else:
        # свежесть неизвестна (обычно сразу после запуска скрипта)
        if comfy.busy:
            loser, why = "lmstudio", "ComfyUI сейчас выполняет задачу"
        else:
            loser = "comfy" if prefer == "lmstudio" else "lmstudio"
            why = f"свежесть неизвестна, по умолчанию оставляем {prefer}"

    if loser == "comfy" and comfy.busy:
        return None, "ComfyUI старше, но ещё выполняет задачу: жду окончания"
    return loser, why


# ------------------------------------------------------------------ цикл

def fmt_age(t: Optional[float], now: float) -> str:
    if t is None:
        return "неизвестно"
    d = max(0, int(now - t))
    return f"{d // 60} мин {d % 60} с назад" if d >= 60 else f"{d} с назад"


def main() -> int:
    p = argparse.ArgumentParser(description="Держит в видеопамяти только один бэкенд: ComfyUI или LM Studio.")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    p.add_argument("--lms-url", default="http://127.0.0.1:1234")
    p.add_argument("--lms-token", default=os.environ.get("LM_API_TOKEN"))
    p.add_argument("--interval", type=float, default=3.0, help="пауза между проверками, сек")
    p.add_argument("--comfy-threshold-gb", type=float, default=1.5,
                   help="сколько ГБ torch VRAM считать признаком загруженной модели")
    p.add_argument("--prefer", choices=["lmstudio", "comfy"], default="lmstudio",
                   help="кого оставить, если непонятно, чья задача свежее")
    p.add_argument("--once", action="store_true", help="одна проверка и выход")
    p.add_argument("--dry-run", action="store_true", help="не выгружать, только сообщать")
    p.add_argument("--status", action="store_true", help="показать состояние и выйти")
    args = p.parse_args()

    comfy = Comfy(args.comfy_url, args.comfy_threshold_gb)
    lms = LMStudio(args.lms_url, args.lms_token)

    if args.status:
        now = time.time()
        for backend in (comfy, lms):
            s = backend.poll(now)
            state = "недоступен" if not s.reachable else ("ЗАГРУЖЕН" if s.loaded else "свободен")
            print(f"{backend.name:10} {state:11} {s.detail}; задача: {fmt_age(s.activity, now)}")
        return 0

    log(f"Слежу: ComfyUI {args.comfy_url}, LM Studio {args.lms_url}"
        + (" (dry-run)" if args.dry_run else ""))
    last_note = ""
    last_shape = None

    try:
        while True:
            now = time.time()
            cs, ls = comfy.poll(now), lms.poll(now)

            shape = (cs.reachable, cs.loaded, ls.reachable, ls.loaded)
            if shape != last_shape:
                def word(s: State) -> str:
                    return "недоступен" if not s.reachable else ("загружен" if s.loaded else "свободен")
                log(f"ComfyUI: {word(cs)} | LM Studio: {word(ls)}")
                last_shape = shape

            target, why = decide(cs, ls, args.prefer)

            if target is None:
                if why and why != last_note:
                    log(why)
                last_note = why
            else:
                backend = comfy if target == "comfy" else lms
                last_note = ""
                if args.dry_run:
                    log(f"[dry-run] выгрузил бы {backend.name}: {why}")
                else:
                    try:
                        backend.free(now)
                        log(f"Выгрузил {backend.name}: {why}")
                    except NET_ERRORS as e:
                        log(f"Не удалось выгрузить {backend.name}: {e}")

            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("Остановлен.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

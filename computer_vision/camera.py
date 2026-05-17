import cv2
import time
from threading import Thread
from queue import Queue, Empty


def _resolve_source(source: str):
    """
    Si source est une URL YouTube/HTTP, utilise yt-dlp pour obtenir l'URL directe du flux.
    Sinon retourne la source telle quelle (chemin fichier local).
    """
    if source.startswith("http://") or source.startswith("https://") or "youtube" in source:
        try:
            import yt_dlp
            ydl_opts = {"format": "best[ext=mp4]/best", "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                return info["url"], True
        except Exception as e:
            raise RuntimeError(f"[CameraStream] Impossible de résoudre le stream YouTube: {e}")
    return source, False


class CameraStream:
    def __init__(self, source, simulate_realtime=True, loop=True):
        """
        source : chemin fichier local  OU  URL YouTube live
        simulate_realtime : uniquement pour les fichiers locaux
        loop  : uniquement pour les fichiers locaux
        """
        self.source = source
        self.simulate_realtime = simulate_realtime
        self.loop = loop

        resolved, self._is_live = _resolve_source(source)
        self._resolved = resolved

        if self._is_live:
            # ── Mode live : thread dédié à la lecture ──────────────────
            self._queue = Queue(maxsize=2)
            self._running = True
            self._t = Thread(target=self._live_reader, daemon=True)
            self._t.start()
            self.fps = 25          # valeur par défaut pour un live
            self.delay = 1 / self.fps
            self.cap = None
        else:
            # ── Mode fichier : comportement original intact ─────────────
            self.cap = cv2.VideoCapture(resolved)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open source: {source}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
            self.delay = 1.0 / self.fps

    # ────────────────────────────────────────────────────────────────────
    # Thread interne (live uniquement)
    # ────────────────────────────────────────────────────────────────────
    def _live_reader(self):
        while self._running:
            try:
                # Ré-résout l'URL à chaque reconnexion (les URLs HLS expirent)
                resolved, _ = _resolve_source(self.source)
                cap = cv2.VideoCapture(resolved)

                if not cap.isOpened():
                    print("[CameraStream] Impossible d'ouvrir le flux, retry dans 5s...")
                    time.sleep(5)
                    continue

                real_fps = cap.get(cv2.CAP_PROP_FPS)
                if real_fps and real_fps > 0:
                    self.fps = real_fps
                    self.delay = 1.0 / self.fps

                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        print("[CameraStream] Frame perdue — reconnexion...")
                        break

                    # Garde uniquement la frame la plus récente (pas de décalage)
                    if self._queue.full():
                        try:
                            self._queue.get_nowait()
                        except Empty:
                            pass
                    self._queue.put(frame)

                cap.release()

            except Exception as e:
                print(f"[CameraStream] Erreur live: {e}")

            if self._running:
                time.sleep(5)  # pause avant reconnexion

    # ────────────────────────────────────────────────────────────────────
    # Interface publique — identique à avant
    # ────────────────────────────────────────────────────────────────────
    def read(self):
        if self._is_live:
            try:
                frame = self._queue.get(timeout=15)
                return True, frame
            except Empty:
                print("[CameraStream] Timeout — aucune frame reçue depuis 15s")
                return False, None
        else:
            # ── Comportement fichier original ──
            start = time.time()
            ret, frame = self.cap.read()

            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                else:
                    return False, None

            if self.simulate_realtime:
                elapsed = time.time() - start
                time.sleep(max(0, self.delay - elapsed))

            return ret, frame

    def release(self):
        self._running = False
        if self.cap:
            self.cap.release()
"""Filter One Euro (Casiez, Roussel, dan Vogel, CHI 2012) untuk meredam getar kursor.

Frekuensi potong naik seiring kecepatan: saat tangan diam getar diredam kuat
(``min_cutoff`` kecil), saat tangan bergerak cepat lag dikurangi (``beta``).
Waktu dalam detik dari jam monoton, bukan nomor frame, sehingga laju frame yang
berubah tidak mengubah perilaku filter.
"""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff dan d_cutoff harus > 0, beta >= 0")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._t = None
        self._x = None
        self._dx = 0.0

    def __call__(self, x: float, t: float) -> float:
        x = float(x)
        if self._t is None:
            self._t, self._x, self._dx = float(t), x, 0.0
            return x
        dt = float(t) - self._t
        if dt <= 0:
            raise ValueError("Cap waktu harus naik")
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = float(t)
        return self._x


class OneEuro2D:
    """Dua filter One Euro bebas untuk (x, y)."""

    def __init__(self, **kw):
        self.fx = OneEuro(**kw)
        self.fy = OneEuro(**kw)

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t: float) -> tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)

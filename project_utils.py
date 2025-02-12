# Questo file contiene gli utils per il progetto
import numpy as np

# Funzione di trasformazione symlog su array numpy
# La funzione è usata nella data analysis per gestire la compressione logaritmica con vbalori negativi.
def symlog_transform(x, linthresh=1):
    """
        Trasformazione SYMLOG:
        - Valori compresi tra -linthresh e +linthresh rimangono lineari.
        - Al di fuori, si applica una compressione logaritmica.

        Parametri:
            x (array-like): I dati di input (positivi e negativi).
            linthresh (float): Soglia entro cui la scala rimane lineare.

        Ritorna:
            np.ndarray: I dati trasformati secondo la symlog.
    """
    out = np.empty_like(x, dtype=float)
    mask_pos_log = (x >= linthresh)
    mask_neg_log = (x <= -linthresh)
    mask_linear  = ~ (mask_pos_log | mask_neg_log)

    out[mask_linear] = x[mask_linear]
    out[mask_pos_log] = linthresh + np.log(x[mask_pos_log] / linthresh)
    out[mask_neg_log] = - linthresh - np.log((-x[mask_neg_log]) / linthresh)
    return out

def symlog_inverse(y, linthresh=1):
    """
    Inversa della trasformazione symlog semplificata:
      - se |y| < linthresh: resta y (lineare)
      - se y > linthresh:   x = linthresh * exp(y - linthresh)
      - se y < -linthresh:  x = - linthresh * exp(-y - linthresh)
    """
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask_pos = (y >= linthresh)
    mask_neg = (y <= -linthresh)
    mask_lin = ~ (mask_pos | mask_neg)

    # lineare
    out[mask_lin] = y[mask_lin]
    # inversa log positivo
    out[mask_pos] = linthresh * np.exp(y[mask_pos] - linthresh)
    # inversa log negativo
    out[mask_neg] = - linthresh * np.exp(-y[mask_neg] - linthresh)

    return out

# Funzione per generare tick adattivi
def generate_log_ticks(vmin, vmax):
    """ Genera tick logaritmici che si espandono dinamicamente tra vmin e vmax. """
    if vmin == 0:
        vmin = 1e-2  # Evitiamo problemi con log(0)
    if vmax == 0:
        vmax = 1e-2

    min_exp = np.floor(np.log10(abs(vmin)))  # Trova l'esponente minimo
    max_exp = np.ceil(np.log10(abs(vmax)))   # Trova l'esponente massimo

    tick_values = []
    for exp in range(int(min_exp), int(max_exp) + 1):  # Espandiamo dinamicamente
        tick_values.extend([10**exp])  # Aggiungiamo più valori per coprire bene l'intervallo

    return np.array(sorted(set(tick_values)))  # Rimuoviamo duplicati e ordiniamo

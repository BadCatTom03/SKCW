"""PRC-style pseudorandom error-correcting code layer (v2 plan, Section 7.2).

Replaces the 48-bit message+CRC16 payload (CRC is detect-only: at raw acc ~0.94
3 bit errors already make CRC always fail, as observed in diag).

Design (spread-spectrum PRC-lite, soft-decision):
  encode : message (32 bit) -> codeword c in {0,1}^n
           each message bit is spread over r = n/32 pseudorandom chip positions
           and XORed with a keyed pseudorandom one-time pad.
           Codeword is key-dependent => doubles as a detection key (like PRC).
  decode : soft chip values (correlation R from VRF demod, sign = bit) ->
           per-message-bit LLR aggregation over its chips -> hard message +
           per-bit confidence + global detection score.
  detect : score = mean(|bit_llr|) / std_null; clean content gives score ~ 1,
           watermarked gives >> 1.  Threshold controls FPR (B2 L_null pairing).

Soft decode recovers the full message down to chip acc ~0.75-0.8, matching the
B2 + alpha=0.4 working range (raw acc 0.81-0.94).
"""
from __future__ import annotations

import hashlib
from typing import List, Tuple

import numpy as np

MSG_BITS = 32
DEFAULT_N = 512  # chips; r = n / MSG_BITS = 16 chips per message bit
# n=512 in a 16x5x60x104 (~500k dim) latent: carriers remain near-orthogonal;
# soft decode recovers full 32-bit message down to chip acc ~0.75.


def _rng_from_key(key: bytes, label: str) -> np.random.Generator:
    h = hashlib.sha256(key + b"|prc|" + label.encode()).digest()
    return np.random.Generator(np.random.PCG64(list(h[:16])))


def _pad_and_perm(key: bytes, n: int,
                  msg_bits: int = MSG_BITS) -> Tuple[np.ndarray, np.ndarray]:
    """Keyed one-time pad bits[n] and chip->message-bit assignment perm[n].

    msg_bits is the effective payload width.  Redundancy r = n // msg_bits is
    the per-bit repetition (votes).  Capacity experiments sweep msg_bits at
    fixed n to trade redundancy (robustness) for capacity.
    The pad/perm RNG labels include msg_bits so distinct widths get distinct
    keyed assignments (no accidental aliasing between configs).
    """
    pad = (_rng_from_key(key, f"pad{n}").random(n) < 0.5).astype(np.uint8)
    # balanced assignment: each message bit gets exactly r chips, then shuffle
    r = n // msg_bits
    assign = np.repeat(np.arange(msg_bits), r).astype(np.int64)
    rest = n - r * msg_bits
    if rest:
        assign = np.concatenate([assign, np.arange(rest) % msg_bits])
    _rng_from_key(key, f"perm{n}m{msg_bits}").shuffle(assign)
    return pad, assign


def int_to_bits(x: int, n: int = MSG_BITS) -> List[int]:
    return [(x >> (n - 1 - i)) & 1 for i in range(n)]


def bits_to_int(bits) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def prc_encode(message: int, key: bytes, n: int = DEFAULT_N,
               msg_bits: int = MSG_BITS) -> np.ndarray:
    """message (msg_bits-wide int) -> codeword bits {0,1}^n (np.uint8)."""
    mbits = np.asarray(int_to_bits(message, msg_bits), dtype=np.uint8)
    pad, assign = _pad_and_perm(key, n, msg_bits)
    return (mbits[assign] ^ pad).astype(np.uint8)


def prc_soft_decode(soft_chips: np.ndarray, key: bytes,
                    n: int = DEFAULT_N, msg_bits: int = MSG_BITS) -> dict:
    """soft_chips: length-n float array, sign convention chip=1 -> positive
    (e.g. VRF correlation R values).  Returns message + confidence + score."""
    soft = np.asarray(soft_chips, dtype=np.float64)
    assert soft.shape == (n,), f"expected ({n},), got {soft.shape}"
    pad, assign = _pad_and_perm(key, n, msg_bits)
    # remove pad: chip = msg_bit XOR pad  =>  msg-soft = soft * (+1 if pad==0 else -1)
    depad = soft * np.where(pad == 0, 1.0, -1.0)
    # vectorized per-message-bit LLR aggregation (bincount over assignment)
    llr = np.bincount(assign, weights=depad, minlength=msg_bits)
    mbits = (llr > 0).astype(np.uint8)
    # null calibration: chip-level std -> per-bit-llr std under H0
    chip_std = float(soft.std()) + 1e-12
    counts = np.bincount(assign, minlength=msg_bits).astype(np.float64)
    llr_null_std = chip_std * np.sqrt(np.maximum(counts, 1.0))
    z = np.abs(llr) / llr_null_std
    score = float(z.mean())
    return {
        "message": bits_to_int(mbits),
        "message_bits": mbits,
        "bit_llr": llr,
        "bit_conf": z,
        "score": score,           # H0 ~ 0.8 (E|N(0,1)|), watermarked >> 1
        "chip_acc_proxy": float((np.sign(depad) > 0).mean()),
        "msg_bits": msg_bits,
        "redundancy": n // msg_bits,
    }


def chip_acc(soft_chips: np.ndarray, codeword: np.ndarray) -> float:
    """Raw chip accuracy of hard-thresholded soft values vs true codeword."""
    hard = (np.asarray(soft_chips) > 0).astype(np.uint8)
    return float((hard == np.asarray(codeword, dtype=np.uint8)).mean())


def msg_bit_acc(decoded: dict, message: int) -> float:
    mb = int(decoded.get("msg_bits", MSG_BITS))
    true_bits = np.asarray(int_to_bits(message, mb), dtype=np.uint8)
    return float((decoded["message_bits"] == true_bits).mean())


if __name__ == "__main__":  # self-test
    rng = np.random.default_rng(0)
    key = b"test-key-0123456"
    msg = 0xA5A5BEEF
    cw = prc_encode(msg, key)
    for chip_p in (1.0, 0.9, 0.8, 0.75, 0.7):
        soft = (cw * 2.0 - 1.0) + rng.normal(0, 1e-9, cw.shape)
        flip = rng.random(cw.shape) > chip_p
        soft = np.where(flip, -soft, soft) * np.abs(rng.normal(1.0, 0.3, cw.shape))
        dec = prc_soft_decode(soft, key)
        print(f"chip_p={chip_p:.2f} msg_ok={dec['message'] == msg} "
              f"bitacc={msg_bit_acc(dec, msg):.3f} score={dec['score']:.2f}")
    # null
    dec0 = prc_soft_decode(rng.normal(0, 1, cw.shape), key)
    print(f"null score={dec0['score']:.2f} (expect ~0.8)")

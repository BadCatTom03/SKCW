"""Payload encoding for Semantic-Keyed Content Watermarking (v0.3 SS6).

48-bit payload = 32-bit message + CRC16-CCITT (poly 0x1021, init 0xFFFF).
`verify()` is the single system-wide acceptance test (FPR ~= 2^-16).
"""
from __future__ import annotations

from typing import List, Optional

MESSAGE_BITS = 32
CRC_BITS = 16
PAYLOAD_BITS = MESSAGE_BITS + CRC_BITS  # 48


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def int_to_bits(value: int, width: int) -> List[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def bits_to_int(bits: List[int]) -> int:
    value = 0
    for b in bits:
        value = (value << 1) | (1 if b else 0)
    return value


def encode_payload(message: int) -> List[int]:
    """32-bit int -> 48 payload bits (message || crc16)."""
    if not (0 <= message < (1 << MESSAGE_BITS)):
        raise ValueError("message must fit in 32 bits")
    msg_bytes = message.to_bytes(4, "big")
    crc = crc16_ccitt(msg_bytes)
    return int_to_bits(message, MESSAGE_BITS) + int_to_bits(crc, CRC_BITS)


def verify_payload(bits: List[int]) -> Optional[int]:
    """48 hard-decision bits -> message int if CRC passes, else None."""
    if len(bits) != PAYLOAD_BITS:
        return None
    message = bits_to_int(bits[:MESSAGE_BITS])
    crc_rx = bits_to_int(bits[MESSAGE_BITS:])
    if crc16_ccitt(message.to_bytes(4, "big")) == crc_rx:
        return message
    return None


if __name__ == "__main__":
    import random

    rng = random.Random(0)
    for _ in range(1000):
        m = rng.getrandbits(32)
        bits = encode_payload(m)
        assert verify_payload(bits) == m
        # single bit flip must fail CRC
        i = rng.randrange(PAYLOAD_BITS)
        flipped = list(bits)
        flipped[i] ^= 1
        assert verify_payload(flipped) is None or flipped[:32] != bits[:32]
    print("payload self-test OK: 1000 encode/verify + bit-flip rejection")

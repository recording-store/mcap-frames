"""Bitstream parsing: only the headers needed to know where decoding can start and what it depends on."""

from mcap_frames.codecs.detect import codec_from_format, sniff_codec

__all__ = ["codec_from_format", "sniff_codec"]

"""Tests for symbol normalization and the no-data routing sentinel."""

import unittest

import pytest

from tradingagents.dataflows.symbol_utils import (
    NoMarketDataError,
    ashare_suffix,
    crypto_base,
    is_yahoo_safe,
    normalize_symbol,
)


@pytest.mark.unit
class TestNormalizeSymbol(unittest.TestCase):
    def test_plain_equities_unchanged(self):
        for sym in ("AAPL", "MSFT", "TSM", "BRK.B", "0700.HK", "^GSPC", "GC=F"):
            self.assertEqual(normalize_symbol(sym), sym)

    def test_lowercases_are_upper(self):
        self.assertEqual(normalize_symbol("aapl"), "AAPL")
        self.assertEqual(normalize_symbol("  msft  "), "MSFT")

    def test_metal_aliases_map_to_futures(self):
        self.assertEqual(normalize_symbol("XAUUSD"), "GC=F")
        self.assertEqual(normalize_symbol("XAUUSD+"), "GC=F")   # broker CFD suffix
        self.assertEqual(normalize_symbol("xauusd+"), "GC=F")
        self.assertEqual(normalize_symbol("GOLD"), "GC=F")
        self.assertEqual(normalize_symbol("XAGUSD"), "SI=F")

    def test_energy_and_index_aliases(self):
        self.assertEqual(normalize_symbol("USOIL"), "CL=F")
        self.assertEqual(normalize_symbol("SPX500"), "^GSPC")
        self.assertEqual(normalize_symbol("NAS100"), "^NDX")
        self.assertEqual(normalize_symbol("US30"), "^DJI")

    def test_forex_pairs_get_x_suffix(self):
        self.assertEqual(normalize_symbol("EURUSD"), "EURUSD=X")
        self.assertEqual(normalize_symbol("GBPJPY"), "GBPJPY=X")
        self.assertEqual(normalize_symbol("eurusd"), "EURUSD=X")

    def test_crypto_pairs_get_dash_usd(self):
        self.assertEqual(normalize_symbol("BTCUSD"), "BTC-USD")
        self.assertEqual(normalize_symbol("ETHUSD"), "ETH-USD")

    def test_six_letter_non_currency_left_alone(self):
        # GOOGLE-style 6-letter tickers that aren't two currency codes
        # must not be mangled into a fake forex pair.
        self.assertEqual(normalize_symbol("ABCDEF"), "ABCDEF")

    def test_empty_input_passthrough(self):
        self.assertEqual(normalize_symbol(""), "")


@pytest.mark.unit
class TestAshareSuffix(unittest.TestCase):
    def test_sse_prefixes(self):
        # Shanghai main board / STAR market / B-share
        for raw, expected in (
            ("600759", "600759.SS"),   # 洲际油气
            ("600519", "600519.SS"),   # 贵州茅台
            ("601318", "601318.SS"),
            ("688981", "688981.SS"),   # STAR market
            ("900939", "900939.SS"),   # SSE B-share
        ):
            self.assertEqual(normalize_symbol(raw), expected)

    def test_szse_prefixes(self):
        for raw, expected in (
            ("000001", "000001.SZ"),   # 平安银行
            ("002594", "002594.SZ"),
            ("300750", "300750.SZ"),   # ChiNext
            ("301536", "301536.SZ"),
            ("200596", "200596.SZ"),   # SZSE B-share
        ):
            self.assertEqual(normalize_symbol(raw), expected)

    def test_bse_prefixes(self):
        for raw, expected in (
            ("430047", "430047.BJ"),
            ("832566", "832566.BJ"),
            ("873223", "873223.BJ"),
            ("920002", "920002.BJ"),
        ):
            self.assertEqual(normalize_symbol(raw), expected)

    def test_unknown_six_digit_untouched(self):
        # Not a recognizable A-share board number: left as-is (e.g. used by
        # other six-digit markets via explicit suffixes only).
        self.assertEqual(normalize_symbol("123456"), "123456")
        self.assertEqual(normalize_symbol("987654"), "987654")

    def test_suffixed_or_non_numeric_untouched(self):
        # Already-canonical and other-market symbols must not be mangled.
        for raw in (
            "600519.SS",
            "000001.SZ",
            "005930.KS",   # Samsung (Korea) requires the explicit suffix
            "AAPL",
            "12345",
            "1234567",
        ):
            self.assertEqual(normalize_symbol(raw), raw)

    def test_korean_style_bare_code_collides_with_szse_by_design(self):
        # Known trade-off: bare six-digit codes are assumed A-share, so a
        # Korean ticker typed without its .KS suffix reads as Shenzhen.
        # Documented in the jiying help text; explicit suffix wins.
        self.assertEqual(normalize_symbol("005930"), "005930.SZ")
        self.assertEqual(normalize_symbol("005930.KS"), "005930.KS")

    def test_helper_returns_none_for_non_ashare(self):
        self.assertIsNone(ashare_suffix("AAPL"))
        self.assertIsNone(ashare_suffix("12345"))
        self.assertIsNone(ashare_suffix("600519.SS"))
        self.assertIsNone(ashare_suffix(""))
        self.assertEqual(ashare_suffix("600759"), ".SS")


@pytest.mark.unit
class TestNoMarketDataError(unittest.TestCase):
    def test_message_includes_resolution(self):
        err = NoMarketDataError("XAUUSD+", "GC=F", "no rows")
        self.assertIn("XAUUSD+", str(err))
        self.assertIn("GC=F", str(err))
        self.assertEqual(err.symbol, "XAUUSD+")
        self.assertEqual(err.canonical, "GC=F")

    def test_canonical_defaults_to_symbol(self):
        err = NoMarketDataError("FOOBAR")
        self.assertEqual(err.canonical, "FOOBAR")


@pytest.mark.unit
class TestIsYahooSafe(unittest.TestCase):
    def test_accepts_structural_chars(self):
        for sym in ("AAPL", "GC=F", "^GSPC", "BRK.B", "BTC-USD"):
            self.assertTrue(is_yahoo_safe(sym))

    def test_rejects_slash_and_space(self):
        for sym in ("a/b", "AA PL", ""):
            self.assertFalse(is_yahoo_safe(sym))


@pytest.mark.unit
class TestCryptoBase(unittest.TestCase):
    def test_resolves_known_crypto_forms(self):
        for raw in ("BTC-USD", "BTCUSD", "btc-usdt", "BTC-USDC", "BTCUSD+"):
            self.assertEqual(crypto_base(raw), "BTC")
        self.assertEqual(crypto_base("ETH-USD"), "ETH")
        self.assertEqual(crypto_base("sol-usd"), "SOL")

    def test_non_crypto_returns_none(self):
        # Plain equities, class shares, and real tickers that alias elsewhere
        # (GOLD -> gold future on the Yahoo path) must NOT read as crypto.
        for raw in ("AAPL", "BRK-B", "GOLD", "XYZ-USD", "EURUSD", "", None):
            self.assertIsNone(crypto_base(raw))

    def test_agrees_with_normalize_symbol(self):
        # crypto_base is the shared primitive behind the -USD normalization.
        self.assertEqual(normalize_symbol("BTCUSD"), "BTC-USD")
        self.assertEqual(crypto_base("BTCUSD"), "BTC")


if __name__ == "__main__":
    unittest.main()

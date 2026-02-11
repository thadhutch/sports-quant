from sports_quant.processing.over_under import set_total, extract_ou_line


class TestSetTotal:
    def test_over(self):
        assert set_total("45.5 (over)") == 1

    def test_under(self):
        assert set_total("45.5 (under)") == 0

    def test_push_returns_none(self):
        assert set_total("45.5 (push)") is None

    def test_no_parens_returns_none(self):
        assert set_total("45.5") is None


class TestExtractOuLine:
    def test_extracts_line(self):
        assert extract_ou_line("45.5 (over)") == 45.5

    def test_integer_line(self):
        assert extract_ou_line("44 (under)") == 44.0

    def test_invalid_returns_none(self):
        assert extract_ou_line("abc (over)") is None

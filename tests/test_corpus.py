from timesafe.data.corpus import extract_main_text, parse_sgml_items

SGML = """<SEC-DOCUMENT>0000320193-20-000010.txt
<SEC-HEADER>
CONFORMED SUBMISSION TYPE:\t8-K
ITEM INFORMATION:\tResults of Operations and Financial Condition
ITEM INFORMATION:\tFinancial Statements and Exhibits
FILED AS OF DATE:\t20200128
</SEC-HEADER>
<DOCUMENT>
<TYPE>8-K
<TEXT>
<html><body><p>Cover page only.</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<TEXT>
<html><style>p { color: red }</style><body>
<h1>Apple Reports Record First Quarter Results</h1>
<p>Revenue of $91.8&nbsp;billion, up 9 percent.</p>
</body></html>
</TEXT>
</DOCUMENT>
"""


def test_parse_sgml_items_reads_header():
    items = parse_sgml_items(SGML)
    assert "Results of Operations and Financial Condition" in items
    assert len(items) == 2


def test_extract_main_text_prefers_press_release_exhibit():
    text = extract_main_text(SGML)
    assert "Record First Quarter Results" in text
    assert "Revenue of $91.8" in text
    # Cover-page document and markup must not leak through.
    assert "Cover page only" not in text
    assert "<" not in text and "color: red" not in text


def test_extract_main_text_truncates():
    assert len(extract_main_text(SGML, max_chars=50)) <= 50


def test_extract_main_text_falls_back_to_first_document():
    no_exhibit = SGML.replace("EX-99.1", "EX-10.5")
    text = extract_main_text(no_exhibit)
    assert "Cover page only" in text

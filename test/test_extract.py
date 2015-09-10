"""
Tests for dcmstack.extract
"""
import sys, warnings
from os import path
from nose.tools import ok_, eq_, assert_raises

test_dir = path.dirname(__file__)
src_dir = path.normpath(path.join(test_dir, '../src'))
sys.path.insert(0, src_dir)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from dcmstack import extract
dicom, csareader = extract.dicom, extract.csareader

class TestCsa(object):
    def setUp(self):
        data_fn = path.join(test_dir, 'data', 'extract', 'csa_test.dcm')
        self.data = dicom.read_file(data_fn)

    def tearDown(self):
        del self.data

    def test_simplify(self):
        eq_(extract.simplify_csa_dict(None), None)
        csa_dict = csareader.read(self.data[dicom.tag.Tag(0x29, 0x1010)].value)
        simp_dict = extract.simplify_csa_dict(csa_dict)
        for tag in csa_dict['tags']:
            items = csa_dict['tags'][tag]['items']
            if len(items) == 0:
                ok_(not tag in simp_dict)
            elif len(items) == 1:
                eq_(simp_dict[tag], items[0])
            else:
                eq_(simp_dict[tag], items)

    def test_csa_image_trans(self):
        csa_dict = extract.csa_series_trans_func(self.data[(0x29, 0x1010)])
        eq_(csa_dict["EchoLinePosition"], 64)

    def test_parse_phx_line(self):
        ok_(extract._parse_phoenix_line("") is None)
        ok_(extract._parse_phoenix_line("#test = 2") is None)

        eq_(extract._parse_phoenix_line('test = "" 2 ""'), ('test', ' 2 '))
        eq_(extract._parse_phoenix_line('test = "" #2 ""'), ('test', ' #2 '))
        eq_(extract._parse_phoenix_line('test = "" 2 ""#='),
            ('test', ' 2 '))
        eq_(extract._parse_phoenix_line("test = 2#"), ('test', 2))
        eq_(extract._parse_phoenix_line("test = 0x2"), ('test', 2))
        eq_(extract._parse_phoenix_line("test = 2."), ('test', 2.0))

        assert_raises(extract.PhoenixParseError,
                      extract._parse_phoenix_line,
                      'test = blah')
        assert_raises(extract.PhoenixParseError,
                      extract._parse_phoenix_line,
                      '==')
        assert_raises(extract.PhoenixParseError,
                      extract._parse_phoenix_line,
                      'test')
        assert_raises(extract.PhoenixParseError,
                      extract._parse_phoenix_line,
                      'test = "" 2 ""3')

    def test_csa_series_trans(self):
        csa_dict = extract.csa_series_trans_func(self.data[(0x29, 0x1020)])
        eq_(csa_dict['MrPhoenixProtocol.sEFISPEC.bEFIDataValid'], 1)

class TestMetaExtractor(object):
    def setUp(self):
        data_fn = path.join(test_dir, 'data', 'extract', 'csa_test.dcm')
        self.data = dicom.read_file(data_fn)

    def tearDown(self):
        del self.data

    def test_get_elem_key(self):
        ignore_rules = (extract.ignore_pixel_data,)
        extractor = extract.MetaExtractor(ignore_rules=ignore_rules)
        for elem in self.data:
            key = extractor._get_elem_key(elem)
            ok_(key.strip() != '')
            ok_(key[0].isalpha())
            ok_(key[-1].isalnum())

    def test_get_elem_value(self):
        ignore_rules = (extract.ignore_pixel_data,)
        extractor = extract.MetaExtractor(ignore_rules=ignore_rules)
        for elem in self.data:
            value = extractor._get_elem_value(elem)
            if elem.VM > 1:
                ok_(isinstance(value, list))
            if elem.VR in extract.unpack_vr_map.keys() + ['DS', 'IS']:
                if elem.VM == 1:
                    ok_(not isinstance(value, str))
                else:
                    ok_(not any(isinstance(val, str) for val in value))

    def test_dup_trans(self):
        translators = [extract.csa_image_trans, extract.csa_image_trans]
        extractor = extract.MetaExtractor(translators=translators)
        assert_raises(ValueError, extractor, self.data)

    def test_reloc_private(self):
        extractor = extract.MetaExtractor()
        self.data[(0x29, 0x10)].tag = dicom.tag.Tag((0x29, 0x20))
        self.data[(0x29, 0x1010)].tag = dicom.tag.Tag((0x29, 0x2010))
        self.data[(0x29, 0x1020)].tag = dicom.tag.Tag((0x29, 0x2020))
        meta_dict = extractor(self.data)
        eq_(meta_dict["CsaImage.EchoLinePosition"], 64)
        ok_(meta_dict['CsaSeries.MrPhoenixProtocol.sEFISPEC.bEFIDataValid'], 1)

    def test_non_reloc_private(self):
        extractor = extract.MetaExtractor()
        meta_dict = extractor(self.data)
        eq_(meta_dict["CsaImage.EchoLinePosition"], 64)
        ok_(meta_dict['CsaSeries.MrPhoenixProtocol.sEFISPEC.bEFIDataValid'], 1)
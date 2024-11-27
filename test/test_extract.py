"""
Tests for dcmstack.extract
"""
import sys, warnings
from os import path

import pytest
from nibabel.nicom import csareader
try:
    import pydicom
except ImportError:
    import dicom as pydicom

from . import test_dir, src_dir

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from dcmstack import extract

class TestCsa(object):
    def setup_method(self, method):
        data_fn = path.join(test_dir, 'data', 'extract', 'csa_test.dcm')
        self.data = pydicom.read_file(data_fn)

    def teardown_method(self, method):
        del self.data

    def test_simplify(self):
        assert extract.simplify_csa_dict(None) == None
        csa_dict = csareader.read(self.data[pydicom.tag.Tag(0x29, 0x1010)].value)
        simp_dict = extract.simplify_csa_dict(csa_dict)
        for tag in csa_dict['tags']:
            items = csa_dict['tags'][tag]['items']
            if len(items) == 0:
                assert(not tag in simp_dict)
            elif len(items) == 1:
                assert simp_dict[tag] == items[0]
            else:
                assert simp_dict[tag] == items

    def test_csa_image_trans(self):
        csa_dict = extract.csa_series_trans_func(self.data[(0x29, 0x1010)])
        assert csa_dict["EchoLinePosition"] == 64

    def test_parse_phx_line(self):
        assert(extract._parse_phoenix_line("") is None)
        assert(extract._parse_phoenix_line("#test = 2") is None)

        assert extract._parse_phoenix_line('test = "" 2 ""') == ('test', ' 2 ')
        assert extract._parse_phoenix_line('test = "" #2 ""'), ('test', ' #2 ')
        assert extract._parse_phoenix_line('test = "" 2 ""#=') == ('test', ' 2 ')
        assert extract._parse_phoenix_line("test = 2#") == ('test', 2)
        assert extract._parse_phoenix_line("test = 0x2") == ('test', 2)
        assert extract._parse_phoenix_line("test = 2.") == ('test', 2.0)

        
        with pytest.raises(extract.PhoenixParseError):
            extract._parse_phoenix_line('test = blah')
        with pytest.raises(extract.PhoenixParseError):
            extract._parse_phoenix_line('==')
        with pytest.raises(extract.PhoenixParseError):
            extract._parse_phoenix_line('test')
        with pytest.raises(extract.PhoenixParseError):
            extract._parse_phoenix_line('test = "" 2 ""3')

    def test_csa_series_trans(self):
        csa_dict = extract.csa_series_trans_func(self.data[(0x29, 0x1020)])
        assert csa_dict['MrPhoenixProtocol.sEFISPEC.bEFIDataValid'] == 1

class TestMetaExtractor(object):
    def setup_method(self, method):
        data_fn = path.join(test_dir, 'data', 'extract', 'csa_test.dcm')
        self.data = pydicom.read_file(data_fn)

    def teardown_method(self,method):
        del self.data

    def test_get_elem_key(self):
        ignore_rules = (extract.ignore_pixel_data,)
        extractor = extract.MetaExtractor(ignore_rules=ignore_rules)
        for key in extractor(self.data):
            assert(key.strip() != '')
            assert(key[0].isalpha())

    def test_get_elem_value(self):
        ignore_rules = (extract.ignore_pixel_data,)
        extractor = extract.MetaExtractor(ignore_rules=ignore_rules)
        for elem in self.data:
            value = extractor._get_elem_value(elem)
            if elem.VM > 1:
                assert(isinstance(value, list))
            if elem.VR in list(extract.unpack_vr_map) + ['DS', 'IS']:
                if elem.VM == 1:
                    assert(not isinstance(value, str))
                else:
                    assert(not any(isinstance(val, str) for val in value))

    def test_dup_trans(self):
        translators = [extract.csa_image_trans, extract.csa_image_trans]
        extractor = extract.MetaExtractor(translators=translators)
        with pytest.raises(ValueError):
            extractor(self.data)

    def test_reloc_private(self):
        extractor = extract.MetaExtractor()
        elem = self.data[(0x29, 0x10)]
        elem.tag = pydicom.tag.Tag((0x29, 0x20))
        self.data[(0x29, 0x20)] = elem
        elem = self.data[(0x29, 0x1010)]
        elem.tag = pydicom.tag.Tag((0x29, 0x2010))
        self.data[(0x29, 0x2010)] = elem
        elem = self.data[(0x29, 0x1020)]
        elem.tag = pydicom.tag.Tag((0x29, 0x2020))
        self.data[(0x29, 0x2020)] = elem
        del self.data[(0x29, 0x10)]
        del self.data[(0x29, 0x1010)]
        del self.data[(0x29, 0x1020)]
        meta_dict = extractor(self.data)
        assert meta_dict["CsaImage.EchoLinePosition"] == 64
        assert meta_dict['CsaSeries.MrPhoenixProtocol.sEFISPEC.bEFIDataValid'] == 1

    def test_non_reloc_private(self):
        extractor = extract.MetaExtractor()
        meta_dict = extractor(self.data)
        assert meta_dict["CsaImage.EchoLinePosition"] == 64
        assert meta_dict['CsaSeries.MrPhoenixProtocol.sEFISPEC.bEFIDataValid'] == 1

    def test_none_vals(self):
        extractor = extract.MetaExtractor()
        self.data.PercentSampling = None
        meta_data = extractor(self.data)
        assert "PercentSampling" not in meta_data

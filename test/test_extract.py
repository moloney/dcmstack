"""
Tests for dcmstack.extract
"""
import sys
from os import path

test_dir = path.dirname(__file__)
src_dir = path.normpath(path.join(test_dir, '../src'))
sys.path.insert(0, src_dir)

from nose.tools import ok_, eq_, assert_raises
import dicom
from nibabel.nicom import csareader
from dcmstack import extract

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

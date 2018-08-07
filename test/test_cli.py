import sys, os, shutil
from tempfile import mkdtemp
from glob import glob

test_dir = os.path.dirname(__file__)
src_dir = os.path.normpath(os.path.join(test_dir, '../src'))
sys.path.insert(0, src_dir)

from dcmstack import dcmstack_cli, nitool_cli


class TestDcmstackCli(object):
    def setUp(self):
        self.tmp_dir = mkdtemp(prefix='dcmstack_cli_test')
        self.data_dir = os.path.join(test_dir,
                                     'data',
                                     'dcmstack',
                                     '2D_16Echo_qT2')
        self.fns = ('TE_20_SlcPos_-33.707626341697.dcm',
                    'TE_20_SlcPos_-23.207628249046.dcm',
                    'TE_40_SlcPos_-33.707626341697.dcm',
                    'TE_40_SlcPos_-23.207628249046.dcm',
                   )
        for fn in self.fns:
            shutil.copyfile(os.path.join(self.data_dir, fn), 
                            os.path.join(self.tmp_dir, fn))
    
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
    
    def test_defaults(self):
        dcmstack_cli.main(['dcmstack', self.tmp_dir])
        nii_paths = glob(os.path.join(self.tmp_dir, '*.nii.gz'))
        eq_(len(nii_paths), 1)

    def test_embed(self):
        dcmstack_cli.main(['dcmstack', '--embed', self.tmp_dir])
        nii_paths = glob(os.path.join(self.tmp_dir, '*.nii.gz'))
        eq_(len(nii_paths), 1)

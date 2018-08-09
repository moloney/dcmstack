import sys, shutil, json
from os import path as opath
from tempfile import mkdtemp
from glob import glob

import numpy as np

from nose.tools import ok_, eq_

test_dir = opath.dirname(__file__)
src_dir = opath.normpath(opath.join(test_dir, '../src'))
sys.path.insert(0, src_dir)

from dcmstack import dcmstack_cli, nitool_cli

test_data = {}

def setup_module():
    input_dir = mkdtemp(prefix='dcmstack_cli_test_in')
    data_dir = opath.join(test_dir,
                          'data',
                          'dcmstack',
                          '2D_16Echo_qT2')
    fns = ('TE_20_SlcPos_-33.707626341697.dcm',
           'TE_20_SlcPos_-23.207628249046.dcm',
           'TE_40_SlcPos_-33.707626341697.dcm',
           'TE_40_SlcPos_-23.207628249046.dcm',
           )
    for fn in fns:
        shutil.copyfile(opath.join(data_dir, fn), opath.join(input_dir, fn))
    test_data['base_dir'] = input_dir


def teardown_module():
    shutil.rmtree(test_data['base_dir'])
    del test_data['base_dir']


def make_niftis(out_dir, extra_args=tuple()):
    '''Helper function runs dcmstack and returns any Nifti files  
    '''
    assert len(glob(opath.join(out_dir, '*.nii*'))) == 0
    args = ['dcmstack', test_data['base_dir'], '--dest-dir', out_dir]
    args.extend(extra_args)
    dcmstack_cli.main(args)
    return glob(opath.join(out_dir, '*.nii*'))


class CliTest(object):
    def setup(self):
        self.out_dir = mkdtemp(prefix='dcmstack_cli_test_out')

    def teardown(self):
        shutil.rmtree(self.out_dir)


class TestDcmstackCli(CliTest):
    def test_basic(self):
        nii_paths = make_niftis(self.out_dir)
        eq_(len(nii_paths), 1)

    def test_embed(self):
        nii_paths = make_niftis(self.out_dir, ['--embed'])
        eq_(len(nii_paths), 1)


class TestNitoolCli(CliTest):
    def test_basic(self):
        nii_path = make_niftis(self.out_dir, ['--embed'])[0]
        json_path = opath.join(self.out_dir, 'meta.json')
        nitool_cli.main(['nitool', 'dump', nii_path, json_path])
        with open(json_path) as fp:
            meta = json.load(fp)
        print(json.dumps(meta, indent=4))
        ok_('dcmmeta_version' in meta)
        ok_('dcmmeta_affine' in meta)
        ok_('dcmmeta_reorient_transform' in meta)
        eq_(meta['dcmmeta_shape'], [192, 192, 2, 2])
        eq_(meta['dcmmeta_slice_dim'], 2)
        eq_(meta['global']['const']['Rows'], 192)
        eq_(meta['global']['const']['Columns'], 192)
        ok_(np.allclose(meta['time']['samples']['EchoTime'], [20.0, 40.0]))

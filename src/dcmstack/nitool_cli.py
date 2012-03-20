"""
Command line interface for nitool.

@author: moloney
"""
import os, sys, argparse
import nibabel as nb
from .dcmmeta import NiftiWrapper, DcmMetaExtension

prog_descrip = """Work with extended Nifti files created by dcmstack"""

def main(argv=sys.argv):
    #Setup the top level parser
    arg_parser = argparse.ArgumentParser(description=prog_descrip)
    sub_parsers = arg_parser.add_subparsers(title="Subcommands")
    
    #Split command 
    split_parser = sub_parsers.add_parser('split', 
                                          help=("Split src_nii file along a "
                                          "dimension. Defaults to the slice "
                                          "dimension if 3D, otherwise the last"
                                          "dimension.")
                                         )
    split_parser.add_argument('src_nii', nargs=1)
    split_parser.add_argument('-d', '--dimension', default=None, type=int,
                              help=("The dimension to split along"))
    split_parser.add_argument('-o', '--output-format', default=None,
                              help=("Format string used to create the output "
                              "file names."))
    split_parser.set_defaults(func=split)
                              
    #Merge Command
    merge_parser = sub_parsers.add_parser('merge',
                                          help=("Merge the provided Nifti "
                                                "files along a dimension. "
                                                "Defaults to slice, then time, "
                                                "and then vector.")
                                         )
    merge_parser.add_argument('output', nargs=1)
    merge_parser.add_argument('src_niis', nargs='+')
    merge_parser.add_argument('-d', '--dimension', default=None, 
                              help=("The dimension to split along. Must be "
                                    "in range [0, 5)")
                             )
    merge_parser.add_argument('-s', '--sort', default=None,
                              help=("Sort the source files using the provided "
                              "meta data key before merging.")
                             )
    merge_parser.add_argument('-c', '--clear-slices', default=False, 
                              help=("Clear all per slice meta data"))
    merge_parser.set_defaults(func=merge)
    
    #Dump Command
    dump_parser = sub_parsers.add_parser('dump', 
                                         help=("Dump the JSON meta data "
                                         "extension from the provided Nifti")
                                        )
    dump_parser.add_argument('src_nii', nargs=1)
    dump_parser.add_argument('dest_json', nargs='?', type=argparse.FileType('w'),
                             default=sys.stdout)
    dump_parser.add_argument('-m', '--make-empty', default=False, 
                             action='store_true',
                             help=("Make an empty extension of none exists"))
    dump_parser.add_argument('-r', '--remove', default=False, 
                             action='store_true',
                             help="Remove the extension from the Nifti file")
    dump_parser.set_defaults(func=dump)
                             
    #Embed Command
    embed_parser = sub_parsers.add_parser('embed', 
                                          help=("Embed a JSON extension into "
                                                "the Nifti file"))
    embed_parser.add_argument('src_json', nargs='?', type=argparse.FileType('r'),
                              default=sys.stdin)
    embed_parser.add_argument('dest_nii', nargs=1)
    embed_parser.add_argument('-c', '--console', 
                              help="Read from stdin instead of a file")
    embed_parser.add_argument('-f', '--force-overwrite', 
                              help="Overwrite any existing dcmmeta extension")
    embed_parser.set_defaults(func=embed)
                              
    lookup_parser = sub_parsers.add_parser('lookup', 
                                           help=("Lookup the value for the "
                                                 "given meta data key"))
    lookup_parser.add_argument('key', nargs=1)
    lookup_parser.add_argument('src_nii', nargs=1)
    lookup_parser.add_argument('-i', '--index',
                               help=("Use the given voxel index"))
    lookup_parser.set_defaults(func=lookup)
    
    #Parse the arguments and call the appropriate funciton
    args = arg_parser.parse_args(argv[1:])
    args.func(args)
    
def split(args):
    src_path = args.src_nii[0]
    src_fn = os.path.basename(src_path)
    src_dir = os.path.dirname(src_path)
    
    src_wrp = NiftiWrapper.from_filename(src_path)
    for split_idx, split in enumerate(src_wrp.generate_splits(args.dimension)):
        if args.output_format:
            out_name = (args.output_format % 
                        split.meta_ext.get_class_dict(('global', 'const'))
                       )
        else:
            out_name = os.path.join(src_dir, '%d-%s' % (split_idx, src_fn))
        nb.save(split, out_name)
    
def make_key_func(meta_key, index=None):
    def key_func(src_nii):
        result = src_nii.get_meta(meta_key, index)
        if result is None:
            raise ValueError('Key not found: %s' ) % meta_key
        return result
    
    return key_func
    
def merge(args):
    src_wrps = [NiftiWrapper.from_filename(src_path) 
                for src_path in args.src_niis]
    if args.sort:
        src_wrps.sort(key=make_key_func(args.sort))
    
    result_wrp = NiftiWrapper.from_sequence(src_wrps, args.dimension)
    
    if args.clear_slices:
        result_wrp.meta_ext.clear_slice_meta()
        
    out_name = (args.output[0] % 
                result_wrp.meta_ext.get_class_dict(('global', 'const')))
    result_wrp.to_filename(out_name)    

def delete_ext(hdr, meta_ext):
    target_idx = None
    for idx, ext in enumerate(hdr.extensions):
        if id(ext) == id(meta_ext):
            target_idx = idx
            break
    del hdr.extensions[target_idx]
    hdr['vox_offset'] = 352
    
def dump(args):
    src_nii = nb.load(args.src_nii[0])
    src_wrp = NiftiWrapper(src_nii, args.make_empty)
    meta_str = src_wrp.meta_ext.to_json()
    args.dest_json.write(meta_str)
    args.dest_json.write('\n')
    
    if args.remove:
        hdr = src_wrp.nii_img.get_header()
        delete_ext(hdr, src_wrp.meta_ext)
        src_wrp.to_filename(args.src_nii[0])
                
def check_overwrite():
    usr_input = ''
    while not usr_input in ('y', 'n'):
        usr_input = raw_input('Existing DcmMeta extension found, overwrite? '
                              '[y/n]').lower()
    return usr_input == 'y'

def embed(args):
    dest_nii = nb.load(args.dest_nii[0])
    hdr = dest_nii.get_header()
    try:
        src_wrp = NiftiWrapper(dest_nii, False)
    except ValueError:
        pass
    else:
        if not args.force_overwrite:
            if not check_overwrite():
                return
        delete_ext(hdr, src_wrp.meta_ext)
    
    hdr.extensions.append(DcmMetaExtension.from_json(args.src_json.read()))
    nb.save(dest_nii, args.dest_nii[0])
    
def lookup(args):
    src_wrp = NiftiWrapper.from_filename(args.src_nii[0])
    index = tuple(int(idx.strip()) for idx in args.index.split(','))
    meta = src_wrp.get_meta(args.key[0], index)
    if not meta is None:
        print meta
    
if __name__ == '__main__':
    sys.exit(main())
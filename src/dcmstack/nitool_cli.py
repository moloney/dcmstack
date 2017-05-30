"""
Command line interface for nitool.

@author: moloney
"""
from __future__ import print_function

import os, sys, argparse
import nibabel as nb
from .dcmmeta import NiftiWrapper, DcmMetaExtension, MissingExtensionError

prog_descrip = """Work with extended Nifti files created by dcmstack"""

def main(argv=sys.argv):
    #Setup the top level parser
    arg_parser = argparse.ArgumentParser(description=prog_descrip)
    sub_parsers = arg_parser.add_subparsers(title="Subcommands")

    #Split command
    split_help = ("Split src_nii file along a dimension. Defaults to the slice "
                  "dimension if 3D, otherwise the last dimension.")
    split_parser = sub_parsers.add_parser('split', help=split_help)
    split_parser.add_argument('src_nii', nargs=1)
    split_parser.add_argument('-d', '--dimension', default=None, type=int,
                              help=("The dimension to split along. Must be in "
                              "the range [0, 5)"))
    split_parser.add_argument('-o', '--output-format', default=None,
                              help=("Format string used to create the output "
                              "file names. Default is to prepend the index "
                              "number to the src_nii filename."))
    split_parser.set_defaults(func=split)

    #Merge Command
    merge_help = ("Merge the provided Nifti files along a dimension. Defaults "
                  "to slice, then time, and then vector.")
    merge_parser = sub_parsers.add_parser('merge', help=merge_help)
    merge_parser.add_argument('output', nargs=1)
    merge_parser.add_argument('src_niis', nargs='+')
    merge_parser.add_argument('-d', '--dimension', default=None, type=int,
                              help=("The dimension to merge along. Must be "
                              "in the range [0, 5)"))
    merge_parser.add_argument('-s', '--sort', default=None,
                              help=("Sort the source files using the provided "
                              "meta data key before merging"))
    merge_parser.add_argument('-c', '--clear-slices', action='store_true',
                              help="Clear all per slice meta data")
    merge_parser.set_defaults(func=merge)

    #Dump Command
    dump_help = "Dump the JSON meta data extension from the provided Nifti."
    dump_parser = sub_parsers.add_parser('dump', help=dump_help)
    dump_parser.add_argument('src_nii', nargs=1)
    dump_parser.add_argument('dest_json', nargs='?',
                             type=argparse.FileType('w'),
                             default=sys.stdout)
    dump_parser.add_argument('-m', '--make-empty', default=False,
                             action='store_true',
                             help="Make an empty extension if none exists")
    dump_parser.add_argument('-r', '--remove', default=False,
                             action='store_true',
                             help="Remove the extension from the Nifti file")
    dump_parser.set_defaults(func=dump)

    #Embed Command
    embed_help = "Embed a JSON extension into the Nifti file."
    embed_parser = sub_parsers.add_parser('embed', help=embed_help)
    embed_parser.add_argument('src_json', nargs='?', type=argparse.FileType('r'),
                              default=sys.stdin)
    embed_parser.add_argument('dest_nii', nargs=1)
    embed_parser.add_argument('-f', '--force-overwrite', action='store_true',
                              help="Overwrite any existing dcmmeta extension")
    embed_parser.set_defaults(func=embed)

    #Lookup command
    lookup_help = "Lookup the value for the given meta data key."
    lookup_parser = sub_parsers.add_parser('lookup', help=lookup_help)
    lookup_parser.add_argument('key', nargs=1)
    lookup_parser.add_argument('src_nii', nargs=1)
    lookup_parser.add_argument('-i', '--index',
                               help=("Use the given voxel index. The index "
                               "must be provided as a comma separated list of "
                               "integers (one for each dimension)."))
    lookup_parser.set_defaults(func=lookup)

    #Inject command
    inject_help = "Inject meta data into the JSON extension."
    inject_parser = sub_parsers.add_parser('inject', help=inject_help)
    inject_parser.add_argument('dest_nii', nargs=1)
    inject_parser.add_argument('classification', nargs=2)
    inject_parser.add_argument('key', nargs=1)
    inject_parser.add_argument('values', nargs='+')
    inject_parser.add_argument('-f', '--force-overwrite',
                               action='store_true',
                               help=("Overwrite any existing values "
                               "for the key"))
    inject_parser.add_argument('-t', '--type', default=None,
                               help="Interpret the value as this type instead "
                               "of trying to determine the type automatically")
    inject_parser.set_defaults(func=inject)

    #Parse the arguments and call the appropriate function
    args = arg_parser.parse_args(argv[1:])
    return args.func(args)

def split(args):
    src_path = args.src_nii[0]
    src_fn = os.path.basename(src_path)
    src_dir = os.path.dirname(src_path)

    src_nii = nb.load(src_path)
    try:
        src_wrp = NiftiWrapper(src_nii)
    except MissingExtensionError:
        print("No dcmmeta extension found, making empty one...")
        src_wrp = NiftiWrapper(src_nii, make_empty=True)
    for split_idx, split in enumerate(src_wrp.split(args.dimension)):
        if args.output_format:
            out_name = (args.output_format %
                        split.meta_ext.get_class_dict(('global', 'const'))
                       )
        else:
            out_name = os.path.join(src_dir, '%03d-%s' % (split_idx, src_fn))
        nb.save(split, out_name)
    return 0

def make_key_func(meta_key, index=None):
    def key_func(src_nii):
        result = src_nii.get_meta(meta_key, index)
        if result is None:
            raise ValueError('Key not found: %s' ) % meta_key
        return result

    return key_func

def merge(args):
    src_wrps = []
    for src_path in args.src_niis:
        src_nii = nb.load(src_path)
        try:
            src_wrp = NiftiWrapper(src_nii)
        except MissingExtensionError:
            print("No dcmmeta extension found, making empty one...")
            src_wrp = NiftiWrapper(src_nii, make_empty=True)
        src_wrps.append(src_wrp)

    if args.sort:
        src_wrps.sort(key=make_key_func(args.sort))

    result_wrp = NiftiWrapper.from_sequence(src_wrps, args.dimension)

    if args.clear_slices:
        result_wrp.meta_ext.clear_slice_meta()

    out_name = (args.output[0] %
                result_wrp.meta_ext.get_class_dict(('global', 'const')))
    result_wrp.to_filename(out_name)
    return 0

def dump(args):
    src_nii = nb.load(args.src_nii[0])
    src_wrp = NiftiWrapper(src_nii, args.make_empty)
    meta_str = src_wrp.meta_ext.to_json()
    args.dest_json.write(meta_str)
    args.dest_json.write('\n')

    if args.remove:
        src_wrp.remove_extension()
        src_wrp.to_filename(args.src_nii[0])
    return 0

def check_overwrite():
    usr_input = ''
    while not usr_input in ('y', 'n'):
        usr_input = input('Existing DcmMeta extension found, overwrite? '
                              '[y/n]').lower()
    return usr_input == 'y'

def embed(args):
    dest_nii = nb.load(args.dest_nii[0])
    hdr = dest_nii.get_header()
    try:
        src_wrp = NiftiWrapper(dest_nii, False)
    except MissingExtensionError:
        pass
    else:
        if not args.force_overwrite:
            if not check_overwrite():
                return
        src_wrp.remove_extension()

    hdr.extensions.append(DcmMetaExtension.from_json(args.src_json.read()))
    nb.save(dest_nii, args.dest_nii[0])
    return 0

def lookup(args):
    src_wrp = NiftiWrapper.from_filename(args.src_nii[0])
    index = None
    if args.index:
        index = tuple(int(idx.strip()) for idx in args.index.split(','))
    meta = src_wrp.get_meta(args.key[0], index)
    if not meta is None:
        print(meta)
    return 0

def convert_values(values, type_str=None):
    if type_str is None:
        for conv_type in (int, float):
            try:
                values = [conv_type(val) for val in values]
            except ValueError:
                pass
            else:
                break
    else:
        if type_str not in ('str', 'int', 'float'):
            raise ValueError("Unrecognized type: %s" % type_str)
        conv_type = eval(type_str)
        values = [conv_type(val) for val in values]
    if len(values) == 1:
        return values[0]
    return values

def inject(args):
    dest_nii = nb.load(args.dest_nii[0])
    dest_wrp = NiftiWrapper(dest_nii, make_empty=True)
    classification = tuple(args.classification)
    if not classification in dest_wrp.meta_ext.get_valid_classes():
        print("Invalid classification: %s" % (classification,))
        return 1
    n_vals = len(args.values)
    mult = dest_wrp.meta_ext.get_multiplicity(classification)
    if n_vals != mult:
        print(("Invalid number of values for classification. Expected "
               "%d but got %d") % (mult, n_vals))
        return 1
    key = args.key[0]
    if key in dest_wrp.meta_ext.get_keys():
        if not args.force_overwrite:
            print("Key already exists, must pass --force-overwrite")
            return 1
        else:
            curr_class = dest_wrp.meta_ext.get_classification(key)
            curr_dict = dest_wrp.meta_ext.get_class_dict(curr_class)
            del curr_dict[key]
    class_dict = dest_wrp.meta_ext.get_class_dict(classification)
    class_dict[key] = convert_values(args.values, args.type)
    nb.save(dest_nii, args.dest_nii[0])
    return 0

if __name__ == '__main__':
    sys.exit(main())

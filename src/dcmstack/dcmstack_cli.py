"""
Command line interface to dcmstack.

@author: moloney
"""
from __future__ import print_function

import os, sys, argparse, string
from glob import glob
import dicom
from . import dcmstack
from .dcmstack import (parse_and_group, stack_group, DicomOrdering,
                       default_group_keys)
from .dcmmeta import NiftiWrapper
from . import extract
from .info import __version__


prog_descrip = """Stack DICOM files from each source directory into 2D to 5D
volumes, optionally extracting meta data.
"""

prog_epilog = """IT IS YOUR RESPONSIBILITY TO KNOW IF THERE IS PRIVATE HEALTH
INFORMATION IN THE METADATA EXTRACTED BY THIS PROGRAM."""

def parse_tags(opt_str):
    tag_strs = opt_str.split(',')
    tags = []
    for tag_str in tag_strs:
        tokens = tag_str.split('_')
        if len(tokens) != 2:
            raise ValueError('Invalid str format for tags')
        tags.append(dicom.tag.Tag(int(tokens[0].strip(), 16),
                                  int(tokens[1].strip(), 16))
                   )
    return tags

def sanitize_path_comp(path_comp):
    result = []
    for char in path_comp:
        if not char in string.letters + string.digits + '-_.':
            result.append('_')
        else:
            result.append(char)
    return ''.join(result)

def main(argv=sys.argv):
    #Handle command line options
    arg_parser = argparse.ArgumentParser(description=prog_descrip,
                                         epilog=prog_epilog)
    arg_parser.add_argument('src_dirs', nargs='*', help=('The source '
                            'directories containing DICOM files.'))

    input_opt = arg_parser.add_argument_group('Input options')
    input_opt.add_argument('--force-read', action='store_true', default=False,
                           help=('Try reading all files as DICOM, even if they '
                           'are missing the preamble.'))
    input_opt.add_argument('--file-ext', default='.dcm', help=('Only try reading '
                           'files with the given extension. Default: '
                           '%(default)s'))
    input_opt.add_argument('--allow-dummies', action='store_true', default=False,
                           help=('Allow DICOM files that are missing pixel '
                           'data, filling that slice of the output nifti with '
                           'the maximum representable value.'))

    output_opt = arg_parser.add_argument_group('Output options')
    output_opt.add_argument('--dest-dir', default=None,
                            help=('Destination directory, defaults to the '
                            'source directory.'))
    output_opt.add_argument('-o', '--output-name', default=None,
                            help=('Python format string determining the output '
                            'filenames based on DICOM tags.'))
    output_opt.add_argument('--output-ext', default='.nii.gz',
                            help=('The extension for the output file type. '
                            'Default: %(default)s'))
    output_opt.add_argument('-d', '--dump-meta', default=False,
                            action='store_true', help=('Dump the extracted '
                            'meta data into a JSON file with the same base '
                            'name as the generated Nifti'))
    output_opt.add_argument('--embed-meta', default=False, action='store_true',
                            help=('Embed the extracted meta data into a Nifti '
                            'header extension (in JSON format).'))

    stack_opt = arg_parser.add_argument_group('Stacking Options')
    stack_opt.add_argument('-g', '--group-by', default=None,
                           help=("Comma separated list of meta data keys to "
                           "group input files into stacks with."))
    stack_opt.add_argument('--voxel-order', default='LAS',
                           help=('Order the voxels so the spatial indices '
                           'start from these directions in patient space. '
                           'The directions in patient space should be given '
                           'as a three character code: (l)eft, (r)ight, '
                           '(a)nterior, (p)osterior, (s)uperior, (i)nferior. '
                           'Passing an empty string will disable '
                           'reorientation. '
                           'Default: %(default)s'))
    stack_opt.add_argument('-t', '--time-var', default=None,
                           help=('The DICOM element keyword to use for '
                           'ordering the stack along the time dimension.'))
    stack_opt.add_argument('--vector-var', default=None,
                           help=('The DICOM element keyword to use for '
                           'ordering the stack along the vector dimension.'))
    stack_opt.add_argument('--time-order', default=None,
                           help=('Provide a text file with the desired order '
                           'for the values (one per line) of the attribute '
                           'used as the time variable. This option is rarely '
                           'needed.'))
    stack_opt.add_argument('--vector-order', default=None,
                           help=('Provide a text file with the desired order '
                           'for the values (one per line) of the attribute '
                           'used as the vector variable. This option is rarely '
                           'needed.'))

    meta_opt = arg_parser.add_argument_group('Meta Extraction and Filtering '
                                             'Options')
    meta_opt.add_argument('-l', '--list-translators', default=False,
                          action='store_true', help=('List enabled translators '
                          'and exit'))
    meta_opt.add_argument('--disable-translator', default=None,
                          help=('Disable the translators for the provided '
                          'tags. Tags should be given in the format '
                          '"0x0_0x0". More than one can be given in a comma '
                          'separated list. If the word "all" is provided, all '
                          'translators will be disabled.'))
    meta_opt.add_argument('--extract-private', default=False,
                          action='store_true',
                          help=('Extract meta data from private elements, even '
                          'if there is no translator. If the value for the '
                          'element contains non-ascii bytes it will still be '
                          'ignored. The extracted meta data may still be '
                          'filtered out by the regular expressions.'))
    meta_opt.add_argument('-i', '--include-regex', action='append',
                          help=('Include any meta data where the key matches '
                          'the provided regular expression. This will override '
                          'any exclude expressions. Applies to all meta data.'))
    meta_opt.add_argument('-e', '--exclude-regex', action='append',
                          help=('Exclude any meta data where the key matches '
                          'the provided regular expression. This will '
                          'supplement the default exclude expressions. Applies '
                          'to all meta data.'))
    meta_opt.add_argument('--default-regexes', default=False,
                          action='store_true',
                          help=('Print the list of default include and exclude '
                          'regular expressions and exit.'))

    gen_opt = arg_parser.add_argument_group('General Options')
    gen_opt.add_argument('-v', '--verbose',  default=False, action='store_true',
                         help=('Print additional information.'))
    gen_opt.add_argument('--strict', default=False, action='store_true',
                         help=('Fail on the first exception instead of '
                         'showing a warning.'))
    gen_opt.add_argument('--version', default=False, action='store_true',
                         help=('Show the version and exit.'))

    args = arg_parser.parse_args(argv[1:])

    if args.version:
        print(__version__)
        return 0

    #Check if we are just listing the translators
    if args.list_translators:
        for translator in extract.default_translators:
            print('%s -> %s' % (translator.tag, translator.name))
        return 0

    #Check if we are just listing the default exclude regular expressions
    if args.default_regexes:
        print('Default exclude regular expressions:')
        for regex in dcmstack.default_key_excl_res:
            print('\t' + regex)
        print('Default include regular expressions:')
        for regex in dcmstack.default_key_incl_res:
            print('\t' + regex)
        return 0

    #Check if we are generating meta data
    gen_meta = args.embed_meta or args.dump_meta

    if gen_meta:
        #Start with the module defaults
        ignore_rules = extract.default_ignore_rules
        translators = extract.default_translators

        #Disable translators if requested
        if args.disable_translator:
            if args.disable_translator.lower() == 'all':
                translators = tuple()
            else:
                try:
                    disable_tags = parse_tags(args.disable_translator)
                except:
                    arg_parser.error('Invalid tag format to --disable-translator.')
                new_translators = []
                for translator in translators:
                    if not translator.tag in disable_tags:
                        new_translators.append(translator)
                translators = new_translators

        #Include non-translated private elements if requested
        if args.extract_private:
            ignore_rules = (extract.ignore_pixel_data,
                            extract.ignore_overlay_data,
                            extract.ignore_color_lut_data)

        extractor = extract.MetaExtractor(ignore_rules, translators)

    else:
        extractor = extract.minimal_extractor

    #Add include/exclude regexes to meta filter
    include_regexes = dcmstack.default_key_incl_res
    if args.include_regex:
        include_regexes += args.include_regex
    exclude_regexes = dcmstack.default_key_excl_res
    if args.exclude_regex:
        exclude_regexes += args.exclude_regex
    meta_filter = dcmstack.make_key_regex_filter(exclude_regexes,
                                                 include_regexes)

    #Figure out time and vector ordering
    if args.time_var:
        if args.time_order:
            order_file = open(args.time_order)
            abs_order = [line.strip() for line in order_file.readlines()]
            order_file.close()
            time_order = DicomOrdering(args.time_var, abs_order, True)
        else:
            time_order = DicomOrdering(args.time_var)
    else:
        time_order = None
    if args.vector_var:
        if args.vector_order:
            order_file = open(args.vector_order)
            abs_order = [line.strip() for line in order_file.readlines()]
            order_file.close()
            vector_order = DicomOrdering(args.vector_var, abs_order, True)
        else:
            vector_order = DicomOrdering(args.vector_var)
    else:
        vector_order = None

    if len(args.src_dirs) == 0:
        arg_parser.error('No source directories were provided.')

    #Handle group-by option
    if not args.group_by is None:
        group_by = args.group_by.split(',')
    else:
        group_by = default_group_keys

    #Handle each source directory individually
    for src_dir in args.src_dirs:
        if not os.path.isdir(src_dir):
            print('%s is not a directory, skipping' % src_dir, file=sys.stderr)

        if args.verbose:
            print("Processing source directory %s" % src_dir)

        #Build a list of paths to source files
        glob_str = os.path.join(src_dir, '*')
        if args.file_ext:
            glob_str += args.file_ext
        src_paths = glob(glob_str)

        if args.verbose:
            print("Found %d source files in the directory" % len(src_paths))

        #Group the files in this directory
        groups = parse_and_group(src_paths,
                                 group_by,
                                 extractor,
                                 args.force_read,
                                 not args.strict,
                                )

        if args.verbose:
            print("Found %d groups of DICOM images" % len(groups))

        if len(groups) == 0:
            print("No DICOM files found in %s" % src_dir)

        out_idx = 0
        generated_outs = set()
        for key, group in groups.iteritems():
            stack = stack_group(group,
                                warn_on_except=not args.strict,
                                time_order=time_order,
                                vector_order=vector_order,
                                allow_dummies=args.allow_dummies,
                                meta_filter=meta_filter)
            meta = group[0][1]

            #Build an appropriate output format string if none was specified
            if args.output_name is None:
                out_fmt = []
                if 'SeriesNumber' in meta:
                    out_fmt.append('%(SeriesNumber)03d')
                if 'ProtocolName' in meta:
                    out_fmt.append('%(ProtocolName)s')
                elif 'SeriesDescription' in meta:
                    out_fmt.append('%(SeriesDescription)s')
                else:
                    out_fmt.append('series')
                out_fmt = '-'.join(out_fmt)
            else:
                out_fmt = args.output_name

            #Get the output filename from the format string, make sure the
            #result is unique for this source directory
            out_fn = sanitize_path_comp(out_fmt % meta)
            if out_fn in generated_outs:
                out_fn += '-%03d' % out_idx
            generated_outs.add(out_fn)
            out_idx += 1
            out_fn = out_fn + args.output_ext

            if args.dest_dir:
                out_path = os.path.join(args.dest_dir, out_fn)
            else:
                out_path = os.path.join(src_dir, out_fn)

            if args.verbose:
                print("Writing out stack to path %s" % out_path)

            nii = stack.to_nifti(args.voxel_order, gen_meta)

            if args.dump_meta:
                nii_wrp = NiftiWrapper(nii)
                path_tokens = out_path.split('.')
                if path_tokens[-1] == 'gz':
                    path_tokens = path_tokens[:-1]
                if path_tokens[-1] == 'nii':
                    path_tokens = path_tokens[:-1]
                meta_path = '.'.join(path_tokens + ['json'])
                out_file = open(meta_path, 'w')
                out_file.write(nii_wrp.meta_ext.to_json())
                out_file.close()

                if not args.embed_meta:
                    nii_wrp.remove_extension()

                del nii_wrp

            nii.to_filename(out_path)

            del key
            del group
            del stack
            del meta
            del nii
        del groups

    return 0

if __name__ == '__main__':
    sys.exit(main())

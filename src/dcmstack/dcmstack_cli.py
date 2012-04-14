"""
Command line interface to dcmstack.

@author: moloney
"""
import os, sys, argparse, string
from glob import glob
import dicom
from . import dcmstack
from .dcmstack import parse_and_stack, DicomOrdering
from .dcmmeta import NiftiWrapper
from . import extract


prog_descrip = """Stack DICOM files from each source directory into 2D to 5D 
volumes, optionally extracting meta data.

Arguments accepting DICOM tags should be in the format '0x0_0x0'. More than 
one tag can be given in a comma seperated list.
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
    output_opt.add_argument('-o', '--output-name', 
                            default='%(SeriesNumber)03d-%(ProtocolName)s',
                            help=('Python format string determining the output '
                            'filenames based on DICOM tags. Files mapping to '
                            'the same filename will be put in the same stack. '
                            'Default: %(default)s'))
    output_opt.add_argument('--opt-suffix', default='TE_%(EchoTime).3f', 
                            help=("Python format string determining optional "
                            "suffix for the output name. Only used if it "
                            "varies between inputs. Default: %(default)s"))
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
    stack_opt.add_argument('--voxel-order', default='rpi', 
                           help=('Order the voxels so the spatial indices '
                           'start from these directions in patient space. '
                           'The directions in patient space should be given '
                           'as a three character code: (l)eft, (r)ight, '
                           '(a)nterior, (p)osterior, (s)uperior, (i)nferior. '
                           'Default: %(default)s'))
    stack_opt.add_argument('-t', '--time-var', default='AcquisitionTime',
                           help=('The DICOM tag to use for ordering the stack '
                           'along the time dimension. Default: %(default)s'))
    stack_opt.add_argument('--vector-var', default=None,
                           help=('The DICOM tag to use for ordering the stack '
                           'along the vector dimension.'))
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
                          'tags. If the word "all" is provided, all '
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
    
    args = arg_parser.parse_args(argv[1:])
    
    #Start with the module defaults
    ignore_rules = extract.default_ignore_rules
    translators = extract.default_translators
    
    #Check if we are just listing the translators
    if args.list_translators:
        for translator in translators:
            print '%s -> %s' % (translator.tag, translator.name)
        return 0
    
    #Check if we are just listing the default exclude regular expressions
    if args.default_regexes:
        print 'Default exclude regular expressions:'
        for regex in dcmstack.default_key_excl_res:
            print '\t' + regex
        print 'Default include regular expressions:'
        for regex in dcmstack.default_key_incl_res:
            print '\t' + regex
        return 0
    
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
        ignore_rules = [extract.ignore_non_ascii_bytes]
    
    extractor = extract.MetaExtractor(ignore_rules, translators)
    
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
        
    #Handle each source directory individually
    for src_dir in args.src_dirs:
        if not os.path.isdir(src_dir):
            print >> sys.stderr, '%s is not a directory, skipping' % src_dir

        if args.verbose:
            print "Processing source directory %s" % src_dir

        #Build a list of paths to source files
        glob_str = os.path.join(src_dir, '*')
        if args.file_ext:
            glob_str += args.file_ext
        src_paths = glob(glob_str)
        
        if args.verbose:
            print "Found %d source files in the directory" % len(src_paths)
        
        #Build the stacks for this directory
        stacks = parse_and_stack(src_paths, args.output_name, args.opt_suffix,
                                 time_order, vector_order, args.allow_dummies, 
                                 extractor, meta_filter, args.force_read, False)
        
        if args.verbose:
            print "Created %d stacks of DICOM images" % len(stacks)
        
        #Write out the stacks
        for out_fn, stack in stacks.iteritems():
            out_fn = sanitize_path_comp(out_fn) + args.output_ext
            if args.dest_dir:
                out_path = os.path.join(args.dest_dir, out_fn)
            else:
                out_path = os.path.join(src_dir, out_fn)

            if args.verbose:
                print "Writing out stack to path %s" % out_path

            nii = stack.to_nifti(args.voxel_order, 
                                 args.embed_meta or args.dump_meta)
            
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
                                 
            nii.to_filename(out_path)
                
    return 0

if __name__ == '__main__':
    sys.exit(main())
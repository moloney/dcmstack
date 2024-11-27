"""Extract meta data from a DICOM data set into human-readable and JSON compatible dict

We don't do any normalization at this level beyond converting decimal strings to float 
and making sure strings are unicode. The one exception is translating private 
mini-headers embedded in certain elements.
"""
import struct
import warnings
from collections import namedtuple
from typing import Dict, List

import pydicom
from pydicom.tag import BaseTag
from pydicom.dataset import PrivateBlock
from pydicom.datadict import keyword_for_tag, private_dictionaries
from pydicom.charset import decode_element
from nibabel.nicom import csareader
try:
    import chardet
    have_chardet = True
except ImportError:
    have_chardet = False
    pass

from .dcmstack import DicomStack
from .utils import PY2, byte_str


def ignore_private(tag, name, ds):
    '''Ignore rule for `MetaExtractor` to skip private DICOM elements (odd
    group number).'''
    if tag.group % 2 == 1:
        return True
    return False


def ignore_unknown_private(tag, name, ds):
    "Ignore private elements that don't have a name in pydicom private_dictionaries"
    return tag.group % 2 == 1 and name.split(".")[1].startswith("0X")


def ignore_pixel_data(tag, name, ds):
    return tag == pydicom.tag.Tag(0x7fe0, 0x10)


def ignore_overlay_data(tag, name, ds):
    return tag.group & 0xff00 == 0x6000 and tag.elem == 0x3000


def ignore_color_lut_data(tag, name, ds):
    return (
        tag.group == 0x28 and tag.elem in (0x1201, 0x1202, 0x1203, 0x1221, 0x1222, 0x1223)
    )


default_ignore_rules = (ignore_private,
                        ignore_pixel_data,
                        ignore_overlay_data,
                        ignore_color_lut_data)
'''The default tuple of ignore rules for `MetaExtractor`.'''


Translator = namedtuple('Translator', ['name',
                                       'tag',
                                       'priv_creator',
                                       'trans_func']
                       )
'''A namedtuple for storing the four elements of a translator: a name, the
pydicom.tag.Tag that can be translated, the private creator string (optional), and
the function which takes the DICOM element and returns a dictionary.'''


def simplify_csa_dict(csa_dict):
    '''Simplify the result of nibabel.nicom.csareader.

    Parameters
    ----------
    csa_dict : dict
        The result from nibabel.nicom.csareader

    Returns
    -------
    result : dict
        Result where the keys come from the 'tags' sub dictionary of `csa_dict`.
        The values come from the 'items' within that tags sub sub dictionary.
        If items has only one element it will be unpacked from the list.
    '''
    if csa_dict is None:
        return None

    result = {}
    for tag in sorted(csa_dict['tags']):
        items = []
        for item in csa_dict['tags'][tag]['items']:
            if isinstance(item, byte_str):
                item = get_text(item)
            items.append(item)
        if len(items) == 0:
            continue
        elif len(items) == 1:
            result[tag] = items[0]
        else:
            result[tag] = items
    return result


def csa_image_trans_func(elem):
    '''Function for translating the CSA image sub header.'''
    return simplify_csa_dict(csareader.read(elem.value))


csa_image_trans = Translator('CsaImage',
                             pydicom.tag.Tag(0x29, 0x1010),
                             'SIEMENS CSA HEADER',
                             csa_image_trans_func)
'''Translator for the CSA image sub header.'''


class PhoenixParseError(Exception):
    def __init__(self, line):
        '''Exception indicating a error parsing a line from the Phoenix
        Protocol.
        '''
        self.line = line

    def __str__(self):
        return 'Unable to parse phoenix protocol line: %s' % self.line


def _parse_phoenix_line(line, str_delim='""'):
    delim_len = len(str_delim)
    #Handle most comments (not always when string literal involved)
    comment_idx = line.find('#')
    if comment_idx != -1:
        #Check if the pound sign is in a string literal
        if line[:comment_idx].count(str_delim) == 1:
            if line[comment_idx:].find(str_delim) == -1:
                raise PhoenixParseError(line)
        else:
            line = line[:comment_idx]

    #Allow empty lines
    if line.strip() == '':
        return None

    #Find the first equals sign and use that to split key/value
    equals_idx = line.find('=')
    if equals_idx == -1:
        raise PhoenixParseError(line)
    key = line[:equals_idx].strip()
    val_str = line[equals_idx + 1:].strip()

    #If there is a string literal, pull that out
    if val_str.startswith(str_delim):
        end_quote = val_str[delim_len:].find(str_delim) + delim_len
        if end_quote == -1:
            raise PhoenixParseError(line)
        elif not end_quote == len(val_str) - delim_len:
            #Make sure remainder is just comment
            if not val_str[end_quote+delim_len:].strip().startswith('#'):
                raise PhoenixParseError(line)

        return (key, val_str[2:end_quote])

    else: #Otherwise try to convert to an int or float
        val = None
        try:
            val = int(val_str)
        except ValueError:
            pass
        else:
            return (key, val)

        try:
            val = int(val_str, 16)
        except ValueError:
            pass
        else:
            return (key, val)

        try:
            val = float(val_str)
        except ValueError:
            pass
        else:
            return (key, val)

    raise PhoenixParseError(line)


def parse_phoenix_prot(prot_key, prot_val):
    '''Parse the MrPheonixProtocol string.

    Parameters
    ----------
    prot_str : str
        The 'MrPheonixProtocol' string from the CSA Series sub header.

    Returns
    -------
    prot_dict : dict
        Meta data pulled from the ASCCONV section.

    Raises
    ------
    PhoenixParseError : A line of the ASCCONV section could not be parsed.
    '''
    if prot_key == 'MrPhoenixProtocol':
        str_delim = '""'
    elif prot_key == 'MrProtocol':
        str_delim = '"'
    else:
        raise ValueError('Unknown protocol key: %s' % prot_key)
    ascconv_start = prot_val.find('### ASCCONV BEGIN ')
    ascconv_end = prot_val.find('### ASCCONV END ###')
    ascconv = prot_val[ascconv_start:ascconv_end].split('\n')[1:-1]

    result = {}
    for line in ascconv:
        parse_result = _parse_phoenix_line(line, str_delim)
        if parse_result:
            result[parse_result[0]] = parse_result[1]

    return result


def csa_series_trans_func(elem):
    '''Function for parsing the CSA series sub header.'''
    csa_dict = simplify_csa_dict(csareader.read(elem.value))

    #If there is a phoenix protocol, parse it and dump it into the csa_dict
    phx_src = None
    if 'MrPhoenixProtocol' in csa_dict:
        phx_src = 'MrPhoenixProtocol'
    elif 'MrProtocol' in csa_dict:
        phx_src = 'MrProtocol'

    if not phx_src is None:
        phoenix_dict = parse_phoenix_prot(phx_src, csa_dict[phx_src])
        del csa_dict[phx_src]
        for key, val in phoenix_dict.items():
            new_key = '%s.%s' % ('MrPhoenixProtocol', key)
            csa_dict[new_key] = val

    return csa_dict


csa_series_trans = Translator('CsaSeries',
                              pydicom.tag.Tag(0x29, 0x1020),
                              'SIEMENS CSA HEADER',
                              csa_series_trans_func)
'''Translator for parsing the CSA series sub header.'''


default_translators = (csa_image_trans,
                       csa_series_trans,
                      )
'''Default translators for MetaExtractor.'''


def tag_to_str(tag):
    '''Convert a DICOM tag to a string representation using the group and
    element hex values seprated by an underscore.'''
    return '%#X_%#X' % (tag.group, tag.elem)


unpack_vr_map = {'SL' : 'i',
                 'UL' : 'I',
                 'SV' : 'l',
                 'UV' : 'L',
                 'FL' : 'f',
                 'FD' : 'd',
                 'SS' : 'h',
                 'US' : 'H',
                 'US or SS' : 'H',
                 }
'''Dictionary mapping value representations to corresponding format strings for
the struct.unpack function.'''
    

class TextConverter:
    """Converter for byte strings tries to convert to text or returns None"""
    def prep_dataset(self, ds: pydicom.Dataset):
        self.charset = ds._character_set

    def __call__(self, val):
        for encoding in self.charset:
            try:
                return val.decode(encoding)
            except:
                pass


_get_text = TextConverter()


default_conversions = {'DS' : float,
                       'IS' : int,
                       'AT' : tag_to_str,
                       'OW' : _get_text,
                       'OB' : _get_text,
                       'OW or OB' : _get_text,
                       'OB or OW' : _get_text,
                       'UN' : _get_text,
                      }


class MetaExtractor(object):
    '''Callable object for extracting meta data from a dicom dataset

    Parameters
    ----------
    ignore_rules : sequence
        A sequence of callables each of which take an element tag and name, plus a 
        reference to the pydicom.Dataset, and return True if it should be ignored. If 
        None the module default is used.

    translators : sequence
        A sequence of `Translator` objects each of which can convert a
        DICOM element into a dictionary. Overrides any ignore rules. If
        None the module default is used.

    conversions : dict
        Mapping of DICOM value representation (VR) strings to callables
        that perform some conversion on the value

    warn_on_trans_except : bool
        Convert any exceptions from translators into warnings.
    '''

    def __init__(self, ignore_rules=None, translators=None, conversions=None,
                 warn_on_trans_except=True):
        if ignore_rules is None:
            self.ignore_rules = default_ignore_rules
        else:
            self.ignore_rules = ignore_rules
        if translators is None:
            self.translators = default_translators
        else:
            self.translators = translators
        if conversions is None:
            self.conversions = default_conversions
        else:
            self.conversions = conversions
        self._needs_prep_converters = set(
            c for c in self.conversions.values() if hasattr(c, "prep_dataset")
        )
        self.warn_on_trans_except = warn_on_trans_except

    def _get_priv_name(self, tag: BaseTag, pblocks: List[PrivateBlock]):
        '''Get the name to use for any private elements'''
        elem = creator = "Unknown"
        for pblock in pblocks:
            if pblock.block_start == tag.elem & 0xFF00:
                creator = pblock.private_creator
                priv_dict = private_dictionaries.get(creator)
                if priv_dict is None:
                    break
                key = next(iter(priv_dict))
                key = f"{key[:-2]}{tag.elem & 0xFF:02x}"
                priv_info = priv_dict.get(key)
                if priv_info is None:
                    break
                elem = priv_info[2]
                break
        if elem == "Unknown":
            elem = tag_to_str(tag)
        else:
            elem = ''.join([t[0].upper() + t[1:] for t in elem.split()])
        return f"{creator.upper().replace(' ', '_')}.{elem}"

    def _get_elem_value(self, elem):
        '''Get the value for any non-translated elements'''
        # If the VR is implicit, we may need to unpack the values from a byte
        # string. This may require us to make an assumption about whether the
        # value is signed or not, but this is unavoidable.
        unpacked = False
        if elem.VR in unpack_vr_map and isinstance(elem.value, str):
            n_vals = len(elem.value) / struct.calcsize(unpack_vr_map[elem.VR])
            if n_vals != elem.VM:
                warnings.warn("The element's VM and the number of values do "
                              "not match.")
            if n_vals == 1:
                value = struct.unpack(unpack_vr_map[elem.VR], elem.value)[0]
            else:
                value = list(struct.unpack(unpack_vr_map[elem.VR]*n_vals,
                                           elem.value)
                            )
            unpacked = True
        else:
            value = elem.value
        n_vals = elem.VM
        # Handle any conversions
        if elem.VR in self.conversions and value is not None:
            if n_vals == 1:
                value = self.conversions[elem.VR](value)
            else:
                value = [self.conversions[elem.VR](val) for val in value]
        elif n_vals > 1 and not unpacked:
            # Make sure we don't share refences to lists in oringial dataset
            value = value[:]
        return value

    def __call__(self, dcm: pydicom.Dataset):
        '''Extract the meta data from a DICOM dataset.

        Parameters
        ----------
        dcm : pydicom.dataset.Dataset
            The DICOM dataset to extract the meta data from.

        Returns
        -------
        meta : dict
            A dictionary of extracted meta data.

        Notes
        -----
        Non-private tags use the DICOM keywords as keys. Translators have their
        name, followed by a dot, prepended to the keys of any meta elements
        they produce. Values are unchanged, except when the value
        representation is 'DS' or 'IS' (decimal/integer strings) they are
        converted to float and int types.
        '''
        # Some converters need to see full dataset before converting elements
        for conv in self._needs_prep_converters:
            conv.prep_dataset(dcm)
        # Get all the tags included in this dataset
        tags = dcm.keys()
        # Find private blocks in the dataset
        priv_groups = set(t.group for t in tags if t.group % 2 == 1)
        priv_blocks = {
            c : dcm.private_block(g, c) 
            for g in priv_groups
            for c in dcm.private_creators(g)
        }
        grp_blocks = {
            g : [b for b in priv_blocks.values() if b.group == g] for g in priv_groups
        }
        # Find private elements needed by translators
        # TODO: Do we want to support translating non-private elements here? Seems like
        #       just doing VR conversions is sufficient for standard meta data
        trans_map = {}
        for translator in self.translators:
            pblock = priv_blocks.get(translator.priv_creator)
            if pblock is not None:
                src_tag = pblock.get_tag(translator.tag.elem & 0xff)
                if src_tag in trans_map:
                    raise ValueError("Duplicate translators found")
                trans_map[src_tag] = translator
        result = {}
        trans_meta = {}
        charset = dcm._character_set
        for tag in tags:
            is_priv = tag.group % 2 == 1
            if is_priv:
                # If there is a translator for this element, use it
                trans = trans_map.get(tag)
                if trans is not None:
                    try:
                        meta = trans.trans_func(dcm.get(tag))
                    except Exception as e:
                        if not self.warn_on_trans_except:
                            raise
                        warnings.warn("Exception from translator %s: %s" %
                                        (trans.name,
                                        repr(str(e))))
                    else:
                        if meta:
                            trans_meta[trans.name] = meta
                    continue
                name = self._get_priv_name(tag, grp_blocks[tag.group])
            else:
                name = keyword_for_tag(tag)
                if name == "":
                    warnings.warn("Your 'pydicom' is too old, standard tag not in datadict")
                    name = tag_to_str(tag)
            # Check if we are ignoring this element
            if any(rule(tag, name, dcm) for rule in self.ignore_rules):
                continue
            # Parse the element, skip empty values
            elem = dcm.get(tag)
            if elem.is_empty:
                continue
            # Handle elements that are sequences with recursion
            if elem.VR == "SQ":
                value = []
                for val in elem.value:
                    value.append(self(val))
                if all(x is None for x in value):
                    continue
                result[name] =  value
            # Otherwise just make sure the value is unpacked
            else:
                # Handle unicode conversion
                decode_element(elem, charset)
                value = self._get_elem_value(elem)
                if value is None:
                    continue
                result[name] =  value
        # Inject translator results
        for trans_name, meta in trans_meta.items():
            for name, value in meta.items():
                result[f"{trans_name}.{name}"] = value
        return result


def make_ignore_except_rule(include):
    """Make rule that ignores everying not in `include`"""
    def ignore_except(tag, name, ds):
        if tag in include:
            return False
        return name not in include
    return ignore_except


minimal_extractor = MetaExtractor((make_ignore_except_rule(DicomStack.minimal_keys),))


default_extractor = MetaExtractor()

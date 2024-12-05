"""Extract meta data from a DICOM data set into human-readable and JSON compatible dict

We don't do any normalization at this level beyond converting decimal strings to float 
and making sure strings are unicode. The one exception is translating private 
mini-headers embedded in certain elements.
"""
import struct, re, warnings, enum
from collections import namedtuple
from typing import Dict, List

import pydicom
from pydicom.tag import BaseTag
from pydicom.dataset import PrivateBlock
from pydicom.datadict import keyword_for_tag, tag_for_keyword, private_dictionaries
from pydicom.charset import decode_element
from csa_header import CsaHeader

from .dcmstack import DicomStack


# Define some common rules for ignoring DICOM elements
def ignore_pixel_data(tag, name, ds):
    return tag.group == 0x7fe0 and tag.elem in (0x8, 0x9, 0x10)


def ignore_overlay_data(tag, name, ds):
    return tag.group & 0xff00 == 0x6000 and tag.elem == 0x3000


def ignore_color_lut_data(tag, name, ds):
    return (
        tag.group == 0x28 and tag.elem in (0x1201, 0x1202, 0x1203, 0x1221, 0x1222, 0x1223)
    )


IGNORE_BINARY_RULES = (ignore_pixel_data,
                       ignore_overlay_data,
                       ignore_color_lut_data)
'''The default tuple of ignore rules for `MetaExtractor`.'''


def ignore_private(tag, name, ds):
    '''Ignore rule for `MetaExtractor` to skip all private DICOM elements'''
    return tag.group % 2 == 1


def make_ignore_unknown_private(allow_creators=None, reject_creators=None):
    """Make custom ignore rule for private elements, by default allowing "known" elements

    If the element has a name in the pydicom private dictionaries, it is considered 
    "known". Unknown elements are filtered by default, unless the "private creator" for
    that block of private elements is in `allow_creators`. Known elements will also be
    ignored if the private creator is in `reject_creators`.
    """
    allow_creators = tuple() if allow_creators is None else allow_creators
    reject_creators = tuple() if reject_creators is None else reject_creators
    if allow_creators:
        allow_re = f"({'|'.join([x for x in allow_creators])})"
    else:
        allow_re = ".^"
    if reject_creators:
        reject_re = f"({'|'.join([x for x in reject_creators])})"
    else:
        reject_re = ".^"
    def ignore_unknown_private(tag, name, ds):
        if tag.group % 2 == 0:
            return False
        toks = name.split(".")
        if re.match(allow_re, toks[0], flags=re.I):
            return False
        elif re.match(reject_re, toks[0], flags=re.I):
            return True
        return toks[1].startswith("0X")
    return ignore_unknown_private


def make_ignore_except_rule(include):
    """Make rule that ignores everying not in `include`"""
    incl_tags = set([tag_for_keyword(x) for x in include if isinstance(x, str)])
    def ignore_except(tag, name, ds):
        return tag not in incl_tags
    return ignore_except


# Define some translators for internal "mini headers" stored in private elements
Translator = namedtuple('Translator', ['name', 'tag', 'priv_creator', 'trans_func'])
'''A namedtuple for storing the four elements of a translator: a name, the
pydicom.tag.Tag that can be translated, the private creator string (optional), and
the function which takes the DICOM element and returns a dictionary.'''


def parse_csa(raw_bytes):
    '''Simplify the result of csa_header.CsaHeader

    Parameters
    ----------
    csa_dict : dict
        The result from csa_header.CsaHeader

    Returns
    -------
    result : dict
        Simpler key -> value mapping
    '''
    csa_dict = CsaHeader(raw_bytes).read()
    if csa_dict is None:
        return None
    result = {}
    for key, elem in csa_dict.items():
        val = elem['value']
        if val is None:
            continue
        result[key] = val
    return result


csa_image_trans = Translator('CsaImage',
                             pydicom.tag.Tag(0x29, 0x1010),
                             'SIEMENS CSA HEADER',
                             parse_csa)
'''Translator for the CSA image sub header.'''


csa_series_trans = Translator('CsaSeries',
                              pydicom.tag.Tag(0x29, 0x1020),
                              'SIEMENS CSA HEADER',
                              parse_csa)
'''Translator for parsing the CSA series sub header.'''


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

    IGNORE_RULES = IGNORE_BINARY_RULES + (make_ignore_unknown_private(),)

    TRANSLATORS = (csa_image_trans, csa_series_trans,)

    CONVERSIONS = {
        'DS' : float,
        'IS' : int,
        'AT' : tag_to_str,
        'OW' : _get_text,
        'OB' : _get_text,
        'OW or OB' : _get_text,
        'OB or OW' : _get_text,
        'UN' : _get_text,
    }

    def __init__(
        self, 
        ignore_rules=None, 
        translators=None, 
        conversions=None,
        warn_on_trans_except=True
    ):
        if ignore_rules is None:
            self.ignore_rules = self.IGNORE_RULES
        else:
            self.ignore_rules = ignore_rules
        if translators is None:
            self.translators = self.TRANSLATORS
        else:
            self.translators = translators
        if conversions is None:
            self.conversions = self.CONVERSIONS
        else:
            self.conversions = conversions
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
            elem = "%#X" % (tag.elem & 0xFF)
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
        for conv in set(self.conversions.values()):
            if hasattr(conv, "prep_dataset"):
                conv.prep_dataset(dcm)
        # Get all the tags included in this dataset
        tags = list(dcm.keys())
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
                # Skip private creator elements
                if tag.elem & 0xFF00 == 0:
                    continue
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
                # Otherwise we construct a name for the private element
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


# Preconfigure a few extractors with increasing levels of meta data extraction
class ExtractionLevel(enum.Enum):
    MINIMAL = "min"
    MODERATE = "mod"
    MORE = "more"
    MAX = "max"


MIN_KEYS, MIN_TRANS_NAMES = DicomStack.get_min_req_meta()
MIN_TRANSLATORS = tuple(t for t in MetaExtractor.TRANSLATORS if t.name in MIN_TRANS_NAMES)


MODERATE_KEYS = MIN_KEYS[:]
MODERATE_KEYS += [
    "Modality",
    "Manufacturer",
    "ManufacturersModelName",
    "SoftwareVersions",
    "StationName",
    "DeviceUID",
    "MagneticFieldStrength",
    "PatientID",
    "PatientsName",
    "PatientsAge",
    "PatientsSex",
    "PatientsSize",
    "PatientsWeight",
    "PatientPosition",
    "PatientSpeciesDescription",
    "PatientsBodyMassIndex",
    "BodyPartExamined",
    "StudyInstanceUID",
    "StudyID",
    "SeriesNumber",
    "StudyDescription",
    "SeriesInstanceUID",
    "ImagedNucleus",
    "ImagingFrequency",
    "SOPInstanceUID",
    "ImageType",
    "AcqusitionNumber",
    "InstanceNumber",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "SequenceName",
    "AngioFlag",
    "NumberOfAverages",
    "NumberOfPhaseEncodingSteps",
    "EchoTrainLength",
    "PercentSampling",
    "PercentPhaseFieldOfView",
    "PixelBandwidth",
    "TriggerTime",
    "ReceiveCoilName",
    "TransmitCoilName",
    "AcquisitionMatrix",
    "InPlanePhaseEncodingDirection",
]


EXTRACTORS = {
    ExtractionLevel.MINIMAL : MetaExtractor((make_ignore_except_rule(MIN_KEYS),), MIN_TRANSLATORS),
    ExtractionLevel.MODERATE : MetaExtractor((make_ignore_except_rule(MODERATE_KEYS),)),
    ExtractionLevel.MORE : MetaExtractor(),
    ExtractionLevel.MAX : MetaExtractor(IGNORE_BINARY_RULES),
}

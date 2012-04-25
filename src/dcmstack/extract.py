"""
Extract meta data from a DICOM data set.
"""
import struct, warnings
from collections import OrderedDict, namedtuple, Counter
import dicom
from dicom.datadict import keyword_for_tag
from nibabel.nicom import csareader
from nibabel.nicom.dicomwrappers import Wrapper

def ignore_private(elem):
    '''Ignore private DICOM elements (odd group number).'''
    if elem.tag.group % 2 == 1:
        return True
    return False
    
def ignore_non_ascii_bytes(elem):
    '''Ignore elements with VR of 'OW', 'OB', or 'UN' if the byte string has
    non ASCII characters.'''
    if elem.VR in ('OW', 'OB', 'OW or OB', 'UN'):
        if not all(' ' <= c <= '~' for c in elem.value):
            return True
    return False

default_ignore_rules = (ignore_private,
                        ignore_non_ascii_bytes)
'''The default tuple of ignore rules for MetaExtractor.'''


Translator = namedtuple('Translator', ['name', 
                                       'tag', 
                                       'priv_creator',
                                       'trans_func']
                       )
'''A namedtuple for storing the four elements of a translator: a name, the 
DICOM tag that can be translated, the private creator string (optional), and the 
function which takes the DICOM element and returns a dictionary.'''

def simplify_csa_dict(csa_dict):
    '''Simplify the result of nibabel.nicom.csareader to a dictionary of key 
    value pairs'''
    if csa_dict is None:
        return None
    
    result = OrderedDict()
    for tag in csa_dict['tags']:
        items = csa_dict['tags'][tag]['items']
        if len(items) == 0:
            continue
        elif len(items) == 1:
            result[tag] = items[0]
        else:
            result[tag] = items
    return result

def csa_image_trans_func(elem):
    return simplify_csa_dict(csareader.read(elem.value))
'''Function for translating the CSA image sub header.'''
    
csa_image_trans = Translator('CsaImage', 
                             dicom.tag.Tag(0x29, 0x1010), 
                             'SIEMENS CSA HEADER',
                             csa_image_trans_func)
'''Translator for the CSA image sub header.'''

def _parse_phoenix_line(line):
    #Handle comments at end of lines
    comment_idx = line.find('#')   
    if comment_idx != -1:
        line = line[:comment_idx]
        
    #Split on the equals sign and strip the tokens
    tokens = [token.strip() for token in line.split('=')]
    if len(tokens) != 2:
        return None
   
    #Convert the value to a string, integer, or float
    key = tokens[0]
    val_str = tokens[1]
    if val_str.startswith('""') and val_str.endswith('""'):
        return (key, val_str.strip('""'))
    else:
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
    
        raise ValueError('Unable to parse phoenix protocol line: %s' % line)

def parse_phoenix_prot(prot_str):
    '''Parse the MrPheonixProtocol string into a dictionary of key value pairs,
    pulled from the ASCCONV section.'''
    ascconv_start = prot_str.find('### ASCCONV BEGIN ###')
    ascconv_end = prot_str.find('### ASCCONV END ###')
    ascconv = prot_str[ascconv_start:ascconv_end].split('\n')[1:-1]
    
    result = OrderedDict()    
    for line in ascconv:
        parse_result = _parse_phoenix_line(line)
        if parse_result:
            result[parse_result[0]] = parse_result[1]
        
    return result

def csa_series_trans_func(elem):
    '''Function for parsing the CSA series sub header.'''
    csa_dict = simplify_csa_dict(csareader.read(elem.value))
    
    #If there is a phoenix protocol, parse it and dump it into the csa_dict
    if 'MrPhoenixProtocol' in csa_dict:
        phoenix_dict = parse_phoenix_prot(csa_dict['MrPhoenixProtocol'])
        del csa_dict['MrPhoenixProtocol']
        
        for key, val in phoenix_dict.iteritems():
            new_key = '%s.%s' % ('MrPhoenixProtocol', key)
            csa_dict[new_key] = val
        
    return csa_dict

csa_series_trans = Translator('CsaSeries',
                              dicom.tag.Tag(0x29, 0x1020),
                              'SIEMENS CSA HEADER',
                              csa_series_trans_func)
'''Translator for parsing the CSA series sub header.'''


default_translators = (csa_image_trans, 
                       csa_series_trans,
                      )
'''Default translators for MetaExtractor'''

unpack_vr_map = {'SL' : 'i',
                 'UL' : 'I',
                 'FL' : 'f',
                 'FD' : 'd',
                 'SS' : 'h',
                 'US' : 'H',
                 'US or SS' : 'H',
                 }
'''Dictionary mapping value representations to corresponding format 
strings for the struct.unpack function.'''
                
def unpack_value(elem):
    '''Unpack DICOM element values that are stings, while the value 
    representation indicates they should be numeric. List values that do not 
    need to be unpacked will return a copy of the list.'''
    if elem.VR in unpack_vr_map and isinstance(elem.value, str):
        if elem.VM == 1:
            return struct.unpack(unpack_vr_map[elem.VR], elem.value)[0]
        else:
            return list(struct.unpack(unpack_vr_map[elem.VR], elem.value))
            
    if elem.VM == 1:
        if elem.VR == 'DS':
            return float(elem.value)
        elif elem.VR == 'IS':
            return int(elem.value)
        else:
            return elem.value
    else:
        if elem.VR == 'DS':
            return [float(val) for val in elem.value]
        elif elem.VR == 'IS':
            return [int(val) for val in elem.value]
        else:
            return elem.value[:]

def tag_to_str(tag):
    '''Convert a DICOM tag to a string representation using the group and 
    element hex values seprated by an underscore.'''
    return '%#X_%#X' % (tag.group, tag.elem)
    
class MetaExtractor(object):
    '''Callable object for extracting meta data from a dicom dataset'''
    def __init__(self, ignore_rules=None, translators=None, 
                 warn_on_trans_except=True):
        '''Initialize the object with a set of ignore rules and translators. If 
        any of these are 'None' they will be set to the module defaults. You 
        must pass an empty iterable for 'ignore_rules' and 'translators' if you 
        want to do nothing.

        If 'warn_on_trans_except' is True, exceptions generated in translators
        will become warnings (allowing the rest of the parsed meta dict to be
        returned).
        '''
        if ignore_rules is None:
            self.ignore_rules = default_ignore_rules
        else:            
            self.ignore_rules = ignore_rules
        if translators is None:
            self.translators = default_translators
        else:
            self.translators = translators
        self.warn_on_trans_except = warn_on_trans_except
                
    def _get_elem_name(self, elem):
        '''Get an element name for any non-translated elements.'''
        #Use standard DICOM keywords if possible
        name = keyword_for_tag(elem.tag)
        
        #For private tags we take elem.name and convert to camel case
        if name == '':
            name = elem.name
            if name.startswith('[') and name.endswith(']'):
                name = name[1:-1]
            tokens = [token[0].upper() + token[1:] 
                      for token in name.split()]
            name = ''.join(tokens)
        
        return name
        
    def __call__(self, dcm):
        '''Convert a DICOM dataset to a dictionary where the keys are the
        element names instead of numerical tag values. Nested datasets become 
        nested dictionaries and elements can be ignored or parsed by a provided 
        translator.
        '''
        standard_meta = []
        trans_meta_dicts = OrderedDict()
        
        #Make dict mapping tags to tranlators, initially just populate those 
        #where the priv_creator attribute in None
        trans_map = {}
        for translator in self.translators:
            if translator.priv_creator is None:
                if translator.tag in trans_map:
                    raise ValueError('More than one translator given for tag: '
                                     '%s' % translator.tag)
                trans_map[translator.tag] = translator
        
        for elem in dcm:
            if isinstance(elem.value, str) and elem.value.strip() == '':
                continue
            
            #Take square brackets off private element names
            name = self._get_elem_name(elem)
            
            #If it is a private creator element, handle any corresponding 
            #translators
            if elem.name == "Private Creator":
                for translator in self.translators:
                    if translator.priv_creator == elem.value:
                        new_tag = dicom.tag.Tag(elem.tag.group,
                                                (elem.tag.elem * 16**2 | 
                                                 translator.tag.elem)
                                               )
                        if new_tag in trans_map:
                            raise ValueError('More than one translator given '
                                             'for tag: %s' % translator.tag)
                        trans_map[new_tag] = translator
            
            #If there is a translator for this element, use it
            if elem.tag in trans_map:
                try:
                    meta = trans_map[elem.tag].trans_func(elem)
                except Exception, e:
                    if self.warn_on_trans_except:
                        warnings.warn("Exception from translator %s: %s" % 
                                      (trans_map[elem.tag].name,
                                       str(e)))
                    else:
                        raise
                else:
                    if meta:
                        trans_meta_dicts[trans_map[elem.tag].name] = meta
            #Otherwise see if we are supposed to ignore the element
            elif any(rule(elem) for rule in self.ignore_rules):
                continue
            #Handle elements that are sequences with recursion
            elif isinstance(elem.value, dicom.sequence.Sequence):
                value = []
                for val in elem.value:
                    value.append(self(val))
                standard_meta.append((name, value, elem.tag))
            #Otherwise just make sure the value is unpacked
            else:
                standard_meta.append((name, unpack_value(elem), elem.tag))
                
        #Handle name collisions
        name_counts = Counter(elem[0] for elem in standard_meta)
        result = OrderedDict()
        for name, value, tag in standard_meta:
            if name_counts[name] > 1:
                name = name + '_' + tag_to_str(tag)
            result[name] = value
                    
        #Inject translator results
        for trans_name, meta in trans_meta_dicts.iteritems():
            for name, value in meta.iteritems():
                name = '%s.%s' % (trans_name, name)
                result[name] = value
                    
        return result

default_extractor = MetaExtractor()
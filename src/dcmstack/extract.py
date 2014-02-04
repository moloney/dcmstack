"""
Extract meta data from a DICOM data set.
"""
import struct, warnings
from collections import namedtuple, defaultdict
import dicom
from nibabel.nicom import csareader
from .dcmstack import DicomStack
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

# Use keyword_for_tag if available, otherwise use compatibility function
try: # importable from pydicom >= 0.9.7
    from dicom.datadict import keyword_for_tag
except ImportError:
    def keyword_for_tag(tag):
        names = dicom.datadict.AllNamesForTag(tag)
        return names[0]

#This is needed to allow extraction on files with invalid values (e.g. too 
#long of a decimal string)
try: # Available from pydicom >= 0.9.7
    dicom.config.enforce_valid_values = False
except AttributeError:
    pass


def ignore_private(elem):
    '''Ignore rule for `MetaExtractor` to skip private DICOM elements (odd 
    group number).'''
    if elem.tag.group % 2 == 1:
        return True
    return False
    
def ignore_non_ascii_bytes(elem):
    '''Ignore rule for `MetaExtractor` to skip elements with VR of 'OW', 'OB', 
    or 'UN' if the byte string contains non ASCII characters.'''
    if elem.VR in ('OW', 'OB', 'OW or OB', 'UN'):
        if not all(' ' <= c <= '~' for c in elem.value):
            return True
    return False

default_ignore_rules = (ignore_private,
                        ignore_non_ascii_bytes)
'''The default tuple of ignore rules for `MetaExtractor`.'''


Translator = namedtuple('Translator', ['name', 
                                       'tag', 
                                       'priv_creator',
                                       'trans_func']
                       )
'''A namedtuple for storing the four elements of a translator: a name, the 
dicom.tag.Tag that can be translated, the private creator string (optional), and 
the function which takes the DICOM element and returns a dictionary.'''

def simplify_csa_dict(csa_dict):
    '''Simplify the result of nibabel.nicom.csareader.
    
    Parameters
    ----------
    csa_dict : dict
        The result from nibabel.nicom.csareader
        
    Returns
    -------
    result : OrderedDict
        Result where the keys come from the 'tags' sub dictionary of `csa_dict`. 
        The values come from the 'items' within that tags sub sub dictionary. 
        If items has only one element it will be unpacked from the list.
    '''
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
    '''Function for translating the CSA image sub header.'''
    return simplify_csa_dict(csareader.read(elem.value))

    
csa_image_trans = Translator('CsaImage', 
                             dicom.tag.Tag(0x29, 0x1010), 
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
    prot_dict : OrderedDict
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
    ascconv_start = prot_val.find('### ASCCONV BEGIN ###')
    ascconv_end = prot_val.find('### ASCCONV END ###')
    ascconv = prot_val[ascconv_start:ascconv_end].split('\n')[1:-1]
    
    result = OrderedDict()    
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
'''Default translators for MetaExtractor.'''
                
def tag_to_str(tag):
    '''Convert a DICOM tag to a string representation using the group and 
    element hex values seprated by an underscore.'''
    return '%#X_%#X' % (tag.group, tag.elem)

unpack_vr_map = {'SL' : 'i',
                 'UL' : 'I',
                 'FL' : 'f',
                 'FD' : 'd',
                 'SS' : 'h',
                 'US' : 'H',
                 'US or SS' : 'H',
                 }
'''Dictionary mapping value representations to corresponding format strings for 
the struct.unpack function.'''
        
def tm_to_seconds(time_str):            
    '''Convert a DICOM time value (value representation of 'TM') to the number 
    of seconds past midnight.
    
    Parameters
    ----------
    time_str : str
        The DICOM time value string
        
    Returns
    -------
    A floating point representing the number of seconds past midnight
    '''
    #Allow ACR/NEMA style format by removing any colon chars
    time_str = time_str.replace(':', '')
        
    #Only the hours portion is required
    result = int(time_str[:2]) * 3600

    str_len = len(time_str)
    if str_len > 2:
        result += int(time_str[2:4]) * 60
    if str_len > 4:
        result += float(time_str[4:])
    
    return float(result)

default_conversions = {'DS' : float, 
                       'IS' : int,
                       'AT' : str,
                       #'TM' : tm_to_seconds
                      }
    
def make_unicode(in_str):
    '''Try to convetrt in_str to unicode'''
    for encoding in ('utf8', 'latin1'):
        try:
            result = unicode(in_str, encoding=encoding)
        except UnicodeDecodeError:
            pass
        else:
            break
    else:
        raise ValueError("Unable to determine string encoding: %s" % in_str)
    return result

class MetaExtractor(object):
    '''Callable object for extracting meta data from a dicom dataset. 
    Initialize with a set of ignore rules, translators, and type 
    conversions. 
    
    Parameters
    ----------
    ignore_rules : sequence
        A sequence of callables, each of which should take a DICOM element 
        and return True if it should be ignored. If None the module 
        default is used.
        
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
        self.warn_on_trans_except = warn_on_trans_except
                
    def _get_elem_key(self, elem):
        '''Get the key for any non-translated elements.'''
        #Use standard DICOM keywords if possible
        key = keyword_for_tag(elem.tag)
        
        #For private tags we take elem.name and convert to camel case
        if key == '':
            key = elem.name
            if key.startswith('[') and key.endswith(']'):
                key = key[1:-1]
            tokens = [token[0].upper() + token[1:] 
                      for token in key.split()]
            key = ''.join(tokens)
        
        return key
        
    def _get_elem_value(self, elem):
        '''Get the value for any non-translated elements'''
        #If the VR is implicit, we may need to unpack the values from a byte 
        #string. This may require us to make an assumption about whether the 
        #value is signed or not, but this is unavoidable.
        if elem.VR in unpack_vr_map and isinstance(elem.value, str):
            n_vals = len(elem.value)/struct.calcsize(unpack_vr_map[elem.VR])
            if n_vals != elem.VM:
                warnings.warn("The element's VM and the number of values do "
                              "not match.")
            if n_vals == 1:
                value = struct.unpack(unpack_vr_map[elem.VR], elem.value)[0]
            else:
                value = list(struct.unpack(unpack_vr_map[elem.VR]*n_vals, 
                                           elem.value)
                            )
        else:
            #Otherwise, just take a copy if the value is a list
            n_vals = elem.VM
            if n_vals > 1:
                value = elem.value[:]
            else:
                value = elem.value
        
        #Handle any conversions
        if elem.VR in self.conversions:
            if n_vals == 1:
                value = self.conversions[elem.VR](value)
            else:
                value = [self.conversions[elem.VR](val) for val in value]
           
        return value
        
    def __call__(self, dcm):
        '''Extract the meta data from a DICOM dataset.
        
        Parameters
        ----------
        dcm : dicom.dataset.Dataset
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
        standard_meta = []
        trans_meta_dicts = OrderedDict()
        
        #Make dict mapping tags to tranlators
        trans_map = {}
        for translator in self.translators:
            if translator.tag in trans_map:
                raise ValueError('More than one translator given for tag: '
                                 '%s' % translator.tag)
            trans_map[translator.tag] = translator
        
        for elem in dcm:
            if isinstance(elem.value, str) and elem.value.strip() == '':
                continue
            
            #Get the name for non-translated elements
            name = self._get_elem_key(elem)
            
            #If it is a private creator element, handle any corresponding 
            #translators
            if elem.name == "Private Creator":
                moves = []
                for curr_tag, translator in trans_map.iteritems():
                    if translator.priv_creator == elem.value:
                        new_elem = ((translator.tag.elem & 0xff) | 
                                    (elem.tag.elem * 16**2))
                        new_tag = dicom.tag.Tag(elem.tag.group, new_elem)
                        if new_tag != curr_tag:
                            if (new_tag in trans_map or 
                                any(new_tag == move[0] for move in moves)
                               ):
                                raise ValueError('More than one translator '
                                                 'for tag: %s' % new_tag)
                            moves.append((curr_tag, new_tag))
                for curr_tag, new_tag in moves:
                    trans_map[new_tag] = trans_map[curr_tag]
                    del trans_map[curr_tag]
            
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
                standard_meta.append((name, 
                                      self._get_elem_value(elem), 
                                      elem.tag
                                     )
                                    )
                
        #Handle name collisions
        name_counts = defaultdict(int)
        for elem in standard_meta:
            name_counts[elem[0]] += 1
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
                
        #Make sure all string values are unicode
        for name, value in result.iteritems():
            if isinstance(value, str):
                result[name] = make_unicode(value)
            elif (isinstance(value, list) and 
                  len(value) > 0 and 
                  isinstance(value[0], str)):
                result[name] = [make_unicode(val) for val in value]
                    
        return result

def minimal_extractor(dcm):
    '''Meta data extractor that just extracts the minimal set of keys needed 
    by DicomStack objects.
    '''
    result = {}
    for key in DicomStack.minimal_keys:
        try:
            result[key] = dcm.__getattr__(key)
        except AttributeError:
            pass
    return result

default_extractor = MetaExtractor()
'''The default `MetaExtractor`.'''

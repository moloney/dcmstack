import os
import sys
from xml.etree import ElementTree as ET
import dicom
import re
import inspect

class dcm_meta_filter():
    """
        Anonymizes the input dicom file using a xml template
		sample xml template can be found here: 
		https://wiki.nci.nih.gov/display/CIP/Finalized+CTP+Anonymization+Profile+-+Basic

        xml_ctp_template = os.path.abspath('/path/to/ctp_template.xml')
        dcm_filter = dcm_meta_filter(xml_ctp_template)
        dcm_filter.anonymizeDicom('/path/to/source_dcm', '/path/to/dest/')

    """
    _tagFunctions = {}
    _tag_Main_VR = {}
    _tag_Repeat_VR = {}
    _parameters = {}
    _currentTag = None
    _grpTxtTag = {}
    _keepGroups = {}
    _tagCheckList = {}
    _ds = None



    def __init__(self, profile):

        self._profile = profile
        self._setDictionaries(profile)



    def _getTag(self, tagStr):

        splitPoint = int(len(tagStr)/2)
        group = tagStr[0:splitPoint]

        element = tagStr[splitPoint:len(tagStr)]

        return dicom.tag.Tag(group, element)

    def _overlays_callback(self, ds, data_element):
        #print 'DE: ', data_element
        if data_element.tag.group & 0xFF00 == 0x6000:
            #print "DELETING OVERLAYS"
            del self._ds[data_element.tag]


    def _curves_callback(self, ds, data_element):
        #print 'DE: ', data_element
        if data_element.tag.group & 0xFF00 == 0x5000:
            #print "DELETING CURVES"
            del self._ds[data_element.tag]




    def _removeOverlays(self):

        #print "INSIDE OVERLAYS"
        dataset = self._ds

        dataset.walk(self._overlays_callback)


    def _removeCurves(self):

        #print "INSIDE CURVES"
        dataset = self._ds

        dataset.walk(self._curves_callback)

    def _rpg_callback(self, ds, data_element):

        grp = data_element.tag.group

        decimal_grp = int(grp, 16)

        if (decimal_grp % 2 == 1):

            del self._ds[data_element.tag]

    def _removeprivategroups(self):

        #print "Remove Private Groups"

        dataset = self._ds

        dataset.walk(self._rpg_callback)


    def _setTagVR(self):

        from dicom import _dicom_dict

        dicomDict = _dicom_dict.DicomDictionary
        repeatersDict = _dicom_dict.RepeatersDictionary

        for key in dicomDict.keys():
            VR = (dicomDict[key])[0]

            tag = dicom.tag.Tag(key)
            self._tag_Main_VR[tag] = VR

        #ingoring the Repeaters for now: But it is mandatory to implement them.    



    def _setDictionaries(self,profile):

        import re

        self._setTagVR()
        tree = ET.parse(profile)

        for node in tree.getiterator():


            if not ( node.attrib.get('t') == None):
                if node.tag == 'p':

                    self._parameters[node.attrib.get('t')] = node.text

                elif node.tag == 'e':


                    tag = self._getTag(node.attrib.get('t'))
                    self._tagFunctions[tag] = node.text
                elif node.tag == 'k':


                    #for these cases:
                    #<k en="T" t="0018">Keep group 0018</k>/
                    #<k en="T" t="0020">Keep group 0020</k>/
                     #<k en="T" t="0028">Keep group 0028</k>/
                  
                    self._keepGroups[long(node.attrib.get('t'),16)] = node.text
                elif node.tag == 'r':
                    #<r en="T" t="curves">Remove curves</r>/
                    #<r en="T" t="overlays">Remove overlays</r>/
                    #<r en="T" t="privategroups">Remove private groups</r>/
                    #<r en="F" t="unspecifiedelements">Remove unchecked elements</r>/

                    self._grpTxtTag[node.attrib.get('t')] = node.text

                else:
                    raise NotImplementedError
                    sys.exit()

    def _printTags(self):


        #print "TAGS & VALUES:"
        for tag in self._tagFunctions:
            break
            #print tag ,' >> ', self._tagFunctions[tag]


    def _printParameters(self):


        #print "PARAMETERS & VALUES:"
        for parameter in self._parameters:

            break
            #print parameter, ' >> ', self._parameters[parameter]




    def _setDataSet(self,ds):

        self._ds = ds


    def _setCurrentTag(self, Tag):

        self._currentTag = Tag


    def append(self,toReturnVal):

        raise NotImplementedError("Not Implemented Yet!")

    def blank(self, toReturnVal, n):

        str = ""

        for index in range(0, n):
            str += ' '

        return str

    def contents(self, toReturnVal, *args):

        import re

        len_args = len(args)
        if len_args == 0:
            return

        else:
            ElementName = args[0]
            if ElementName in self._ds:

                de = self._ds.data_element(ElementName)

                if len_args == 2:
                    if toReturnVal == 0:
                         self._ds[self._currentTag].value = re.sub(r'%s' % args[1], '', de.value)
                    else:
                        return re.sub(r'%s' % args[1], '', de.value)
                elif len_args == 3:
                    if toReturnVal == 0:
                        self._ds[self._currentTag].value = re.sub(r'%s' % args[1], args[2], de.value)
                    else:
                        return re.sub(r'%s' % args[1], '', de.value)


            else:
                raise AttributeError, ElementName
                return None



    def date(self, toReturnVal, separator):

        import datetime
        now = datetime.datetime.now()
        str  = "\%Y%s\%m%s\%d" %(separator, separator)
        return now.strftime(str)

    def empty(self, toReturnVal):

        Tag = self._currentTag

        if Tag in self._ds.keys():

            if(toReturnVal == 0):
                self._ds[Tag].value = ""
            else:
                return ""
        return ""

    def encrypt(self, toReturnVal, *args):

        from Crypto.Cipher import AES
        import base64
        import os
        BLOCK_SIZE = 32

        PADDING = '{'
        pad = lambda s: s + (BLOCK_SIZE - len(s) % BLOCK_SIZE) * PADDING
        EncodeAES = lambda c, s: base64.b64encode(c.encrypt(pad(s)))


        len_args = len(args)

        if len_args == 1:

            cipher = AES.new(args[0])
            encoded = EncodeAES(cipher, self._ds[self._currentTag].value)

            if toReturnVal == 0:
                self._ds[self._currentTag].value = encoded
            else:
                return encoded
        else:
            cipher = AES.new(args[1])

            ElementName = args[0]
            if ElementName in self._ds:
                de = self.ds.data_element(ElementName)

                val = de.value


                encoded = EncodeAES(cipher, val)

                if toReturnVal == 0:
                    self._ds[self._currentTag].value = encoded
                else:
                    return encoded
            else:

                raise AttributeError, ElementName
                return None


    def hash(self, toReturnVal, *args):

        import md5

        Tag = self._currentTag
        if Tag in self._ds.keys():

            val = (self._ds[Tag]).value

            hashs = md5.new(val).hexdigest()

            baseTenHash = str(int(hashs, 16))

            if(len(args) == 0): #hash(this)

                    if toReturnVal == 0:
                        (self._ds[Tag]).value = baseTenHash

                    else:
                        return baseTenHash

            else:    #hash(this,maxCharsOutput)
                if(int(args[0]) <= len(baseTenHash)): #max chars to output  < len of the hash string

                    if toReturnVal == 0:
                        (self._ds[Tag]).value = baseTenHash[0:int(args[0]) - 1]
                    else:
                        return baseTenHash[0:int(args[0]) - 1]
                else:

                    raise ValueError, "%d is greater than len of md5 Hash with length: %d" % (int(args[0]), len(baseTenHash))
                    return None
        else:
            raise AttributeError, Tag
            return None


    def hashname(self,toReturnVal, *args):

        import md5
        import re

        Tag = self._currentTag
        if Tag in self._ds.keys():
            words = (self._ds[Tag]).value

            if (len(args) == 1):
                words = re.sub(r'"', '', words)

                words = re.sub(r'\'', '', words)

                words = re.sub(r'\.', '', words)

                words = re.sub(r' ', '', words)

                hashs = md5.new(words).hexdigest()
                baseTenHash = str(int(hashs, 16))

                if toReturnVal == 0:
                    (self._ds[Tag]).value = baseTenHash[0: int(args[0])  ]
                else:
                    return baseTenHash[0: int(args[0])  ]

            elif(len(args) > 1):
                words = re.sub(r'"', '', words[0: int(args[1])  ])

                words = re.sub(r'\'', '', words)

                words = re.sub(r'\.', '', words)

                words = re.sub(r' ', '', words)
                hashs = md5.new(words).hexdigest()

                baseTenHash = str(int(hashs, 16))

                if toReturnVal == 0:
                    (self._ds[Tag]).value = baseTenHash[0: int(args[0])  ]
                else:
                    return baseTenHash[0: int(args[0])  ]

        else:
            raise AttributeError, Tag




    def hashptid(self, toReturnVal, *args):

        import md5
        flag = 0
        Tag = self._currentTag
        args = args[0]
        if Tag in self._ds.keys():

            val = None
            if(len(args) >= 2):
                elName = args[1]

                if elName in self._ds:
                    #print 'going? '
                    flag = 1
                    val = self._ds.data_element(args[1]).value

                else:

                    val = (self._ds[Tag]).value

            #print val,' > ', args
            hashs = md5.new('[' + args[0] + ']' + val).hexdigest()

            baseTenHash = str(int(hashs, 16))

            if(len(args) < 3): #hash(this)

                if flag == 1:
                    if toReturnVal == 0:
                        (self._ds[Tag]).value = baseTenHash
                    else:
                        return baseTenHash
                else:
                    if toReturnVal == 0:
                        (self._ds[Tag]).value = baseTenHash[0:int(args[1])]
                    else:
                        return baseTenHash[0:int(args[1])]

            else:    #hash(this,maxCharsOutput)
                if(int(args[2]) <= len(baseTenHash)): #max chars to output  < len of the hash string
                    if toReturnVal == 0:
                        (self._ds[Tag]).value = baseTenHash[0:int(args[2])]
                    else:
                        return baseTenHash[0:int(args[2])]
                else:

                    raise ValueError, "%d is greater than len of md5 Hash with length: %d" % (int(args[0]), len(baseTenHash))
        else:
            raise AttributeError, Tag


        return

    def hashuid(self, toReturnVal, root):

        import md5

        Tag = self._currentTag

        if Tag in self._ds.keys():
            existingID = (self._ds[Tag]).value

            hashs = md5.new(existingID).hexdigest()

            baseTenHash = str(int(hashs, 16))

            if  root.endswith('.'):

                if toReturnVal == 0:
                    (self._ds[Tag]).value = str(root) + baseTenHash
                else:
                    return str(root) + baseTenHash
            else:
                if toReturnVal == 0:
                    (self._ds[Tag]).value = str(root) + '.' + baseTenHash
                else:
                    return str(root) + '.' + baseTenHash

        else:
            raise AttributeError, Tag

            return None

    def incrementdate(self, toReturnVal, incInDays):

        import datetime

        Tag = self._currentTag
        if Tag in self._ds.keys():
            currD = (self._ds[Tag]).value
            #print 'fsdfsfd ',type(currD)

            try:
                date = datetime.datetime.strptime(currD, '%Y%m%d')
            except:

                date = datetime.datetime.now()

            inc = datetime.timedelta(days=int(incInDays))


            if toReturnVal == 0:
                (self._ds[Tag]).value = (date + inc).strftime("%Y%m%d")
            else:
                return (date + inc).strftime("%Y%m%d")
        else:
            raise AttributeError, Tag


    def initials(self, toReturnVal, ElementName):

        if ElementName in self.ds:

            de = self.ds.data_element(ElementName)

            values = de.value.split('^')

            initials = ""

            for index in range(1,len(values)):

                initials += ((values[index])[0]).upper()

            initials += ((values[0])[0]).upper()

            return initials
        else:
            raise AttributeError, ElementName
            return None


    def integer(self, toReturnVal, ElementName, KeyType, width):

        raise NotImplementedError


    def keep(self, toReturnVal):

        raise NotImplementedError


    def lookup(self, toReturnVal, ElementName, KeyType):

        raise NotImplementedError

    def modifydate(self, toReturnVal, ElementName, year, month, day):

        raise NotImplementedError


    def param(self, toReturnVal, *args):
        args = args[0]
        if toReturnVal == 0:
            self._ds[self._currentTag].value = args[0]
        else:
            return args[0]


    def process(self, toReturnVal):

        raise NotImplementedError


    def remove(self, toReturnVal):

        tag = self._currentTag
        if tag in self._ds.keys():

            del self._ds[tag]
        else:
            raise AttributeError, tag
            return


    #3 functions into one require(), require(ElementName), require(ElementName,"default value")
    def require(self, toReturnVal, *args):


        tag = self._currentTag

        dataset = self._ds

        if not (tag in self._ds.keys() ):

            if(len(args) == 0):

                if(tag in self._tag_Main_VR):
                    #print '1 for tag ',tag,' add VR:',self._tag_Main_VR[tag]
                    dataset.add_new(tag, self._tag_Main_VR[tag], "")
                else:
                    dataset.add_new(tag, "SH", "")
                    raise NotImplementedError

            elif(len(args) == 1):

                check = dataset.dir(args[0])

                if len(check) == 0: # Named Element Does Not Exist

                    if(tag in self._tag_Main_VR):
                        #print '2 for tag ',tag,' add VR:',self._tag_Main_VR[tag]
                        dataset.add_new(tag, self._tag_Main_VR[tag], "")
                    else:
                        dataset.add_new(tag, "SH", "")
                        raise NotImplementedError

                else: # Named Element Found

                    de = dataset.data_element(check[0])
                    dataset.add_new(tag, de.VR, de.value)

            elif(len(args) == 2):

                check = dataset.dir(args[0])

                if len(check) == 0: # Named Element Does Not Exist
                    if(tag in self._tag_Main_VR):
                        #print '3 for tag ',tag,' add VR:',self._tag_Main_VR[tag]
                        dataset.add_new(tag, self._tag_Main_VR[tag], args[1])
                    else:
                        dataset.add_new(tag, "SH", args[1])
                        raise NotImplementedError
                else: # Named Element Found

                    de = dataset.data_element(check[0])
                    dataset.add_new(tag, de.VR, de.value)


    def round(self, toReturnVal, ElementName, groupsize):

        raise NotImplementedError


    def time(self, toReturnVal, *args):

        import datetime

        timenow = datetime.datetime.now()

        if len(args) == 0 or args[0] == ":":

            strtime = "%H:%M:%S"
        else:
            separator = args[0]
            strtime = "%H%s%M%s%S" % (separator, separator)

        times = timenow.strftime(strtime)

        self._ds[self._currentTag].value = times

        return times

    def truncate(self, toReturnVal, ElementName, n):

        if ElementName in self.ds:

            de = self.ds.data_element(ElementName)

            value = de.value()

            if n == 0:
                return ""
            elif n > len(value):
                return value
            elif n < 0:

                return value[n:len(value)]

            else:
                return value[0:n]

        else:
            raise AttributeError, ElementName
            return None


    def _processArgs(self, func, args):

        pArg = []
        for arg in args:

            if not arg == 'this':

                if '@' in arg:

                    val = (arg.split('@'))[1]

                    if val in self._parameters:

                        val = self._parameters[val]
                        pArg.append(val)

                    else:
                        #print 'something fishy with %s -> ' %(func) + str(args)
                        return None

                else:
                    pArg.append(arg)

        return pArg

    def _simpleParser(self, val):

        if '@' in val:

            fA = (val.split('@', 1))[1]

            split = fA.split('(', 1)

            func = split[0]
            args = re.sub(r'\)', '', split[1])

            if args == '':
                args = None
            else:
                args = args.split(',')
                args = self._processArgs(func, args)

            #print func, ' ', str(args)
            return (func, args)
        else:

            return (None, val)

    def _processIFC(self, args):

        conditions = ['exists', 'isblank', 'equals', 'contains', 'matches']

        elName = args[0]

        dataElement = None

        if elName.lower() == 'this':

            dataElement = self._ds[self._currentTag]
        else:

            if len((self._ds).dir[elName]) > 0:
                dataElement = self._ds.data_element(elName)


        if args[1].lower() == 'exists':

            if dataElement == None:
                return 0
            return 1

        elif args[1].lower() == 'isblank':

            if dataElement == None:
                return 1
            else:

                val = dataElement.value

                if val == "":
                    return 0

                else:

                    for v in val:

                        if val != ' ':
                            return 1

                    return 0

        elif args[1].lower() == 'equals':

            if dataElement == None:
                return 1
            else:
                val = dataElement.value

                if val == args[2]:
                    return 0
                return 1

        elif args[1].lower() == 'contains':

            if dataElement == None:
                return 1
            else:
                val = dataElement.value

                if args[2] in val:
                    return 0
                return 1


        elif args[1].lower() == 'matches':


            import re

            if dataElement == None:
                return 1
            else:
                x = re.compile(args[2])

                val = dataElement.value

                if re.match(x, val):

                    return 0
                return 1

        else:
            print 'did nothing'
        return 1


    def _ifParser(self, val):

        #print val

        splits = val.split('}{')

        falseClause = (splits[1].split('}'))[0]
        trueClause = (splits[0].split('){'))[1]
        ifCondition = (splits[0].split('){'))[0]

        args = (ifCondition.split('@if('))[1]

        args = args.split(',')

        #print str(args)

        index = self._processIFC(args)

        #print "idx: ",index
        func = None
        args = None
        if index == 0:
            func, args = self._simpleParser(trueClause)
        else:
            func, args = self._simpleParser(falseClause)

        #print "func, args from ifparser: %s , %s" %(func, str(args)) 
        return func, args



    def _getFunctionAndArgs(self, val):

        if('@if' not in val):
            func, args = self._simpleParser(val)
        else:
            func, args = self._ifParser(val)

        return (func, args)

    def _execFunc(self, func, key, funcHandle, args):

        tag = key
        #print 'TAG>> ',tag
        if (tag in (self._ds).keys()) or ('require' in func.lower()):

            self._setCurrentTag(tag)
            if args == None:
                #print 'called from none'
                funcHandle(0)

            else:
                #print 'called from args'

                funcHandle(0,*args)

        #else:

            #print 'tag : ',tag,' not in the dataset'


    def _removeuncheckedElements(self):

        #tagx = "0x00080008"

        tagx = dicom.tag.Tag("0008", "0008")
        for tag in self._tagCheckList:

            group = tag.group


            if not group in self._keepGroups and tag in self._ds.keys() and not tag == tagx:

                #print 'removing data_element: ', (self._ds[tag]).name

                del self._ds[tag]

    def _anonSpecialCases(self):

        for key in self._grpTxtTag:


            if 'overlays' in key or 'curves' in key or 'privategroups' in key or 'unspecifiedelements' in key:
                txt = self._grpTxtTag[key]
                txt = re.sub(r' ', '', txt)
                txt = txt.lower()
                #print ">> ", str(txt)
                if(txt == 'removeoverlays'):
                    self._removeOverlays()

                elif(txt == 'removecurves'):
                    self._removeCurves()

                elif(txt == 'privategroups'):
                    self._removeprivategroups()
                else:
                    #print 'unspecified/ unchecked elements'
                    self._removeuncheckedElements()


    def _doAnonymization(self):

        for key in self._tagFunctions:

            val = None

            val = self._tagFunctions[key]



            if not val == None and key in self._ds.keys():
                if '@' in val and (not '-@' in val):

                    self._setCurrentTag(key)
                    func, args = self._getFunctionAndArgs(val)

                    if not func == None and not func == 'param':
                        if hasattr(self, func):
                            funcHanldle = getattr(self, func)

                            #print 'calling func: %s args:%s' %(func,str(args))
                            self._execFunc(func, key, funcHanldle, args)
                elif('-@' in val):
                    self._setCurrentTag(key)

                    #print "case:  ",val
                    vals = val.split('-')

                    func1, args1 = self._getFunctionAndArgs(vals[0])

                    func2, args2 = self._getFunctionAndArgs(vals[1])

                    #print args1, ' > ', args2

                    funcHanldle1 = getattr(self, func1)
                    funcHanldle2 = getattr(self, func2)
                    self._ds[self._currentTag].value = str(funcHanldle1(1, args1))+'-'+str(funcHanldle2(1, args2))
                    #print 'done here ',self._ds[self._currentTag].value 
                else:

                    #implement cases like CTP etc
                    self._setCurrentTag(key)
                    if key in self._ds.keys():
                        self._ds[ self._currentTag].value = val
            else:
                #make provisions for removing unchecked elements
                if key in (self._ds).keys():
                    self._tagCheckList[key] = 1


        self._anonSpecialCases()

    def _saveas(self, targetFile):

        #try
        self._ds.save_as(targetFile)
        #except:
        #    raise Exception("Dir %s does not exist or not enough permissions to create output file"%(os.path.dirname(targetFile)))

    def anonymizeDicom(self, dicomFile, targetFile):

        ds = dicom.read_file(dicomFile)

        self._setDataSet(ds)

        self._doAnonymization()

        #print self._ds
        self._saveas(targetFile)

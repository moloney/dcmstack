"""Provide a 'slight' normalization layer on top off DICOM meta data in a DcmMetaExtension
"""
import re
from dataclasses import dataclass
from typing import Optional, List, Callable, Set, Dict, Union, Tuple, Any

import pint

from .dcmmeta import DcmMetaExtension
from .convert import dt_to_datetime


ureg = pint.UnitRegistry()


@dataclass(frozen=True)
class SrcAttr:
    """A source attribute used to produce normalized attributes"""
    name: Union[str, Tuple[str]]

    units: Optional[pint.Quantity] = None

    def __post_init__(self):
        if isinstance(self.name, str):
            object.__setattr__(self, "name", (self.name,))

    def get_source(self, meta: DcmMetaExtension):
        for name in self.name:
            cls = meta.get_classification(name)
            if cls is not None:
                return name


# Define some common source attributes
SRCS = {
    "AcquisitionDateTime" : SrcAttr(
        ("FrameReferenceDateTime", "FrameAcquisitionDateTime", "AcquisitionDateTime")
    ),
    "MosaicRefAcqTimes" : SrcAttr(
        ("CsaImage.MosaicRefAcqTimes", "SIEMENS_MR_HEADER.MosaicRefAcqTimes")
    ),
    "FlipAngle" : SrcAttr("FlipAngle", ureg.degrees),
    "RepetitionTime" : SrcAttr("RepetitionTime", ureg.milliseconds),
    "EchoTime" : SrcAttr(("EffectiveEchoTime", "EchoTime"), ureg.milliseconds),
    "InversionTime" : SrcAttr(("InversionTime", "InversionTimes"), ureg.milliseconds),
    "DiffusionBValue" : SrcAttr(
        (
            "DiffusionBValue", 
            "SIEMENS_MR_HEADER.0XC", 
            "SIEMENS_MR_HEADER.B_value", 
            "CsaImage.B_value",
            "GEMS_PARM_01.0X39",
            "PHILIPS_IMAGING_DD_001.0X3",
        ),
        ureg.seconds / ureg.mm ** 2,
    ),
}


@dataclass(frozen=True)
class SndAttr:
    """A normalized attribute we can extract from the meta data"""
    name: str

    units: Optional[pint.Quantity] = None

    short_name: Optional[str] = None

    def convert_single_source(self, src: SrcAttr, val):
        """Handle simple case of single source with potential unit conversion"""
        if self.units is None or src.units is None:
            return val
        if not isinstance(val, (float, int)):
            return [self.units.m_from(x * src.units) for x in val]
        return self.units.m_from(val * src.units)


# Define the normalized attributes
SND_ATTRS = [
    SndAttr("AcquisitionDescription"),
    SndAttr("AcquisitionTimeStamp", ureg.seconds),
    SndAttr("AcquisitionTimeStampSources"),
    SndAttr("FlipAngle", ureg.degrees, "FA"),
    SndAttr("RepetitionTime", ureg.seconds, "TR"),
    SndAttr("EchoTime", ureg.seconds, "TE"),
    SndAttr("InversionTime", ureg.seconds, "TI"),
    SndAttr("DiffusionBValue", ureg.seconds / ureg.mm ** 2, "b-value"),
]
SND_ATTR_MAP = {x.name: x for x in SND_ATTRS}


# Define some source attributes we assume are needed and don't need to be specified
GLOBAL_SOURCES = ["Manufacturer", "Modality"]


@dataclass(frozen=True)
class MetaNorm:
    """Define conversion from source meta data into normalized meta data
    
    Default `convert` function just returns the first source attribute that exists, 
    adjusting the units if needed.
    """
    out: Union[Union[SndAttr, str], Tuple[Union[SndAttr, str], ...]]

    sources: Optional[Tuple[Union[str, SrcAttr], ...]] = None

    convert: Optional[Callable[[Tuple[SrcAttr], DcmMetaExtension], Tuple[Any, ...]]] = None

    @property
    def n_out(self):
        return self._n_out

    def __post_init__(self):
        out = self.out
        n_out = 1
        if not isinstance(out, SndAttr):
            if isinstance(out, str):
                out = SND_ATTR_MAP[out] if out in SND_ATTR_MAP else SndAttr(out)
            else:
                n_out = len(out)
                out = tuple(SndAttr(x) if isinstance(x, str) else x for x in out)
            object.__setattr__(self, 'out', out)
        srcs = self.sources
        if srcs is None:
            srcs = [out.name] if n_out == 1 else [x.name for x in out]
        srcs = [SRCS.get(x, x) for x in srcs]
        object.__setattr__(
            self, 
            'sources', 
            tuple(SrcAttr(x) if isinstance(x, str) else x for x in srcs)
        )
        if self.convert is None:
            if n_out > 1:
                raise ValueError("No default 'convert' supports multiple 'out'")
            object.__setattr__(self, 'convert', self._default_convert)
        object.__setattr__(self, '_n_out', n_out)

    def _default_convert(self, meta: DcmMetaExtension):
        for src in self.sources:
            src_attr = src.get_source(meta)
            if src_attr is not None:
                val, cls = meta.get_values_and_class(src_attr)
                val = self.out.convert_single_source(src, val)
                return (val, cls)
        return (None, None)


@dataclass(frozen=True)
class ParseGroup:
    """Define parsing for group of related attributes"""
    attr_norms: Tuple[Union[str, MetaNorm], ...]

    applies: Optional[Callable] = None

    applies_sources: Optional[List[str]] = None
    
    def __post_init__(self):
        object.__setattr__(
            self, 
            'attr_norms', 
            tuple(MetaNorm(x) if isinstance(x, str) else x for x in self.attr_norms)
        )

    @property
    def sources(self) -> Set:
        res = set()
        if self.applies_sources:
            for src in self.applies_sources:
                res.add(src)
        for norm in self.attr_norms:
            for src in norm.sources:
                res.add(src)
        return res


def get_acq_descr(meta: DcmMetaExtension):
    gmeta = meta.get_class_dict(("global", "const"))
    srcs = ("SeriesDescription", "ProtocolName")
    mfgr = gmeta.get("Manufacturer")
    if mfgr is not None and re.search('Siemens', mfgr, re.IGNORECASE):
        srcs = ("ProtocolName", "SeriesDescription")
    for src in srcs:
        res = gmeta.get(src)
        if res is not None:
            return (res, ("global", "const"))
    return (None, None)


def get_acq_ts(meta: DcmMetaExtension):
    srcs = []
    # The "FrameReferenceDateTime" is the ideal source, as it should correspond to the
    # center of k-space
    for key in ("FrameReferenceDateTime", "FrameAcquisitionDateTime" "AcquisitionDateTime"):
        acq_dt, cls = meta.get_values_and_class(key)
        if acq_dt is not None:
            srcs.append(key)
            if isinstance(acq_dt, str):
                acq_dt = [acq_dt]
            break
    else:
        for key in ("AcquisitionTime",):
            acq_tm, cls = meta.get_values_and_class(key)
            if acq_tm is not None:
                if isinstance(acq_tm, str):
                    acq_tm = [acq_tm]
                srcs.append(key)
                break
        else:
            return ((None, None), (None, None))
        # TODO: If AcquisitionDate is define can we skip doing a wrap check?
        needs_wrp_check = False
        for key in ("AcquisitionDate", "SeriesDate"):
            acq_date = meta.get_values(key)
            if acq_date is not None:
                srcs.append(key)
                break
        else:
            acq_date = "19000101"
        if isinstance(acq_date, str):
            needs_wrp_check = True
            acq_dt = [acq_date + tm for tm in acq_tm]
        else:
            for dt, tm in zip(acq_date, acq_tm):
                acq_dt = [dt + tm]
        # TODO: Implement heurstic check if times span more than one day
    acq_ts = [dt_to_datetime(dt).timestamp() for dt in acq_dt]
    for key in ("CsaImage.MosaicRefAcqTimes", "SIEMENS_MR_HEADER.MosaicRefAcqTimes"):
        vals = meta.get_values(key)
        if vals is not None:
            assert len(acq_ts) == 1
            base_ts = acq_ts[0]
            acq_ts = [base_ts + (x / 1000.0) for x in vals]
            cls = ("global", "slices")
            srcs.append(key)
            break
    if cls == ("global", "const"):
        acq_ts = acq_ts[0]
    return ((acq_ts, cls), (srcs, ("global", "const")))
    


GEN_ATTRS = ParseGroup(
    [
        MetaNorm(
            "AcquisitionDescription", ("SeriesDescription", "ProtocolName"), get_acq_descr
        ),
        MetaNorm(
            ("AcquisitionTimeStamp", "AcquisitionTimeStampSources"), 
            ("AcquisitionTime", "AcquisitionDate", "AcquisitionDateTime", "MosaicRefAcqTimes"),
            get_acq_ts,
        )
    ]
)


MR_ATTRS = ParseGroup(
    [
        "FlipAngle",
        "RepetitionTime",
        "EchoTime",
        "InversionTime",
        "DiffusionBValue",
    ],
    lambda meta: meta.get_values("Modality") == "MR"
)


SND_GROUPS = (GEN_ATTRS, MR_ATTRS)


def get_snd_sources(groups=SND_GROUPS) -> Set:
    """Get set containing all the needed source meta data keys"""
    res = set(GLOBAL_SOURCES)
    for grp in groups:
        for src in grp.sources:
            res |= set(src.name)
    return res


SND_VERS = "20241118"


def gen_snd_meta(meta: DcmMetaExtension, groups: Tuple[ParseGroup, ...] = SND_GROUPS):
    """Generate the normalized meta data
    
    At least for now we assume `meta` comes from a single DICOM file.
    """
    for grp in groups:
        if grp.applies is not None and not grp.applies(meta):
            continue
        for norm in grp.attr_norms:
            res = norm.convert(meta)
            if norm.n_out == 1:
                val, cls = res
                if val is not None:
                    yield (cls, norm.out.name, val)
            else:
                for out_idx, (val, cls) in enumerate(res):
                    if val is not None:
                        if val is not None:
                            yield (cls, norm.out[out_idx].name, val)
    yield (("global", "const"), "__version__", SND_VERS)


def inject(
    meta: DcmMetaExtension, 
    prefix: str = "SND.", 
    groups: Tuple[ParseGroup, ...] = SND_GROUPS
):
    """Inject normalized meta data into the DcmMetaExtension"""
    for cls, key, val in gen_snd_meta(meta, groups):
        meta.get_class_dict(cls)[f"{prefix}{key}"] = val

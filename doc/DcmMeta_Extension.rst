DcmMeta Extension
=================

The DcmMeta extension is a complete but concise representation of the meta 
data in a series of source DICOM files. 

The primary goals are:

#. Preserve as much meta data as possible
#. Make the meta data more accessible
#. Make the meta data human readable and editable

Extraction
----------

The meta data is extracted from each DICOM input into a set of key/value pairs. 
Each non-private DICOM element uses the standard DICOM keyword as its key. 
Values are generally left unchanged (except for 'DS' and 'IS' value 
representations which are converted from strings to float and integer numbers 
respectively).

Translators are used to convert private elements into sets of key/value pairs. 
These are then added to the standard DICOM meta data with the translator name 
(followed by a dot) prepended to each of the keys it generates.

Private DICOM elements without translators are ignored by default, but this 
can be overridden. Any element with a value representation of 'OW' or 'OB' is 
ignored if it contains non-ASCII characters.

Summarizing
-----------

The meta data from individual input files is summarized over the dimensions of 
the Nifti file. Most of the meta data will be constant across all of the input 
files. Other meta data will be constant across each time/vector sample, or 
repeating for the slices in each time/vector sample. We summarize the meta data 
into one or more dictionaries as follows.

There will always be a dictionary 'global' with two nested dictionaries inside, 
'const' and 'slices'. The meta data that is constant across all input files get 
stored under the 'const' dictionary. The meta data that varies across all  
slices will be stored under slices, where each value is a list of values (one 
for each slice).

If there is a time dimension there will also be 'time' dictionary containing 
two nested dictionaries, 'samples' and 'slices'. Meta data that is constant 
across a time sample will be stored in the 'samples' dictionary with each being 
value a list of values (one for each time sample). Values that repeat across 
the slices in a time sample (a single volume) will be stored in the 'slices' 
dictionary with each value being a list of values (one for each slice in a time 
point).

If there is a vector dimension there will be a 'vector' dictionary, handled in 
the same manner as the 'time' dictionary.

Encoding
--------

The dictionaries of summarized meta data are encoded with JSON. A small amount 
of "meta meta" data that describes the DcmMeta extension is also included. 
This includes the affine ('dcmmeta_affine'), shape ('dcmmeta_shape'), any 
reorientation transform ('dcmmeta_reorient_transform'), and the slice dimension 
('dcmmeta_slice_dim') of the data described by the meta data. A version number 
for the DcmMeta extension ('dcmmeta_version') is also included.

The affine, shape, and slice dimension are used to determine if varying meta 
data is still valid. For example, if the image affine no longer matches 
the meta data affine (i.e. the image has been coregistered) then we cannot 
directly match the per-slice meta data values to slices of the data array.

The reorientation transform can be used to update directional meta data to 
match the image orientation. This transform encodes any reordering of the 
voxel data that occurred during conversion. If the image affine does not match 
the meta data affine, then an additional transformation needs to be done after 
applying the reorientation transform (from the meta data space to the image 
space).

Example
-------

Below is an example DcmMeta extension created from a data set with two slices 
and three time points (each with a different EchoTime). The meta data has been 
abridged (the "..." line) for clarity.

.. code-block:: python

    {
        "global": {
            "const": {
                "SpecificCharacterSet": "ISO_IR 100", 
                "ImageType": [
                    "ORIGINAL", 
                    "PRIMARY", 
                    "M", 
                    "ND"
                ], 
                "StudyTime": 69244.484, 
                "SeriesTime": 71405.562, 
                "Modality": "MR", 
                "Manufacturer": "SIEMENS", 
                "SeriesDescription": "2D 16Echo qT2", 
                "ManufacturerModelName": "TrioTim", 
                "ScanningSequence": "SE", 
                "SequenceVariant": "SP", 
                "ScanOptions": "SAT1", 
                "MRAcquisitionType": "2D", 
                "SequenceName": "se2d16", 
                "AngioFlag": "N", 
                "SliceThickness": 7.0, 
                "RepetitionTime": 3000.0, 
                "NumberOfAverages": 1.0, 
                "ImagingFrequency": 123.250392, 
                "ImagedNucleus": "1H", 
                "MagneticFieldStrength": 3.0, 
                "SpacingBetweenSlices": 10.5, 
                "NumberOfPhaseEncodingSteps": 96, 
                "EchoTrainLength": 1, 
                "PercentSampling": 50.0, 
                "PercentPhaseFieldOfView": 100.0, 
                "PixelBandwidth": 420.0, 
                "SoftwareVersions": "syngo MR B17", 
                "ProtocolName": "2D 16Echo qT2", 
                "TransmitCoilName": "TxRx_Head", 
                "AcquisitionMatrix": [
                    0, 
                    192, 
                    96, 
                    0
                ], 
                "InPlanePhaseEncodingDirection": "ROW", 
                "FlipAngle": 180.0, 
                "VariableFlipAngleFlag": "N", 
                "SAR": 0.11299714843984, 
                "dBdt": 0.0, 
                "StudyID": "1", 
                "SeriesNumber": 3, 
                "AcquisitionNumber": 1, 
                "ImageOrientationPatient": [
                    1.0, 
                    -2.051034e-10, 
                    0.0, 
                    2.051034e-10, 
                    1.0, 
                    1.98754e-11
                ], 
                "SamplesPerPixel": 1, 
                "PhotometricInterpretation": "MONOCHROME2", 
                "Rows": 192, 
                "Columns": 192, 
                "PixelSpacing": [
                    0.66666668653488, 
                    0.66666668653488
                ], 
                "BitsAllocated": 16, 
                "BitsStored": 12, 
                "HighBit": 11, 
                "PixelRepresentation": 0, 
                "SmallestImagePixelValue": 0, 
                "WindowCenterWidthExplanation": "Algo1", 
                "PerformedProcedureStepStartTime": 69244.546, 
                "CsaImage.EchoLinePosition": 48, 
                "CsaImage.UsedChannelMask": 1, 
                "CsaImage.MeasuredFourierLines": 0, 
                "CsaImage.SequenceMask": 134217728, 
                "CsaImage.RFSWDDataType": "predicted", 
                "CsaImage.RealDwellTime": 6200, 
                "CsaImage.ImaCoilString": "C:HE", 
                "CsaImage.EchoColumnPosition": 96, 
                "CsaImage.PhaseEncodingDirectionPositive": 1, 
                "CsaImage.GSWDDataType": "predicted", 
                "CsaImage.SliceMeasurementDuration": 286145.0, 
                "CsaImage.MultistepIndex": 0, 
                "CsaImage.ImaRelTablePosition": [
                    0, 
                    0, 
                    0
                ], 
                "CsaImage.NonPlanarImage": 0, 
                "CsaImage.EchoPartitionPosition": 32, 
                "CsaImage.AcquisitionMatrixText": "96*192s", 
                "CsaImage.ImaAbsTablePosition": [
                    0, 
                    0, 
                    -1630
                ], 
                "CsaSeries.TalesReferencePower": 334.36266914, 
                "CsaSeries.Operation_mode_flag": 2, 
                "CsaSeries.dBdt_thresh": 0.0, 
                "CsaSeries.ProtocolChangeHistory": 0, 
                "CsaSeries.GradientDelayTime": [
                    12.0, 
                    14.0, 
                    10.0
                ], 
                "CsaSeries.SARMostCriticalAspect": [
                    3.2, 
                    1.84627729, 
                    0.0
                ], 
                "CsaSeries.B1rms": [
                    7.07106781, 
                    1.59132133
                ], 
                "CsaSeries.RelTablePosition": [
                    0, 
                    0, 
                    0
                ], 
                "CsaSeries.NumberOfPrescans": 0, 
                "CsaSeries.dBdt_limit": 0.0, 
                "CsaSeries.Stim_lim": [
                    45.73709869, 
                    27.64929962, 
                    31.94370079
                ], 
                "CsaSeries.PatReinPattern": "1;FFS;45.36;10.87;3;0;2;866892320", 
                "CsaSeries.B1rmsSupervision": "NO", 
                "CsaSeries.ReadoutGradientAmplitude": 0.0, 
                "CsaSeries.MrProtocolVersion": 21710006, 
                "CsaSeries.RFSWDMostCriticalAspect": "Head", 
                "CsaSeries.SequenceFileOwner": "SIEMENS", 
                "CsaSeries.GradientMode": "Fast", 
                "CsaSeries.SliceArrayConcatenations": 1, 
                "CsaSeries.FlowCompensation": "No", 
                "CsaSeries.TransmitterCalibration": 128.29875, 
                "CsaSeries.Isocentered": 0, 
                "CsaSeries.AbsTablePosition": -1630, 
                "CsaSeries.ReadoutOS": 2.0, 
                "CsaSeries.dBdt_max": 0.0, 
                "CsaSeries.RFSWDOperationMode": 0, 
                "CsaSeries.SelectionGradientAmplitude": 0.0, 
                "CsaSeries.PhaseGradientAmplitude": 0.0, 
                "CsaSeries.RfWatchdogMask": 0, 
                "CsaSeries.CoilForGradient2": "AS092", 
                "CsaSeries.Stim_mon_mode": 2, 
                "CsaSeries.CoilId": [
                    255, 
                    196, 
                    238, 
                    238, 
                    238, 
                    238, 
                    238, 
                    238, 
                    238, 
                    238, 
                    238
                ], 
                "CsaSeries.Stim_max_ges_norm_online": 0.62600064, 
                "CsaSeries.CoilString": "C:HE", 
                "CsaSeries.CoilForGradient": "void", 
                "CsaSeries.TablePositionOrigin": [
                    0, 
                    0, 
                    -1630
                ], 
                "CsaSeries.MiscSequenceParam": [
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    93, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0, 
                    0
                ], 
                "CsaSeries.LongModelName": "NUMARIS/4", 
                "CsaSeries.Stim_faktor": 1.0, 
                "CsaSeries.SW_korr_faktor": 1.0, 
                "CsaSeries.Sed": [
                    1000000.0, 
                    156.13387238, 
                    156.13387238
                ], 
                "CsaSeries.PositivePCSDirections": "+LPH", 
                "CsaSeries.SliceResolution": 1.0, 
                "CsaSeries.Stim_max_online": [
                    0.22781265, 
                    17.30016327, 
                    0.5990392
                ], 
                "CsaSeries.t_puls_max": 0.0, 
                "CsaSeries.MrPhoenixProtocol.ulVersion": 21710006, 
                "CsaSeries.MrPhoenixProtocol.tSequenceFileName": "%SiemensSeq%\\se_mc", 
                "CsaSeries.MrPhoenixProtocol.tProtocolName": "2D 16Echo qT2", 
                ...
                "CsaSeries.MrPhoenixProtocol.sAsl.ulMode": 1, 
                "CsaSeries.MrPhoenixProtocol.ucAutoAlignInit": 1
            }, 
            "slices": {
                "InstanceCreationTime": [
                    71405.671, 
                    71405.562, 
                    71405.671, 
                    71405.578, 
                    71405.671, 
                    71405.578
                ], 
                "AcquisitionTime": [
                    71118.2425, 
                    71116.7375, 
                    71118.2625, 
                    71116.7575, 
                    71118.2825, 
                    71116.7775
                ], 
                "ContentTime": [
                    71405.671, 
                    71405.562, 
                    71405.671, 
                    71405.578, 
                    71405.671, 
                    71405.578
                ], 
                "InstanceNumber": [
                    1, 
                    2, 
                    7, 
                    8, 
                    13, 
                    14
                ], 
                "LargestImagePixelValue": [
                    2772, 
                    2828, 
                    2077, 
                    2085, 
                    1470, 
                    1397
                ], 
                "WindowCenter": [
                    1585.0, 
                    1513.0, 
                    1495.0, 
                    1455.0, 
                    1100.0, 
                    1084.0
                ], 
                "WindowWidth": [
                    3191.0, 
                    3212.0, 
                    2750.0, 
                    2731.0, 
                    2120.0, 
                    2073.0
                ], 
                "CsaImage.TimeAfterStart": [
                    1.505, 
                    0.0, 
                    1.525, 
                    0.02, 
                    1.545, 
                    0.04
                ], 
                "CsaImage.ICE_Dims": [
                    "1_1_1_1_1_1_1_4_1_1_1_1_490", 
                    "1_1_1_1_1_1_1_1_1_1_2_1_490", 
                    "1_2_1_1_1_1_1_4_1_1_1_1_490", 
                    "1_2_1_1_1_1_1_1_1_1_2_1_490", 
                    "1_3_1_1_1_1_1_4_1_1_1_1_490", 
                    "1_3_1_1_1_1_1_1_1_1_2_1_490"
                ]
            }
        }, 
        "time": {
            "samples": {
                "EchoTime": [
                    20.0, 
                    40.0, 
                    60.0
                ], 
                "EchoNumbers": [
                    1, 
                    2, 
                    3
                ]
            }, 
            "slices": {
                "ImagePositionPatient": [
                    [
                        -64.000001919919, 
                        -118.13729284881, 
                        -33.707626344045
                    ], 
                    [
                        -64.000001919919, 
                        -118.13729284881, 
                        -23.207628251394
                    ]
                ], 
                "SliceLocation": [
                    -33.707626341697, 
                    -23.207628249046
                ], 
                "CsaImage.ProtocolSliceNumber": [
                    0, 
                    1
                ], 
                "CsaImage.SlicePosition_PCS": [
                    [
                        -64.00000192, 
                        -118.13729285, 
                        -33.70762634
                    ], 
                    [
                        -64.00000192, 
                        -118.13729285, 
                        -23.20762825
                    ]
                ]
            }
        }, 
        "dcmmeta_shape": [
            192, 
            192, 
            2, 
            3
        ], 
        "dcmmeta_affine": [
            [
                -0.6666666865348816, 
                1.3673560894655878e-10, 
                0.0, 
                64.0
            ], 
            [
                1.3673560894655878e-10, 
                0.6666666865348816, 
                0.0, 
                -9.196043968200684
            ], 
            [
                0.0, 
                -1.325026720289113e-11, 
                10.499998092651367, 
                -33.70762634277344
            ], 
            [
                0.0, 
                0.0, 
                0.0, 
                1.0
            ]
        ], 
        "dcmmeta_reorient_transform": [
            [
                -0.0, 
                -1.0, 
                -0.0, 
                191.0
            ], 
            [
                1.0, 
                0.0, 
                0.0, 
                0.0
            ], 
            [
                0.0, 
                0.0, 
                1.0, 
                0.0
            ], 
            [
                0.0, 
                0.0, 
                0.0, 
                1.0
            ]
        ], 
        "dcmmeta_slice_dim": 2, 
        "dcmmeta_version": 0.6
    }

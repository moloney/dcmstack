from dcmstack.convert import tm_to_sec


def test_dcm_time_to_sec():
    assert tm_to_sec('100235.123456') == 36155.123456
    assert tm_to_sec('100235') == 36155
    assert tm_to_sec('1002') == 36120
    assert tm_to_sec('10') == 36000
    #Allow older NEMA style values
    assert tm_to_sec('10:02:35.123456') == 36155.123456
    assert tm_to_sec('10:02:35') == 36155
    assert tm_to_sec('10:02') == 36120
    assert tm_to_sec('10') == 36000

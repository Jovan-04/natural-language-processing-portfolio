from pretoken import Pretoken

def test_pretoken():
    pt = Pretoken("hello")
    assert pt

def test_characterize():
    pt = Pretoken("hello")

    assert pt.text == 'h'
    assert pt.next.text == 'e' # type: ignore
    assert pt.next.next.text == 'l' # type: ignore
    assert pt.next.next.next.text == 'l' # type: ignore
    assert pt.next.next.next.next.text == 'o' # type: ignore
    assert pt.next.next.next.next.next == None # type: ignore

def test_single_merge_rule():
    pt = Pretoken("hello")
    did_change = pt.apply_merge_rule("lo")

    assert did_change is True

    assert pt.text == 'h'
    assert pt.next.text == 'e' # type: ignore
    assert pt.next.next.text == 'l' # type: ignore
    assert pt.next.next.next.text == 'lo' # type: ignore
    assert pt.next.next.next.next == None # type: ignore

def consecutive_single_merge_rule():
    pt = Pretoken("hello")
    
    did_change = pt.apply_merge_rule("lo")
    assert did_change is True

    did_change = pt.apply_merge_rule("llo")
    assert did_change is True

    assert pt.text == 'h'
    assert pt.next.text == 'e' # type: ignore
    assert pt.next.next.text == 'llo' # type: ignore
    assert pt.next.next.next == None # type: ignore

def test_multi_merge():
    pt = Pretoken("hellolo")

    did_change = pt.apply_merge_rule("lo")
    assert did_change is True

    assert pt.text == 'h'
    assert pt.next.text == 'e' # type: ignore
    assert pt.next.next.text == 'l' # type: ignore
    assert pt.next.next.next.text == 'lo' # type: ignore
    assert pt.next.next.next.next.text == 'lo' # type: ignore
    assert pt.next.next.next.next.next == None # type: ignore
    
def test_to_tokens():
    pt = Pretoken("hello")
    pt.apply_merge_rule("lo")

    tokens = pt.get_tokens()

    assert tokens == ['h', 'e', 'l', 'lo']

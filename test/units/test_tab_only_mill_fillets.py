import math

import pytest
import shapely.wkt
from shapely.geometry import Point, box
from shapely.geometry.polygon import orient

from kikit.common import fromMm
from kikit.panelize import Panel

R = fromMm(1)
MM2 = fromMm(1) ** 2
# Area one mill fillet adds to a 90-degree inner corner
FILLET_AREA = (1 - math.pi / 4) * R ** 2


def pt(x, y):
    return Point(fromMm(x), fromMm(y))


def mmBox(x0, y0, x1, y1):
    return box(fromMm(x0), fromMm(y0), fromMm(x1), fromMm(y1))


def makePanel(tmp_path, parts, tabs):
    panel = Panel(str(tmp_path / "panel.kicad_pcb"))
    panel.boardSubstrate.union(list(parts) + list(tabs))
    panel.forwardTabs = list(tabs)
    panel.boardSubstrate.orient()
    return panel, panel.boardSubstrate.substrates


def added(original, result):
    return result.difference(original)


def assertNothingRemoved(original, result):
    assert original.difference(result).area < 1e-4 * MM2


def assertOnlyNearTabs(original, result, tabs, radius=R):
    near = tabs[0].union(tabs[-1]).buffer(radius * 1.01)
    assert added(original, result).difference(near).area < 1e-4 * MM2


def stackedBoards(gap, tabOrientation=1.0):
    """
    Two 10x10 mm boards stacked vertically with a gap, joined by a 4 mm wide
    tab spanning the gap at x = 3..7 mm.
    """
    boardA = mmBox(0, 0, 10, 10)
    boardB = mmBox(0, 10 + gap, 10, 20 + gap)
    tab = orient(mmBox(3, 10, 7, 10 + gap), sign=tabOrientation)
    return [boardA, boardB], [tab]


@pytest.mark.parametrize("tabOrientation", [1.0, -1.0], ids=["ccw", "cw"])
def test_slotEndBecomesSemicircle(tmp_path, tabOrientation):
    # A 2R wide slot: the two fillets at each slot end form a semicircle
    parts, tabs = stackedBoards(2, tabOrientation)
    panel, original = makePanel(tmp_path, parts, tabs)

    panel.addTabOnlyMillFillets(R)
    result = panel.boardSubstrate.substrates

    assert added(original, result).area == pytest.approx(4 * FILLET_AREA, rel=0.01)
    # Corners of the four fillets are filled
    for p in [pt(2.95, 10.2), pt(7.05, 10.2), pt(2.95, 11.8), pt(7.05, 11.8)]:
        assert result.contains(p)
    # The semicircular slot ends stay open, and so does the rest of the slot
    for p in [pt(2.9, 11.0), pt(7.1, 11.0), pt(1.0, 11.0), pt(9.0, 11.0)]:
        assert not result.contains(p)
    assertNothingRemoved(original, result)
    assertOnlyNearTabs(original, result, tabs)


def test_wideGapGetsSeparateQuarterFillets(tmp_path):
    # A 3R wide slot: four separate fillets with a straight section between
    parts, tabs = stackedBoards(3)
    panel, original = makePanel(tmp_path, parts, tabs)

    panel.addTabOnlyMillFillets(R)
    result = panel.boardSubstrate.substrates

    assert added(original, result).area == pytest.approx(4 * FILLET_AREA, rel=0.01)
    for p in [pt(2.95, 10.2), pt(2.95, 12.8), pt(7.05, 10.2), pt(7.05, 12.8)]:
        assert result.contains(p)
    for p in [pt(2.95, 11.5), pt(7.05, 11.5)]:
        assert not result.contains(p)
    assertNothingRemoved(original, result)
    assertOnlyNearTabs(original, result, tabs)


def test_boardFeaturesAwayFromTabsAreUnchanged(tmp_path):
    # Board A has a notch with a relief hole and an internal cutout; plain
    # milling would round all of them, tab-only milling must not touch them
    notch = mmBox(20, 0, 26, 4)
    relief = pt(20, 4).buffer(fromMm(0.3))
    cutout = mmBox(12, 3, 15, 6)
    boardA = mmBox(0, 0, 30, 10).difference(notch).difference(relief).difference(cutout)
    boardB = mmBox(0, 12, 30, 22)
    tab = mmBox(3, 10, 7, 12)
    probes = [pt(20, 4), pt(25.95, 3.95), pt(12.05, 3.05), pt(14.95, 5.95)]

    # Plain milling fills these corners, so the probes are meaningful
    plain, _ = makePanel(tmp_path, [boardA, boardB], [tab])
    plain.addMillFillets(R)
    assert sum(plain.boardSubstrate.substrates.contains(p) for p in probes) == 4

    panel, original = makePanel(tmp_path, [boardA, boardB], [tab])
    panel.addTabOnlyMillFillets(R)
    result = panel.boardSubstrate.substrates

    for p in probes:
        assert not result.contains(p)
    assert added(original, result).area == pytest.approx(4 * FILLET_AREA, rel=0.01)
    assertNothingRemoved(original, result)
    assertOnlyNearTabs(original, result, [tab])


def test_tabOnNotchFloorLeavesFloorEndsAlone(tmp_path):
    # The tab stands on the floor of a notch, overlapping it by 1 um as KiKit's
    # tabs do. A relief hole and a sharp corner at the ends of the same floor
    # must stay as they are.
    notch = mmBox(5, 6, 30, 10)
    relief = pt(5, 6).buffer(fromMm(0.3))
    boardA = mmBox(0, 0, 40, 10).difference(notch).difference(relief)
    boardB = mmBox(0, 12, 40, 22)
    tab = mmBox(20, 6 - 0.001, 24, 12)
    panel, original = makePanel(tmp_path, [boardA, boardB], [tab])

    panel.addTabOnlyMillFillets(R)
    result = panel.boardSubstrate.substrates

    for p in [pt(5, 6), pt(4.75, 6.0), pt(29.95, 6.05)]:
        assert not result.contains(p)
    # Fillets where the tab meets the notch floor and board B
    for p in [pt(19.95, 6.05), pt(24.05, 6.05), pt(19.95, 11.95), pt(24.05, 11.95)]:
        assert result.contains(p)
    assert added(original, result).area == pytest.approx(4 * FILLET_AREA, rel=0.01)
    assertNothingRemoved(original, result)
    assertOnlyNearTabs(original, result, [tab])


# A cropped outline from a real panel (coordinates in nm): a notch floor running
# from a relief hole at its left end to a tab. Earlier unions left sub-nanometre
# noise in these coordinates (note the floor's y), so the milled outline differs
# from the original by a zero-width sliver along the floor. That sliver used to
# join the tab's fillet to the relief hole and fill it. The noise is what
# triggers it: rounding to whole nanometres, or to 6 decimals, hides the problem.
NOISY_NOTCH_WKT = (
    "POLYGON ((68399997.5 5000000, 60399997.5 5000000, 60399997.5 6099999.414136846, "
    "41107112 6099999, 41082768 6122776, 41057172 6143721, 41030424 6162808, "
    "41002626 6180009, 40973880 6195296, 40944287 6208644, 40913949 6220023, "
    "40882968 6229408, 40851445 6236771, 40819482 6242085, 40787180 6245322, "
    "40754641 6246456, 40721966 6245458, 40689258 6242303, 40656617 6236962, "
    "40624145 6229409, 40592248 6219715, 40561310 6208020, 40531406 6194399, "
    "40502611 6178926, 40474998 6161675, 40448643 6142721, 40423619 6122138, "
    "40400001 6100000, 40377863 6076382, 40357280 6051358, 40338326 6025003, "
    "40321075 5997390, 40305601 5968595, 40291980 5938692, 40280285 5907754, "
    "40270591 5875857, 40263038 5843386, 40257697 5810746, 40254541 5778038, "
    "40253544 5745364, 40254678 5712825, 40257916 5680523, 40263229 5648561, "
    "40270593 5617038, 40279978 5586057, 40291358 5555719, 40304705 5526127, "
    "40319993 5497381, 40337193 5469583, 40356280 5442835, 40377225 5417238, "
    "40400001 5392894, 40400001 5000000, 39000000 5000000, 39000000 7000000, "
    "70000000 7000000, 70000000 6099999.620208393, 68399997.5 6099999.58586309, "
    "68399997.5 5000000))"
)
NOISY_NOTCH_TAB_WKT = (
    "POLYGON ((68399997.5 5000000, 60399997.5 5000000, 60399997.5 6100999.414136846, "
    "68399997.5 6100999.58586309, 68399997.5 5000000))"
)


def test_zeroWidthSliverDoesNotSpreadFillet(tmp_path):
    board = shapely.wkt.loads(NOISY_NOTCH_WKT)
    tab = shapely.wkt.loads(NOISY_NOTCH_TAB_WKT)
    panel, original = makePanel(tmp_path, [board], [tab])

    panel.addTabOnlyMillFillets(R)
    result = panel.boardSubstrate.substrates

    # The relief hole below the floor stays open
    for p in [pt(40.75, 6.2), pt(40.5, 6.15), pt(41.0, 6.15)]:
        assert not original.contains(p)
        assert not result.contains(p)
    # The tab gets a fillet on each side where it meets the floor
    assert result.contains(pt(60.35, 6.05))
    assert result.contains(pt(68.45, 6.05))
    assert added(original, result).area == pytest.approx(2 * FILLET_AREA, rel=0.02)
    assertNothingRemoved(original, result)
    assertOnlyNearTabs(original, result, [tab])


def test_noTabsOrZeroRadiusIsNoop(tmp_path):
    parts, tabs = stackedBoards(2)

    panel, original = makePanel(tmp_path, parts + tabs, [])
    panel.addTabOnlyMillFillets(R)
    assert panel.boardSubstrate.substrates.equals(original)

    panel, original = makePanel(tmp_path, parts, tabs)
    panel.addTabOnlyMillFillets(0)
    assert panel.boardSubstrate.substrates.equals(original)

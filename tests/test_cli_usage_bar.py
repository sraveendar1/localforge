from localforge.cli import _usage_bar


def _block_counts(rendered: str) -> tuple[int, int]:
    local = rendered.split("[success]")[1].split("[/success]")[0]
    frontier = rendered.split("[warning]")[1].split("[/warning]")[0]
    return len(local), len(frontier)


def test_usage_bar_splits_blocks_proportionally():
    rendered = _usage_bar(local_tokens=777, frontier_tokens=350, width=40)
    green, yellow = _block_counts(rendered)
    assert green + yellow == 40
    assert green == round(40 * 777 / (777 + 350))
    assert "69% open-weighted" in rendered
    assert "31% paid" in rendered


def test_usage_bar_all_local():
    rendered = _usage_bar(local_tokens=1000, frontier_tokens=0, width=40)
    green, yellow = _block_counts(rendered)
    assert (green, yellow) == (40, 0)
    assert "100% open-weighted / 0% paid" in rendered


def test_usage_bar_all_frontier():
    rendered = _usage_bar(local_tokens=0, frontier_tokens=1000, width=40)
    green, yellow = _block_counts(rendered)
    assert (green, yellow) == (0, 40)
    assert "0% open-weighted / 100% paid" in rendered


def test_usage_bar_zero_tokens_does_not_divide_by_zero():
    rendered = _usage_bar(local_tokens=0, frontier_tokens=0, width=40)
    assert "no tokens used" in rendered

import pandas as pd

from src import nhanes3


def test_fixed_width_reader_uses_sas_positions_and_blank_codes(tmp_path, monkeypatch):
    monkeypatch.setattr(nhanes3, 'RAW', tmp_path)
    (tmp_path / 'toy.sas').write_text('DATA X;\nINPUT\n  SEQN 1-5\n  ATPSI 6-8\n  CRP 9-13\n  ;\nRUN;\n')
    (tmp_path / 'toy.dat').write_text('00003 25 0.30\n00004888 8888\n00005 410.21\n')
    out = nhanes3.read_fixed('toy.dat', 'toy.sas', ['ATPSI', 'CRP'])
    assert out.SEQN.tolist() == [3, 4, 5]
    assert out.ATPSI.tolist()[0] == 25 and pd.isna(out.ATPSI[1]) and out.ATPSI[2] == 41
    assert out.CRP[0] == 0.30 and pd.isna(out.CRP[1]) and out.CRP[2] == 0.21


def test_every_mapped_target_is_a_model_input():
    from src.combined_features import COMBINED_FEATURES
    targets = {target for _, _, target, _, _ in nhanes3.VARIABLES if target}
    assert targets <= set(COMBINED_FEATURES)
    table = nhanes3.variable_map()
    assert table.model_input.tolist() == list(COMBINED_FEATURES)
    assert table.status.eq('mapped').sum() == len(targets) + 4     # plus the repeated BP readings

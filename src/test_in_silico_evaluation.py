import pandas as pd

from src.in_silico_evaluation import alert_reason, alerted, answer, forecast_summary, reported


def test_answers_follow_the_true_adherence_and_over_reporting_raises_them():
    assert [answer(a) for a in (0.9, 0.7, 0.5, 0.2)] == ['most', 'most', 'half', 'few']
    assert reported(0.3) == 0.15 and reported(0.3, share=0.3) == 0.5      # 0.3 / 0.7 = 0.43 -> about half


def test_alert_needs_two_no_progress_checkins_in_a_row_and_its_reason_comes_from_the_truth():
    stuck = 'no progress despite effort'
    rows = [{'month': 1, 'cause': stuck}, {'month': 2, 'cause': None}, {'month': 3, 'cause': stuck}]
    assert not alerted(rows)
    assert alerted(rows + [{'month': 4, 'cause': stuck}])
    pair = [{'true_adherence': 0.3}, {'true_adherence': 0.5}]
    assert alert_reason(pair, 1.0, 0.8) == 'intake higher than reported'
    pair = [{'true_adherence': 0.8}, {'true_adherence': 0.9}]
    assert alert_reason(pair, 0.7, 0.8) == 'slow body' and alert_reason(pair, 1.1, 0.8) == 'ordinary weight drift'


def test_forecast_summary_scores_each_method_against_the_truth():
    frame = pd.DataFrame([{'cutoff': 3, 'ahead': 3, 'true_kg': 80, 'twin_kg': 81, 'twin_low': 78, 'twin_high': 84,
                           'textbook_kg': 77, 'textbook_low': 79, 'textbook_high': 85, 'last_kg': 80}])
    out = forecast_summary(frame)['month 3 -> month 6']
    assert out['twin_mae_kg'] == 1 and out['textbook_mae_kg'] == 3 and out['last_weight_mae_kg'] == 0
    assert out['twin_coverage_90'] == 1 and out['textbook_coverage_90'] == 1

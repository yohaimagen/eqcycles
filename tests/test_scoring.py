import numpy as np
import pytest
from eqcycles.analysis.scoring import calculate_ot_score, get_transport_plan, normalize_coords

def test_calculate_ot_score_basic():
    # Two identical sets of events
    coords1 = np.array([[10.0, 1900.0], [20.0, 1910.0]])
    masses1 = np.array([5.0, 5.0])
    
    coords2 = coords1.copy()
    masses2 = masses1.copy()
    
    config = {
        'reg': 0.1,
        'reg_m': 10.0,
        'scale_x': 1.0,
        'scale_t': 1.0,
        'seq_weight': 0.0
    }
    
    score = calculate_ot_score(coords1, masses1, coords2, masses2, config)
    # Identical clouds should have near-zero score
    assert score >= 0
    assert score < 0.1

def test_sequence_penalty():
    # Historical: Event A then Event B
    coords_hist = np.array([[10.0, 0.0], [20.0, 1.0]])
    masses_hist = np.array([1.0, 1.0])
    
    # Simulation 1: Matches order (Event A' then Event B')
    coords_sim_ordered = np.array([[11.0, 0.1], [21.0, 1.1]])
    masses_sim_ordered = np.array([1.0, 1.0])
    
    # Simulation 2: Inverted order (Event B' then Event A')
    # Note: OT might still match them based on proximity, but if we swap them in the array,
    # and they are far enough that they match 1->1 and 2->2, but the 2nd is earlier in time...
    # Wait, the penalty is based on the index in the array.
    # coords2 is usually sorted by time in the find_best_sequence.
    
    # Let's say:
    # Hist: index 0 (time 0), index 1 (time 1)
    # Sim: index 0 (time 0.1), index 1 (time 1.1)
    
    config = {
        'reg': 0.01, # Low regularization to ensure strict matching
        'reg_m': 100.0,
        'seq_weight': 10.0
    }
    
    score_ordered = calculate_ot_score(coords_hist, masses_hist, coords_sim_ordered, masses_sim_ordered, config)
    
    # Sim Inverted: index 0 is at time 1.1, index 1 is at time 0.1
    # But wait, the indices in sim_indices = np.arange(len(coords2)) are just 0, 1, 2...
    # The penalty uses expected_sim_indices = P @ sim_indices / row_sums.
    # If Hist[0] matches Sim[1] and Hist[1] matches Sim[0], then:
    # expected_sim_indices = [1, 0]
    # index_steps = [0 - 1] = [-1]
    # inversions = [1]
    # penalty = 1 * seq_weight = 10.0
    
    coords_sim_inverted = np.array([[21.0, 1.1], [11.0, 0.1]])
    masses_sim_inverted = np.array([1.0, 1.0])
    
    score_inverted = calculate_ot_score(coords_hist, masses_hist, coords_sim_inverted, masses_sim_inverted, config)
    
    # The inverted score should be significantly higher due to the penalty
    assert score_inverted > score_ordered + 5.0

def test_get_transport_plan():
    coords1 = np.array([[10.0, 0.0], [20.0, 1.0]])
    masses1 = np.array([1.0, 1.0])
    coords2 = np.array([[11.0, 0.1], [21.0, 1.1]])
    masses2 = np.array([1.0, 1.0])
    
    config = {
        'reg': 0.1,
        'reg_m': 10.0
    }
    
    P = get_transport_plan(coords1, masses1, coords2, masses2, config)
    
    assert P.shape == (2, 2)
    assert np.all(P >= 0)
    # Total mass should be roughly equal to the input masses (it's unbalanced, so sum(P) is approx sum(masses))
    assert np.abs(np.sum(P) - 2.0) < 0.2

def test_zero_mass_protection():
    # In prepare_event_data, we now filter. 
    # calculate_ot_score should handle empty inputs gracefully (returns inf)
    coords1 = np.array([]).reshape(0, 2)
    masses1 = np.array([])
    coords2 = np.array([[10.0, 0.0]])
    masses2 = np.array([1.0])
    
    config = {'reg': 0.1, 'reg_m': 10.0}
    score = calculate_ot_score(coords1, masses1, coords2, masses2, config)
    assert score == np.inf

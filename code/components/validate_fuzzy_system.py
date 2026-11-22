from pathlib import Path
import argparse
import csv
import hashlib
import json
from itertools import product
import numpy as np

from fuzzy_system import (SLICE_NAMES, WEIGHTS, ALLOWED_ACTIONS, membership_degrees,
    plan_score, select_plan, allowed_masks, gate_actions, reward_components,
    crisp_components, fuzzy_reward, crisp_reward, slice_weights, rule_records,
    export_specification)


def validate(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    export_specification(output)
    checks = []

    rules = rule_records()
    keys = [tuple(r[k] for k in ("load", "latency_requirement", "reliability_requirement", "isolation", "fault")) for r in rules]
    assert len(rules) == len(set(keys)) == 162
    checks.append({"check": "complete_unique_rule_antecedents", "count": len(rules), "status": "PASS"})

    centers = np.asarray(list(product([0., .5, 1.], repeat=4)))
    center_rows = []
    for fault in [False, True]:
        score = np.asarray(plan_score(*centers.T, fault=fault))
        categories = select_plan(score, fault=fault)
        for i, row in enumerate(centers):
            expected = next(r for r in rules if tuple(r[k] for k in ("load", "latency_requirement", "reliability_requirement", "isolation", "fault")) ==
                            tuple([("Low", "Medium", "High")[[0., .5, 1.].index(v)] for v in row] + [int(fault)]))
            assert abs(score[i] - expected["score"]) < 1e-12
            assert categories[i] == expected["plan"]
            if fault:
                assert categories[i] == "REJECT" and score[i] == 0.
            center_rows.append({"load": row[0], "latency_requirement": row[1],
                                "reliability_requirement": row[2], "isolation": row[3],
                                "fault": int(fault), "score": float(score[i]), "plan": categories[i]})
    checks.append({"check": "all_162_linguistic_centers", "status": "PASS"})


    grid = np.asarray(list(product(np.linspace(0., 1., 11), repeat=4)))
    denominator = np.prod(np.stack([membership_degrees(v).sum(axis=-1) for v in grid.T], axis=-1), axis=-1)
    assert np.allclose(denominator, 1.0, atol=1e-14, rtol=0.)
    ordinary = np.asarray(plan_score(*grid.T, fault=False))
    faulted = np.asarray(plan_score(*grid.T, fault=True))
    assert np.all(np.isfinite(ordinary)) and np.all((ordinary >= 0.) & (ordinary <= 100.))
    assert np.all(faulted == 0.)
    checks.append({"check": "complete_domain_grid", "points_per_fault_state": len(grid),
                   "total_points": 2 * len(grid), "minimum_activation_denominator": float(denominator.min()),
                   "maximum_activation_denominator": float(denominator.max()),
                   "minimum_no_fault_score": float(ordinary.min()), "maximum_no_fault_score": float(ordinary.max()),
                   "status": "PASS"})

    representatives = [
        ((0., 0., 1., 1., False), "IMMEDIATE"),
        ((.5, 0., 1., .5, False), "IMMEDIATE"),
        ((1., 0., 1., 1., False), "OPTIMIZE"),
        ((.5, .5, .5, .5, False), "OPTIMIZE"),
        ((1., .5, .5, 0., False), "DEFER"),
        ((.5, .5, .5, .5, True), "REJECT"),
    ]
    correspondence=[]
    for args, expected in representatives:
        score = plan_score(*args)
        observed = select_plan(score, fault=args[-1])
        assert observed == expected
        correspondence.append({"inputs": args, "expected_plan": expected, "observed_plan": observed, "score": score})
    checks.append({"check": "six_original_representative_rules", "rules": correspondence, "status": "PASS"})

    category_boundaries = [(0., "REJECT"), (30., "REJECT"), (np.nextafter(30., np.inf), "DEFER"),
                           (50., "DEFER"), (np.nextafter(50., np.inf), "OPTIMIZE"),
                           (76., "OPTIMIZE"), (np.nextafter(76., np.inf), "IMMEDIATE"), (100., "IMMEDIATE")]
    for value, expected in category_boundaries:
        assert select_plan(value) == expected
        assert select_plan(value, fault=True) == "REJECT"
    checks.append({"check": "closed_score_boundaries_and_fault_override", "cases": 16, "status": "PASS"})


    extremes = np.array([-1e300, -1., 0., np.nextafter(0., 1.),
                         np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.),
                         np.nextafter(1., 0.), 1., 2., 1e300])
    memberships = membership_degrees(extremes)
    assert np.all(np.isfinite(memberships)) and np.all((memberships >= 0.) & (memberships <= 1.))
    assert np.allclose(memberships.sum(axis=-1), 1., atol=1e-14, rtol=0.)
    assert np.array_equal(memberships[0], memberships[2])
    assert np.array_equal(memberships[-1], memberships[-3])
    for value in [0., .5, 1.]:
        left = membership_degrees(np.nextafter(value, -np.inf))
        right = membership_degrees(np.nextafter(value, np.inf))
        assert np.max(abs(left-right)) < 1e-12
    checks.append({"check": "membership_saturation_partition_and_continuity", "status": "PASS"})

    rewards = []

    values=np.asarray(list(product([0., .9, 1., 1.5, 2.], repeat=4)))
    for name in SLICE_NAMES:
        assert np.isclose(WEIGHTS[name].sum(), 1.)
        for function in [fuzzy_reward, crisp_reward]:
            score=np.asarray(function(*values.T, slice_type=name))
            assert np.all(np.isfinite(score)) and np.all((score >= 0.) & (score <= 1.+1e-12))
            rewards.append({"slice": name, "reward": function.__name__, "grid_points": len(values),
                            "minimum": float(score.min()), "maximum": float(score.max())})
        for is_on in [True,False]:
            assert np.isclose(slice_weights(name, is_on).sum(),1.)
        expected={"eMBB":[.4,.4,.2], "uRLLC":[.6,.3,.1], "mMTC":[.3,.5,.2]}[name]
        assert np.allclose(slice_weights(name,False)[:3],expected,atol=1e-14)
    checks.append({"check": "bounded_reward_grid_and_weight_normalization", "results": rewards, "status": "PASS"})


    isolation_effect=[]
    for name in SLICE_NAMES:
        full_low=fuzzy_reward(.8,1.,.8,0.,slice_type=name)
        full_high=fuzzy_reward(.8,1.,.8,1.,slice_type=name)
        off_low=fuzzy_reward(.8,1.,.8,0.,slice_type=name,isolation_enabled=False)
        off_high=fuzzy_reward(.8,1.,.8,1.,slice_type=name,isolation_enabled=False)
        assert abs((full_high-full_low)-WEIGHTS[name][3])<1e-12
        assert abs(off_low-off_high)<1e-12
        isolation_effect.append({"slice":name,"full_isolation_low":full_low,"full_isolation_high":full_high,
                                 "difference":full_high-full_low,"no_isolation_difference":off_high-off_low})
    checks.append({"check": "nonzero_isolation_reward_contribution", "results":isolation_effect,"status":"PASS"})

    masks=allowed_masks(list(ALLOWED_ACTIONS))
    assert np.all(masks.sum(axis=-1)>=1)
    gate_cases=[]
    for plan,allowed in ALLOWED_ACTIONS.items():
        for action in range(4):
            data=gate_actions(action,plan,return_details=True)
            assert data['executed'] in allowed
            assert data['replaced']==(action not in allowed)
            if action in allowed:assert data['executed']==action
            gate_cases.append({"plan":plan,"proposed":action,"executed":data['executed'],"replaced":data['replaced']})
    for category in select_plan(ordinary):
        assert allowed_masks(category).any()
    rng_a=np.random.default_rng(39);rng_b=np.random.default_rng(39)
    aa=[gate_actions(1,"REJECT",mode="random",rng=rng_a) for _ in range(100)]
    bb=[gate_actions(1,"REJECT",mode="random",rng=rng_b) for _ in range(100)]
    assert aa==bb and set(aa)=={0,2}
    checks.append({"check":"component_plan_masks_and_seeded_reference_override", "cases":gate_cases,"status":"PASS"})

    batch=fuzzy_reward(np.array([.8,1.2,2.]),np.ones(3),np.ones(3),np.ones(3),slice_type=SLICE_NAMES)
    individual=np.array([fuzzy_reward(lat,1.,1.,1.,slice_type=s) for lat,s in zip([.8,1.2,2.],SLICE_NAMES)])
    assert np.allclose(batch,individual,atol=1e-14)

    center_plans=np.array([r['plan'] for r in center_rows])
    action_grid=np.tile(np.arange(4),(len(center_plans),1))
    plan_grid=np.repeat(center_plans[:,None],4,axis=1)
    executed=gate_actions(action_grid,plan_grid)
    gate_masks=allowed_masks(plan_grid)
    assert np.all(np.take_along_axis(gate_masks,executed[...,None],axis=-1))
    checks.append({"check":"numpy_broadcast_and_all_648_center_action_pairs", "status":"PASS"})

    invalid_count=0
    for invalid in [np.nan,np.inf,-np.inf]:
        for function,args in [(membership_degrees,(invalid,)),(plan_score,(invalid,.5,.5,.5)),
                              (fuzzy_reward,(invalid,1.,1.,1.)),(crisp_reward,(1.,invalid,1.,1.))]:
            try:function(*args)
            except ValueError:invalid_count+=1
            else:raise AssertionError(f'{function.__name__} accepted non-finite input')
    for function,args in [(plan_score,(.5,.5,.5,.5,.3)),(gate_actions,(4,'OPTIMIZE')),
                          (allowed_masks,('UNKNOWN',)),(fuzzy_reward,(1.,1.,1.,1.,'UNKNOWN'))]:
        try:function(*args)
        except ValueError:invalid_count+=1
        else:raise AssertionError(f'{function.__name__} accepted invalid input')
    checks.append({"check":"invalid_inputs_rejected", "cases":invalid_count,"status":"PASS"})


    with (output/'all_162_center_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(center_rows[0]));writer.writeheader();writer.writerows(center_rows)
    result={"status":"PASS","checks":checks,
            "implementation_sha256":hashlib.sha256(Path(__file__).with_name('fuzzy_system.py').read_bytes()).hexdigest(),
            "validation_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope":"Complete rule coverage, finite bounded inference, reward isolation contribution and reference action-mask semantics. Controller-specific performance and the isolation ablation are measured separately by the training campaign."}
    (output/'fuzzy_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({"status":"PASS","checks":len(checks),"rule_centers":162,"coverage_grid_points":2*len(grid),"center_action_pairs":648},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Exhaustive component validation without training or fitted performance data.')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'validation/fuzzy')
    args=parser.parse_args();validate(args.output)

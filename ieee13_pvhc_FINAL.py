# =============================================================================
# IEEE 13-Bus Distribution Feeder — PVHC Analysis
# EEE 306 Group 04
# Tool: pandapower (Backward-Forward Sweep)
# =============================================================================

import pandapower as pp
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

configs = {
    '601': (0.3465 / 1.60934,  1.0179 / 1.60934),
    '602': (0.7526 / 1.60934,  1.1814 / 1.60934),
    '603': (1.3294 / 1.60934,  1.3471 / 1.60934),
    '604': (1.3294 / 1.60934,  1.3471 / 1.60934),
    '605': (1.3292 / 1.60934,  1.3475 / 1.60934),
    '606': (0.7982 / 1.60934,  0.4463 / 1.60934),
    '607': (1.3425 / 1.60934,  0.5124 / 1.60934),
}

# -----------------------------------------------------------------------
# FIXED: these ampacities were arbitrary and did NOT match the real
# conductors used in the official IEEE 13-bus test feeder (Kersting,
# "Radial Distribution Test Feeders", IEEE DSA Subcommittee, Table 3 -
# Conductor Data + Configuration tables). That mismatch — not a real
# thermal problem — was the source of the >100% ("150%") line loading.
#
#   Config | Real conductor              | Real ampacity | Old (wrong) value
#   -------|------------------------------|---------------|--------------------
#    601   | 556,500 26/7 ACSR            |     730 A     |   400 A
#    602   | 4/0 ACSR                     |     340 A     |   400 A
#    603   | 1/0 ACSR                     |     230 A     |   200 A
#    604   | 1/0 ACSR                     |     230 A     |   200 A
#    605   | 1/0 ACSR                     |     230 A     |   200 A
#    606   | 250,000 AA underground cable |     260 A     |   350 A
#    607   | 1/0 AA underground cable     |     165 A     |   300 A
# -----------------------------------------------------------------------
config_ratings_ka = {
    '601': 0.730,
    '602': 0.340,
    '603': 0.230,
    '604': 0.230,
    '605': 0.230,
    '606': 0.260,
    '607': 0.165,
}

# =============================================================================
# VOLTAGE REGULATOR (RG-1) — models the real feeder's substation regulator,
# which sits between bus 650 and bus 632 on the official IEEE 13-bus test
# feeder (Kersting report, "Regulator Data": Line Segment 650-632) but was
# missing from the original pandapower model.
#
# IMPORTANT CORRECTION — read this before touching REGULATOR_TAP_POS:
#   The >100% (up to ~150%) loading previously seen on line 650-632 (and,
#   downstream, on 632-671) was NOT a real thermal problem. It was caused
#   by `config_ratings_ka` using the wrong current limits (see the fix
#   just above). Config 601 is the 556,500 26/7 ACSR main trunk conductor,
#   rated 730 A in the real feeder — the model was using only 400 A, i.e.
#   less than 55% of the real rating. With the ratings corrected, line
#   650-632 sits at roughly 70-75% loading in the base case with NO
#   regulator needed at all to clear the "overload".
#
#   The regulator is still worth keeping, for two genuine reasons:
#     1. It matches the official feeder topology — RG-1 is a real
#        component of the IEEE 13-bus test feeder, not something added
#        just to patch a bug.
#     2. It's a step-VOLTAGE regulator: it raises voltage, it does NOT
#        add current-carrying capacity. It's the right tool for improving
#        the voltage profile (more headroom for the Case 1 PVHC sweep and
#        the Case 3 SVC sweep at buses like 675/611), but it was never the
#        right tool for clearing a line-loading violation — that always
#        had to come from correcting the conductor rating.
#
#   tap_pos=4 (a modest +2.5% boost) is kept as a sensible default: it
#   improves the voltage profile without pushing any bus close to the
#   1.05 pu ceiling. After any tap change, re-run run_base_case() to
#   confirm max system voltage is still within limits.
# =============================================================================
REGULATOR_TAP_POS    = 4      # each step = 0.625% boost; range -16..+16 (±10%)
REGULATOR_STEP_PCT   = 0.625


def build_ieee13():
    net = pp.create_empty_network(f_hz=60, sn_mva=1.0)

    b650  = pp.create_bus(net, vn_kv=4.16, name='650')
    b650r = pp.create_bus(net, vn_kv=4.16, name='650r')  # regulator output bus
    b632 = pp.create_bus(net, vn_kv=4.16, name='632')
    b633 = pp.create_bus(net, vn_kv=4.16, name='633')
    b634 = pp.create_bus(net, vn_kv=0.48, name='634')
    b645 = pp.create_bus(net, vn_kv=4.16, name='645')
    b646 = pp.create_bus(net, vn_kv=4.16, name='646')
    b671 = pp.create_bus(net, vn_kv=4.16, name='671')
    b680 = pp.create_bus(net, vn_kv=4.16, name='680')
    b684 = pp.create_bus(net, vn_kv=4.16, name='684')
    b652 = pp.create_bus(net, vn_kv=4.16, name='652')
    b692 = pp.create_bus(net, vn_kv=4.16, name='692')
    b675 = pp.create_bus(net, vn_kv=4.16, name='675')
    b611 = pp.create_bus(net, vn_kv=4.16, name='611')

    bus_map = {
        '650': b650, '632': b632, '633': b633, '634': b634,
        '645': b645, '646': b646, '671': b671, '680': b680,
        '684': b684, '652': b652, '692': b692, '675': b675,
        '611': b611
        # NOTE: '650r' (regulator output bus) deliberately NOT included —
        # it's an internal modelling bus, not a valid PV/SVC test location.
    }

    pp.create_ext_grid(net, bus=b650, vm_pu=1.0, name='Slack @ 650')

    # Voltage regulator RG-1 (650 -> 650r): a near-ideal, high-capacity
    # 4.16/4.16 kV transformer whose only job is to hold a fixed tap boost.
    # sn_mva is set large (5 MVA, well above the ~3.7 MVA feeder total) so
    # the regulator itself is never the bottleneck.
    pp.create_transformer_from_parameters(
        net,
        hv_bus=b650, lv_bus=b650r,
        sn_mva=5.0,
        vn_hv_kv=4.16, vn_lv_kv=4.16,
        vkr_percent=0.01, vk_percent=0.1,
        pfe_kw=0, i0_percent=0,
        tap_side='lv', tap_neutral=0,
        tap_min=-16, tap_max=16,
        tap_step_percent=REGULATOR_STEP_PCT,
        tap_pos=REGULATOR_TAP_POS,
        tap_changer_type='Ratio',
        name='Voltage Regulator RG-1 (650-632)'
    )

    pp.create_transformer_from_parameters(
        net,
        hv_bus=b633, lv_bus=b634,
        sn_mva=0.5,
        vn_hv_kv=4.16, vn_lv_kv=0.48,
        vkr_percent=1.1, vk_percent=2.0,
        pfe_kw=0, i0_percent=0,
        name='XFM-1'
    )

    def add_line(from_bus, to_bus, length_ft, config, name):
        length_km = max(length_ft * 0.0003048, 0.01)
        r, x = configs[config]
        imax  = config_ratings_ka[config]
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus, to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=r,
            x_ohm_per_km=x,
            c_nf_per_km=0,
            max_i_ka=imax,
            name=name
        )

    add_line(b650r, b632, 2000, '601', '650-632')   # now fed FROM the regulator
    add_line(b632, b633,  500, '602', '632-633')
    add_line(b632, b645,  500, '603', '632-645')
    add_line(b632, b671, 2000, '601', '632-671')
    add_line(b645, b646,  300, '603', '645-646')
    add_line(b671, b684,  300, '604', '671-684')
    add_line(b671, b680, 1000, '601', '671-680')
    add_line(b671, b692,   33, '601', '671-692')
    add_line(b684, b652,  800, '607', '684-652')
    add_line(b684, b611,  300, '605', '684-611')
    add_line(b692, b675,  500, '606', '692-675')

    pp.create_load(net, bus=b634, p_mw=0.400,  q_mvar=0.290,  name='Load 634')
    pp.create_load(net, bus=b645, p_mw=0.170,  q_mvar=0.125,  name='Load 645')
    pp.create_load(net, bus=b646, p_mw=0.230,  q_mvar=0.132,  name='Load 646')
    pp.create_load(net, bus=b652, p_mw=0.128,  q_mvar=0.086,  name='Load 652')
    pp.create_load(net, bus=b671, p_mw=1.155,  q_mvar=0.660,  name='Load 671')
    pp.create_load(net, bus=b675, p_mw=0.843,  q_mvar=0.462,  name='Load 675')
    pp.create_load(net, bus=b692, p_mw=0.170,  q_mvar=0.151,  name='Load 692')
    pp.create_load(net, bus=b611, p_mw=0.170,  q_mvar=0.080,  name='Load 611')
    pp.create_load(net, bus=b671, p_mw=0.200,  q_mvar=0.116,  name='Dist Load 632-671')
    pp.create_load(net, bus=b680, p_mw=0.001,  q_mvar=0.000,  name='Dummy 680')

    pp.create_shunt(net, bus=b675, q_mvar=-0.600, p_mw=0, name='Cap 675')
    pp.create_shunt(net, bus=b611, q_mvar=-0.100, p_mw=0, name='Cap 611')

    return net, bus_map


def violation_exists(net, base_line_loading_pct, v_max=1.05):
    if net.res_bus['vm_pu'].max() > v_max:
        return True
    current_loading = net.res_line['loading_percent'].values
    for i in range(len(current_loading)):
        if current_loading[i] > 100.0 and current_loading[i] > base_line_loading_pct[i] + 0.5:
            return True
    return False


def _apply_intervention(net, bus_map, intervention):
    if intervention == 'svc':
        pp.create_shunt(net, bus=bus_map['675'],
                        q_mvar=-0.500, p_mw=0, name='SVC @ 675')
    elif intervention == 'shunt_cap':
        pp.create_shunt(net, bus=bus_map['675'],
                        q_mvar=-0.300, p_mw=0, name='Extra Cap @ 675')
    elif intervention == 'line_length':
        for idx in net.line.index:
            net.line.at[idx, 'length_km'] *= 1.20


def run_base_case():
    print("=" * 60)
    print("BASE CASE LOAD FLOW (No PV)")
    print("=" * 60)

    net, _ = build_ieee13()

    try:
        pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50)
    except Exception as e:
        print(f"  Load flow failed: {e}")
        return

    if not net.converged:
        print("  Load flow did not converge.")
        return

    print("  Converged successfully.\n")
    print(f"  {'Bus':<6} {'V (pu)':>10} {'Angle (deg)':>12}")
    print("  " + "-" * 32)

    for idx in net.res_bus.index:
        name = net.bus.at[idx, 'name']
        vm   = net.res_bus.at[idx, 'vm_pu']
        va   = net.res_bus.at[idx, 'va_degree']
        print(f"  {name:<6} {vm:>10.4f} {va:>12.4f}")

    print()
    print(f"  Max voltage      : {net.res_bus['vm_pu'].max():.4f} pu")
    print(f"  Min voltage      : {net.res_bus['vm_pu'].min():.4f} pu")
    print(f"  Max line loading : {net.res_line['loading_percent'].max():.2f}%")
    print(f"  Regulator loading: {net.res_trafo.loc[net.trafo.name == 'Voltage Regulator RG-1 (650-632)', 'loading_percent'].values[0]:.2f}%"
          f"  (tap_pos={REGULATOR_TAP_POS}, boost=+{REGULATOR_TAP_POS*REGULATOR_STEP_PCT:.2f}%)")
    if net.res_line['loading_percent'].max() > 100.0:
        print(f"  NOTE: a line is still showing >100% loading. This now points to a")
        print(f"        genuine thermal issue (or a data entry mistake) rather than the")
        print(f"        config_ratings_ka bug, which has already been corrected to the")
        print(f"        real IEEE-13 conductor ampacities. See the comment above")
        print(f"        build_ieee13() for reference values.")
    else:
        print(f"  All lines within thermal limits in the base case (ratings now match")
        print(f"        the real IEEE-13 conductor ampacities).")
    print()


def run_case1():
    print("=" * 60)
    print("CASE 1: PVHC AT EACH BUS (No Compensation)")
    print("=" * 60)

    load_buses = ['632', '633', '634', '645', '646',
                  '671', '680', '684', '652', '692', '675', '611']
    pv_step_mw = 0.05
    pv_max_mw  = 10.0

    net_base, _ = build_ieee13()
    pp.runpp(net_base, algorithm='bfsw', numba=False, max_iteration=50)
    base_line_loading = net_base.res_line['loading_percent'].values.copy()
    base_min_v        = net_base.res_bus['vm_pu'].min()
    base_max_v        = net_base.res_bus['vm_pu'].max()
    base_max_loading  = net_base.res_line['loading_percent'].max()

    print(f"  Base case: min V = {base_min_v:.4f} pu  "
          f"max V = {base_max_v:.4f} pu  "
          f"max line loading = {base_max_loading:.2f}%")
    print(f"  (Bus 650 excluded — slack/substation bus, not a valid PV candidate)")
    print(f"  (Pre-existing overloads/undervoltage are NOT counted as violations)")
    print()

    pvhc_results = {}

    for bus_name in load_buses:
        pvhc = 0.0
        pv   = pv_step_mw

        while pv <= pv_max_mw:
            net, bus_map = build_ieee13()
            target_bus   = bus_map[bus_name]

            pp.create_sgen(net, bus=target_bus, p_mw=pv, q_mvar=0.0,
                           name=f'PV @ {bus_name}')

            try:
                pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50)
            except Exception:
                break

            if not net.converged:
                break

            if violation_exists(net, base_line_loading):
                break

            pvhc = pv
            pv   = round(pv + pv_step_mw, 4)

        pvhc_results[bus_name] = pvhc
        print(f"  Bus {bus_name:>4s}  →  PVHC = {pvhc:.3f} MW")

    best_bus  = max(pvhc_results, key=pvhc_results.get)
    best_pvhc = pvhc_results[best_bus]

    print()
    print(f"  ★  Best bus : {best_bus}")
    print(f"     Highest PVHC : {best_pvhc:.3f} MW")
    print()

    return pvhc_results, best_bus, best_pvhc


def pvhc_single_bus(bus_name, intervention=None,
                    pv_step_mw=0.05, pv_max_mw=10.0):

    net_base, bus_map_base = build_ieee13()
    _apply_intervention(net_base, bus_map_base, intervention)
    pp.runpp(net_base, algorithm='bfsw', numba=False, max_iteration=50)
    base_line_loading = net_base.res_line['loading_percent'].values.copy()

    pvhc = 0.0
    pv   = pv_step_mw

    while pv <= pv_max_mw:
        net, bus_map = build_ieee13()
        _apply_intervention(net, bus_map, intervention)

        pp.create_sgen(net, bus=bus_map[bus_name], p_mw=pv, q_mvar=0.0,
                       name=f'PV @ {bus_name}')

        try:
            pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50)
        except Exception:
            break

        if not net.converged:
            break

        if violation_exists(net, base_line_loading):
            break

        pvhc = pv
        pv   = round(pv + pv_step_mw, 4)

    return pvhc


def run_case2(best_bus, base_pvhc):
    print("=" * 60)
    print(f"CASE 2: INTERVENTIONS AT BUS {best_bus}")
    print("=" * 60)

    interventions = {
        'No Compensation (baseline)':   None,
        'SVC at Bus 675':               'svc',
        'Shunt Capacitor at Bus 675':   'shunt_cap',
        'Line Length +20% (uniform)':   'line_length',
    }

    results = {}
    for label, code in interventions.items():
        pvhc        = pvhc_single_bus(best_bus, intervention=code)
        improvement = pvhc - base_pvhc
        results[label] = pvhc
        print(f"  {label:<45s}  PVHC = {pvhc:.3f} MW  "
              f"(Δ = {improvement:+.3f} MW)")

    intervention_only = {k: v for k, v in results.items()
                         if k != 'No Compensation (baseline)'}
    best_label = max(intervention_only, key=intervention_only.get)
    best_val   = intervention_only[best_label]

    print()
    print(f"  ★  Best intervention : {best_label}")
    print(f"     PVHC improved from {base_pvhc:.3f} MW  →  {best_val:.3f} MW")
    print()

    return results


def run_case2_lowest(pvhc_results, base_pvhc_best):
    worst_bus  = min(pvhc_results, key=pvhc_results.get)
    worst_pvhc = pvhc_results[worst_bus]

    print("=" * 60)
    print(f"CASE 2 (LOWEST PVHC BUS): INTERVENTIONS AT BUS {worst_bus}")
    print("=" * 60)

    interventions = {
        'No Compensation (baseline)':   None,
        'SVC at Bus 675':               'svc',
        'Shunt Capacitor at Bus 675':   'shunt_cap',
        'Line Length +20% (uniform)':   'line_length',
    }

    results = {}
    for label, code in interventions.items():
        pvhc        = pvhc_single_bus(worst_bus, intervention=code)
        improvement = pvhc - worst_pvhc
        results[label] = pvhc
        print(f"  {label:<45s}  PVHC = {pvhc:.3f} MW  "
              f"(Δ = {improvement:+.3f} MW)")

    intervention_only = {k: v for k, v in results.items()
                         if k != 'No Compensation (baseline)'}
    best_label = max(intervention_only, key=intervention_only.get)
    best_val   = intervention_only[best_label]

    print()
    print(f"  ★  Best intervention : {best_label}")
    print(f"     PVHC improved from {worst_pvhc:.3f} MW  →  {best_val:.3f} MW")
    print()

    return worst_bus, results


# =============================================================================
# CASE 3 (NEW) — SVC REACTIVE POWER (NEGATIVE VAR) IMPACT STUDY
#
# Teacher's ask: if we add an SVC (modelled as a shunt injecting NEGATIVE
# Mvar, i.e. capacitive/leading reactive power support) at each bus one at
# a time, what happens to the network's performance?
#
# Method (mirrors Case 1's step-sweep logic, but sweeps Q instead of P):
#   - At each of the 12 load buses, inject capacitive Mvar in small steps
#     (q_mvar = -q) using pp.create_shunt.
#   - Keep increasing the magnitude until a violation occurs (voltage > 1.05
#     pu, or a line becomes newly overloaded beyond the base case) — the
#     last safe value is that bus's "SVC Hosting Capacity" (how much
#     reactive support that bus can absorb before it becomes a problem).
#   - Separately, apply each bus's own hosting-capacity SVC and record the
#     resulting change in minimum system voltage and maximum line loading,
#     compared to the base case — this is the "performance impact".
#   - The bus with the largest SVC hosting capacity is the BEST bus for SVC
#     placement; the bus with the smallest is the WORST bus.
# =============================================================================

def svc_hosting_capacity_single_bus(bus_name, q_step_mvar=0.05, q_max_mvar=5.0):
    """Sweep capacitive (negative) Mvar at one bus and return the largest
    magnitude that does not trigger a voltage or current violation."""

    net_base, _ = build_ieee13()
    pp.runpp(net_base, algorithm='bfsw', numba=False, max_iteration=50)
    base_line_loading = net_base.res_line['loading_percent'].values.copy()

    svc_hc = 0.0
    q      = q_step_mvar

    while q <= q_max_mvar:
        net, bus_map = build_ieee13()
        target_bus   = bus_map[bus_name]

        pp.create_shunt(net, bus=target_bus, q_mvar=-q, p_mw=0,
                        name=f'SVC @ {bus_name}')

        try:
            pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50)
        except Exception:
            break

        if not net.converged:
            break

        if violation_exists(net, base_line_loading):
            break

        svc_hc = q
        q      = round(q + q_step_mvar, 4)

    return svc_hc


def run_case3():
    print("=" * 60)
    print("CASE 3: SVC (NEGATIVE VAR) IMPACT AT EACH BUS")
    print("=" * 60)

    load_buses   = ['632', '633', '634', '645', '646',
                    '671', '680', '684', '652', '692', '675', '611']
    q_step_mvar  = 0.05
    q_max_mvar   = 5.0

    net_base, _ = build_ieee13()
    pp.runpp(net_base, algorithm='bfsw', numba=False, max_iteration=50)
    base_min_v       = net_base.res_bus['vm_pu'].min()
    base_max_v       = net_base.res_bus['vm_pu'].max()
    base_max_loading = net_base.res_line['loading_percent'].max()

    # Which bus is the weakest (lowest voltage) BEFORE any SVC is added.
    base_min_v_idx = net_base.res_bus['vm_pu'].idxmin()
    base_min_v_bus = net_base.bus.at[base_min_v_idx, 'name']

    print(f"  Base case: min V = {base_min_v:.4f} pu (at Bus {base_min_v_bus})  "
          f"max V = {base_max_v:.4f} pu  "
          f"max line loading = {base_max_loading:.2f}%")
    print(f"  (Bus 650 excluded — slack/substation bus)")
    print(f"  (SVC modelled as a shunt injecting negative Mvar, i.e. capacitive support)")
    print()

    # --- Step 1: SVC hosting capacity sweep at every bus -------------------
    svc_results = {}
    for bus_name in load_buses:
        svc_hc = svc_hosting_capacity_single_bus(bus_name, q_step_mvar, q_max_mvar)
        svc_results[bus_name] = svc_hc
        print(f"  Bus {bus_name:>4s}  →  SVC Hosting Capacity = {svc_hc:.3f} Mvar")

    # --- Step 2: performance impact of each bus's own hosting-capacity SVC -
    print()
    print(f"  {'Bus':<6}{'SVC(Mvar)':>10}{'MinV before':>13}{'@Bus':>6}"
          f"{'MinV after':>12}{'@Bus':>6}{'ΔMinV':>9}"
          f"{'MaxLoad b4':>12}{'MaxLoad after':>15}{'ΔLoad':>9}")
    print("  " + "-" * 108)

    impact_results = {}
    for bus_name, q in svc_results.items():
        if q <= 0:
            impact_results[bus_name] = {
                'svc_mvar': 0.0,
                'min_v_before': base_min_v, 'min_v_bus_before': base_min_v_bus,
                'min_v_after':  base_min_v, 'min_v_bus_after':  base_min_v_bus,
                'd_min_v': 0.0,
                'load_before': base_max_loading, 'load_after': base_max_loading,
                'd_load': 0.0,
            }
        else:
            net, bus_map = build_ieee13()
            pp.create_shunt(net, bus=bus_map[bus_name], q_mvar=-q, p_mw=0,
                            name=f'SVC @ {bus_name}')
            pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50)

            new_min_v_idx = net.res_bus['vm_pu'].idxmin()
            new_min_v     = net.res_bus.at[new_min_v_idx, 'vm_pu']
            new_min_v_bus = net.bus.at[new_min_v_idx, 'name']
            new_loading   = net.res_line['loading_percent'].max()

            impact_results[bus_name] = {
                'svc_mvar': q,
                'min_v_before': base_min_v, 'min_v_bus_before': base_min_v_bus,
                'min_v_after':  new_min_v,  'min_v_bus_after':  new_min_v_bus,
                'd_min_v': new_min_v - base_min_v,
                'load_before': base_max_loading, 'load_after': new_loading,
                'd_load': new_loading - base_max_loading,
            }

        r = impact_results[bus_name]
        print(f"  {bus_name:<6}{r['svc_mvar']:>10.3f}{r['min_v_before']:>13.4f}"
              f"{r['min_v_bus_before']:>6}{r['min_v_after']:>12.4f}"
              f"{r['min_v_bus_after']:>6}{r['d_min_v']:>+9.4f}"
              f"{r['load_before']:>12.2f}{r['load_after']:>15.2f}{r['d_load']:>+9.2f}")

    best_bus_svc  = max(svc_results, key=svc_results.get)
    worst_bus_svc = min(svc_results, key=svc_results.get)

    print()
    print(f"  ★  Best bus for SVC   : {best_bus_svc}  "
          f"(Hosting Capacity = {svc_results[best_bus_svc]:.3f} Mvar, "
          f"ΔMin V = {impact_results[best_bus_svc]['d_min_v']:+.4f} pu, "
          f"weakest bus was {base_min_v_bus} → now {impact_results[best_bus_svc]['min_v_bus_after']})")
    print(f"  ▼  Worst bus for SVC  : {worst_bus_svc}  "
          f"(Hosting Capacity = {svc_results[worst_bus_svc]:.3f} Mvar, "
          f"ΔMin V = {impact_results[worst_bus_svc]['d_min_v']:+.4f} pu, "
          f"weakest bus was {base_min_v_bus} → now {impact_results[worst_bus_svc]['min_v_bus_after']})")
    print()

    return svc_results, impact_results, best_bus_svc, worst_bus_svc, base_min_v_bus


def plot_case3_results(svc_results, impact_results, best_bus_svc, worst_bus_svc,
                       base_min_v_bus):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Chart A: SVC hosting capacity per bus ────────────────────────────
    ax1    = axes[0]
    buses  = list(svc_results.keys())
    values = list(svc_results.values())
    colors = ['#e74c3c' if b == best_bus_svc
              else '#e67e22' if b == worst_bus_svc
              else '#3498db' for b in buses]
    bars = ax1.bar(buses, values, color=colors,
                   edgecolor='white', linewidth=0.8)
    ax1.set_title('Case 3: SVC (Negative VAR) Hosting Capacity per Bus',
                  fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel('Bus', fontsize=10)
    ax1.set_ylabel('SVC Capacity (Mvar)', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, values):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    ax1.annotate(f'★ Best: Bus {best_bus_svc}',
                 xy=(0.98, 0.97), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 color='#e74c3c', fontweight='bold')
    ax1.annotate(f'▼ Worst: Bus {worst_bus_svc}',
                 xy=(0.98, 0.89), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 color='#e67e22', fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.2 + 0.2)
    ax1.grid(axis='y', alpha=0.3)

    # ── Chart B: voltage-support impact (ΔMin V) per bus ─────────────────
    ax2      = axes[1]
    d_min_v      = [impact_results[b]['d_min_v'] for b in buses]
    weakest_after = [impact_results[b]['min_v_bus_after'] for b in buses]
    colors2  = ['#e74c3c' if b == best_bus_svc
                else '#e67e22' if b == worst_bus_svc
                else '#2ecc71' for b in buses]
    bars2 = ax2.bar(buses, d_min_v, color=colors2,
                    edgecolor='white', linewidth=0.8)
    ax2.set_title(f'Case 3: Voltage-Support Impact (ΔMin V) of Each Bus\'s SVC\n'
                  f'(system-wide weakest bus BEFORE any SVC: Bus {base_min_v_bus})',
                  fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel('Bus (where SVC is installed)', fontsize=10)
    ax2.set_ylabel('Δ Min System Voltage (pu)', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    for bar, val, wk in zip(bars2, d_min_v, weakest_after):
        # value label
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.0005 if val >= 0 else -0.0015),
                 f'{val:+.4f}', ha='center',
                 va='bottom' if val >= 0 else 'top', fontsize=7)
        # which bus is now the system's weakest point, shown just above the x-axis
        ax2.text(bar.get_x() + bar.get_width() / 2, 0.002,
                 f'wk:{wk}', ha='center', va='bottom',
                 fontsize=6, color='#555555', rotation=90)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('svc_hosting_capacity_results.png', dpi=150, bbox_inches='tight')
    print("  Plot saved as svc_hosting_capacity_results.png")
    plt.show()


def plot_results(pvhc_results, case2_results, best_bus,
                 case2_lowest_results, worst_bus):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # ── Chart 1: Case 1 PVHC per bus ─────────────────────────────────────────
    ax1    = axes[0]
    buses  = list(pvhc_results.keys())
    values = list(pvhc_results.values())
    colors = ['#e74c3c' if b == best_bus
              else '#e67e22' if b == worst_bus
              else '#3498db' for b in buses]
    bars = ax1.bar(buses, values, color=colors,
                   edgecolor='white', linewidth=0.8)
    ax1.set_title('Case 1: PVHC at Each Bus (No Compensation)',
                  fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel('Bus', fontsize=10)
    ax1.set_ylabel('PVHC (MW)', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, values):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.05,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    ax1.annotate(f'★ Highest: Bus {best_bus}',
                 xy=(0.98, 0.97), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 color='#e74c3c', fontweight='bold')
    ax1.annotate(f'▼ Lowest: Bus {worst_bus}',
                 xy=(0.98, 0.89), xycoords='axes fraction',
                 ha='right', va='top', fontsize=9,
                 color='#e67e22', fontweight='bold')
    ax1.set_ylim(0, max(values) * 1.2 + 0.5)
    ax1.grid(axis='y', alpha=0.3)

    # ── Chart 2: Case 2 best bus intervention comparison ─────────────────────
    ax2     = axes[1]
    labels2 = list(case2_results.keys())
    vals2   = list(case2_results.values())
    colors2 = ['#95a5a6', '#e74c3c', '#f39c12', '#2ecc71']
    bars2   = ax2.bar(labels2, vals2,
                      color=colors2[:len(vals2)],
                      edgecolor='white', linewidth=0.8)
    ax2.set_title(f'Case 2: Interventions at Best Bus {best_bus}',
                  fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel('Intervention', fontsize=10)
    ax2.set_ylabel('PVHC (MW)', fontsize=10)
    ax2.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars2, vals2):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax2.set_ylim(0, max(vals2) * 1.2 + 0.5)
    ax2.grid(axis='y', alpha=0.3)

    # ── Chart 3: Case 2 worst bus intervention comparison ────────────────────
    ax3     = axes[2]
    labels3 = list(case2_lowest_results.keys())
    vals3   = list(case2_lowest_results.values())
    bars3   = ax3.bar(labels3, vals3,
                      color=colors2[:len(vals3)],
                      edgecolor='white', linewidth=0.8)
    ax3.set_title(f'Case 2: Interventions at Lowest Bus {worst_bus}',
                  fontsize=11, fontweight='bold', pad=12)
    ax3.set_xlabel('Intervention', fontsize=10)
    ax3.set_ylabel('PVHC (MW)', fontsize=10)
    ax3.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars3, vals3):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    ax3.set_ylim(0, max(vals3) * 1.2 + 0.5)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('pvhc_results.png', dpi=150, bbox_inches='tight')
    print("  Plot saved as pvhc_results.png")
    plt.show()


def plot_schematic():
    import pandapower.plotting as plot
    import matplotlib.pyplot as plt

    net, _ = build_ieee13()
    pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50)

    # NOTE: '650r' (the regulator's internal output bus, created inside
    # build_ieee13()) MUST get a coordinate here too. If any bus in net.bus
    # is left without one, pandapower's simple_plot() decides the geodata
    # is "incomplete" and tries to auto-generate coordinates for the WHOLE
    # bus table — but since the other buses already have manually-set
    # coordinates, that auto-generation refuses to run and raises
    # "Table bus is not empty - use overwrite=True to overwrite existing
    # geodata". Giving every bus a coordinate up front avoids that path
    # entirely.
    bus_coords = {
        '650': (4, 6), '650r': (4, 5.5), '632': (4, 5), '633': (6, 5), '634': (7, 5),
        '645': (2, 5), '646': (1, 5), '671': (4, 3), '680': (4, 1),
        '684': (3, 2), '652': (3, 1), '692': (6, 3), '675': (6, 1),
        '611': (2, 2),
    }

    for bus_name, (x, y) in bus_coords.items():
        idx = net.bus[net.bus['name'] == bus_name].index[0]
        net.bus.at[idx, 'geo'] = \
            f'{{"type": "Point", "coordinates": [{x}, {y}]}}'

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor('#f7f7f7')
    plot.simple_plot(net, ax=ax, plot_loads=True, plot_sgens=False,
                     bus_size=0.15, show_plot=False)
    for bus_name, (x, y) in bus_coords.items():
        if bus_name == '650r':
            continue  # internal regulator bus — plotted but not labeled
        ax.annotate(
            bus_name, xy=(x, y), xytext=(0, 14),
            textcoords='offset points', fontsize=10, fontweight='bold',
            color='darkblue', ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.8)
        )
    ax.set_title('IEEE 13-Bus Distribution Feeder — pandapower Model',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig('ieee13_schematic.png', dpi=150)
    print("  Schematic saved as ieee13_schematic.png")
    plt.show()

# =============================================================================
# CASE 4 (NEW) — DOES SVC (NEGATIVE VAR) MAGNITUDE AFFECT PVHC?
#
# Case 2 only tested ONE fixed SVC size (-0.5 Mvar at Bus 675) and found it
# can even slightly REDUCE PVHC at some buses. This case explains why, by
# sweeping the SVC size itself and tracking how PVHC at a chosen bus
# responds.
#
# Two competing effects are in play:
#   1. HELPS  : SVC raises voltage, undoing load-caused voltage drop -> more
#               room for PV to raise voltage further before hitting 1.05 pu.
#   2. HURTS  : SVC raises voltage even with ZERO PV -> it already uses up
#               part of the 1.05 pu headroom before any PV is added, leaving
#               LESS room for PV's own voltage rise.
#
# The result is often a hump-shaped curve: PVHC rises with SVC size at
# first, peaks, then falls as the SVC starts eating into the same voltage
# margin the PV needs.
# =============================================================================

def _apply_svc(net, bus_map, svc_bus, svc_mvar):
    """Install a capacitive shunt (SVC) of a chosen size at a chosen bus.
    svc_mvar is a magnitude (>=0); it is injected as q_mvar = -svc_mvar."""
    if svc_bus is not None and svc_mvar > 0:
        pp.create_shunt(net, bus=bus_map[svc_bus], q_mvar=-svc_mvar, p_mw=0,
                        name=f'SVC @ {svc_bus} ({svc_mvar:.3f} Mvar)')


def pvhc_with_svc(target_bus, svc_bus, svc_mvar,
                  pv_step_mw=0.05, pv_max_mw=10.0):
    """PVHC at target_bus, with a fixed-size SVC pre-installed at svc_bus.
    Returns None if the SVC alone (with zero PV) already fails to converge
    or already violates limits at this size -- i.e. this SVC size is itself
    infeasible, independent of any PV."""

    net_base, bus_map_base = build_ieee13()
    _apply_svc(net_base, bus_map_base, svc_bus, svc_mvar)

    try:
        pp.runpp(net_base, algorithm='bfsw', numba=False, max_iteration=50)
    except Exception:
        return None
    if not net_base.converged:
        return None

    base_line_loading = net_base.res_line['loading_percent'].values.copy()

    pvhc = 0.0
    pv   = pv_step_mw

    while pv <= pv_max_mw:
        net, bus_map = build_ieee13()
        _apply_svc(net, bus_map, svc_bus, svc_mvar)

        pp.create_sgen(net, bus=bus_map[target_bus], p_mw=pv, q_mvar=0.0,
                       name=f'PV @ {target_bus}')

        try:
            pp.runpp(net, algorithm='bfsw', numba=False, max_iteration=50)
        except Exception:
            break

        if not net.converged:
            break

        if violation_exists(net, base_line_loading):
            break

        pvhc = pv
        pv   = round(pv + pv_step_mw, 4)

    return pvhc


def svc_size_vs_pvhc(target_bus, svc_bus='675', q_max_mvar=None,
                     q_step_mvar=0.25):
    """Sweep SVC size at svc_bus (0 up to q_max_mvar) and record how PVHC
    at target_bus responds at each size. If q_max_mvar is not given, it
    defaults to that bus's own SVC hosting capacity (Case 3's result) so
    the sweep never wanders into an already-infeasible SVC size."""

    if q_max_mvar is None:
        q_max_mvar = svc_hosting_capacity_single_bus(svc_bus)
        if q_max_mvar <= 0:
            q_max_mvar = 1.0  # fallback so the sweep still runs a few points

    q_values = [round(q, 4) for q in np.arange(0, q_max_mvar + 1e-9, q_step_mvar)]

    results = {}
    print(f"  Sweeping SVC @ Bus {svc_bus} from 0 to {q_max_mvar:.2f} Mvar "
          f"(step {q_step_mvar} Mvar) -> PVHC @ Bus {target_bus}")
    for q in q_values:
        pvhc = pvhc_with_svc(target_bus, svc_bus, q)
        results[q] = pvhc
        pvhc_str = f"{pvhc:.3f} MW" if pvhc is not None else "INFEASIBLE (SVC itself violates limits)"
        print(f"    SVC = {q:>6.3f} Mvar  ->  PVHC = {pvhc_str}")
    print()

    return results


def run_case4():
    print("=" * 60)
    print("CASE 4: EFFECT OF SVC (NEGATIVE VAR) MAGNITUDE ON PVHC")
    print("=" * 60)
    print("  (Case 2 used one fixed SVC size; this sweeps SVC size itself)")
    print()

    # Case A: SVC always at Bus 675 (the weakest bus), PV at two different
    # buses -- this directly extends Case 2's best-bus / worst-bus buses.
    print("-- Case A: SVC @ Bus 675 (remote from PV bus) --")
    results_best_remote  = svc_size_vs_pvhc(target_bus='632', svc_bus='675')
    results_worst_remote = svc_size_vs_pvhc(target_bus='611', svc_bus='675')

    # Case B: SVC installed at the SAME bus as the PV (local compensation),
    # for comparison against the remote-SVC case above.
    print("-- Case B: SVC @ same bus as PV (local compensation) --")
    results_best_local  = svc_size_vs_pvhc(target_bus='632', svc_bus='632')
    results_worst_local = svc_size_vs_pvhc(target_bus='611', svc_bus='611')

    case4_results = {
        'PV@632, SVC@675 (remote)': results_best_remote,
        'PV@611, SVC@675 (remote)': results_worst_remote,
        'PV@632, SVC@632 (local)':  results_best_local,
        'PV@611, SVC@611 (local)':  results_worst_local,
    }

    # Report the peak of each curve, which is the practically useful number:
    # "how much SVC should you actually install to maximize PVHC here?"
    print("  Peak PVHC per curve:")
    for label, res in case4_results.items():
        valid = {q: v for q, v in res.items() if v is not None}
        if not valid:
            print(f"    {label:<28s}  no feasible SVC size found")
            continue
        best_q   = max(valid, key=valid.get)
        best_val = valid[best_q]
        zero_val = valid.get(0.0, None)
        trend = ""
        if zero_val is not None and best_q > 0:
            if best_val > zero_val:
                trend = "  (SVC helps up to this point)"
            elif best_val < zero_val:
                trend = "  (any SVC here HURTS PVHC)"
        print(f"    {label:<28s}  peak PVHC = {best_val:.3f} MW at SVC = {best_q:.2f} Mvar{trend}")
    print()

    return case4_results


def plot_case4_results(case4_results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ['o', 's', '^', 'D']
    colors  = ['#e74c3c', '#3498db', '#e67e22', '#2ecc71']

    for (label, res), marker, color in zip(case4_results.items(), markers, colors):
        qs   = [q for q, v in res.items() if v is not None]
        vals = [v for v in res.values() if v is not None]
        ax.plot(qs, vals, marker=marker, color=color, label=label, linewidth=2)

    ax.set_title('Case 4: Effect of SVC (Negative VAR) Magnitude on PVHC',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('SVC Size (Mvar)', fontsize=11)
    ax.set_ylabel('PVHC (MW)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('svc_vs_pvhc.png', dpi=150, bbox_inches='tight')
    print("  Plot saved as svc_vs_pvhc.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # Step 1: Base case validation
    run_base_case()

    # Step 2: Case 1 — PVHC at every bus
    pvhc_results, best_bus, best_pvhc = run_case1()

    # Step 3: Case 2 — interventions at highest PVHC bus
    case2_results = run_case2(best_bus, best_pvhc)

    # Step 4: Summary tables
    print("=" * 60)
    print("SUMMARY — CASE 1: PVHC PER BUS")
    print("=" * 60)
    df1 = pd.DataFrame.from_dict(pvhc_results, orient='index',
                                  columns=['PVHC (MW)'])
    df1.index.name = 'Bus'
    print(df1.to_string())
    print()

    print("=" * 60)
    print("SUMMARY — CASE 2: INTERVENTION COMPARISON (BEST BUS)")
    print("=" * 60)
    df2 = pd.DataFrame.from_dict(case2_results, orient='index',
                                  columns=['PVHC (MW)'])
    df2.index.name = 'Intervention'
    print(df2.to_string())
    print()

    # Step 5: Case 2 — interventions at lowest PVHC bus
    worst_bus, case2_lowest_results = run_case2_lowest(pvhc_results, best_pvhc)

    print("=" * 60)
    print(f"SUMMARY — CASE 2 (LOWEST PVHC BUS {worst_bus}): INTERVENTION COMPARISON")
    print("=" * 60)
    df3 = pd.DataFrame.from_dict(case2_lowest_results, orient='index',
                                  columns=['PVHC (MW)'])
    df3.index.name = 'Intervention'
    print(df3.to_string())
    print()

    # Step 6: Plots (unchanged from before)
    plot_results(pvhc_results, case2_results, best_bus,
                 case2_lowest_results, worst_bus)
    plot_schematic()

    # Step 7 (NEW): Case 3 — SVC (negative VAR) impact study at every bus
    svc_results, impact_results, best_bus_svc, worst_bus_svc, base_min_v_bus = run_case3()

    print("=" * 60)
    print("SUMMARY — CASE 3: SVC HOSTING CAPACITY PER BUS")
    print("=" * 60)
    df4 = pd.DataFrame.from_dict(svc_results, orient='index',
                                  columns=['SVC Capacity (Mvar)'])
    df4.index.name = 'Bus'
    print(df4.to_string())
    print()

    print("=" * 60)
    print("SUMMARY — CASE 3: PERFORMANCE IMPACT PER BUS")
    print(f"(system-wide weakest bus BEFORE any SVC = Bus {base_min_v_bus};"
          f" 'min_v_bus_after' shows which bus is weakest AFTER that SVC is added)")
    print("=" * 60)
    df5 = pd.DataFrame.from_dict(impact_results, orient='index')
    df5.index.name = 'Bus'
    print(df5.to_string())
    print()

    plot_case3_results(svc_results, impact_results, best_bus_svc, worst_bus_svc,
                       base_min_v_bus)

    # Step 8 (NEW): Case 4 — does SVC magnitude itself affect PVHC?
    case4_results = run_case4()
    plot_case4_results(case4_results)

    print("Done. All results printed above.")

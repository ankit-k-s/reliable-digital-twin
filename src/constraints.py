# src/constraints.py
import numpy as np

class WADIConstraintChecker:
    """
    Rule-based physical twin for WADI water distribution system.
    Encodes process physics that CANNOT be violated under normal operation.
    These rules catch attacks that fool the LSTM by being statistically 
    plausible but physically impossible.
    """

    def __init__(self, sensor_cols):
        self.sensor_cols = sensor_cols
        self.col_idx = {name.strip(): i 
                        for i, name in enumerate(sensor_cols)}
        
        # Subsystem definitions — maps zone to its critical sensors
        self.subsystems = {
            'zone1_intake': [
                '1_LT_001_PV', '1_FIT_001_PV',
                '1_MV_001_STATUS', '1_P_001_STATUS'
            ],
            'zone2_treatment': [
                '2_LT_001_PV', '2_LT_002_PV',
                '2_FIT_001_PV', '2_FIT_002_PV',
                '2_PIT_001_PV', '2_PIT_002_PV'
            ],
            'zone2_ro_a': [
                '2A_AIT_001_PV', '2A_AIT_002_PV',
                '2A_AIT_003_PV', '2A_AIT_004_PV'
            ],
            'zone2_ro_b': [
                '2B_AIT_001_PV', '2B_AIT_002_PV',
                '2B_AIT_003_PV', '2B_AIT_004_PV'
            ],
            'zone3_distribution': [
                '3_LT_001_PV', '3_FIT_001_PV',
                '3_AIT_001_PV', '3_AIT_002_PV'
            ],
            'leak_detection': [
                'LEAK_DIFF_PRESSURE'
            ]
        }

        # Criticality ranking — used by Layer 5 weighter
        # Higher = more critical to overall system operation
        self.criticality = {
            'zone1_intake':       0.6,
            'zone2_treatment':    0.9,
            'zone2_ro_a':         0.8,
            'zone2_ro_b':         0.8,
            'zone3_distribution': 1.0,  # most critical — end consumer
            'leak_detection':     0.7,
        }

    def _get(self, window, sensor_name):
        """Get last timestep value of a named sensor."""
        idx = self.col_idx.get(sensor_name.strip())
        if idx is None:
            return None
        val = window[-1, idx]
        return float(val) if hasattr(val, 'item') else float(val)

    def _get_mean(self, window, sensor_name):
        """Get mean value across the window for trend analysis."""
        idx = self.col_idx.get(sensor_name.strip())
        if idx is None:
            return None
        return float(window[:, idx].mean())

    def check(self, window):
        """
        Args:
            window: numpy array [60, 127] — one sliding window

        Returns:
            violations:       list of human-readable violation strings
            violation_score:  float 0-1, fraction of rules violated
            subsystem_flags:  dict mapping subsystem name to bool
        """
        violations = []
        subsystem_flags = {k: False for k in self.subsystems}

        # -------------------------------------------------------
        # ZONE 1 RULES — Raw water intake
        # -------------------------------------------------------

        # Rule 1: If intake valve MV_001 is closed, flow should be near zero
        mv001 = self._get(window, '1_MV_001_STATUS')
        fit001 = self._get(window, '1_FIT_001_PV')
        if mv001 is not None and fit001 is not None:
            if mv001 < 0.1 and fit001 > 0.15:
                violations.append(
                    "Z1: Intake valve closed but flow detected — possible FDI on flow sensor"
                )
                subsystem_flags['zone1_intake'] = True

        # Rule 2: If all Zone 1 pumps off, tank level should not increase
        p_statuses = [
            self._get(window, f'1_P_00{i}_STATUS') for i in range(1, 7)
        ]
        lt001_mean = self._get_mean(window, '1_LT_001_PV')
        lt001_last = self._get(window, '1_LT_001_PV')
        p_statuses = [p for p in p_statuses if p is not None]
        if p_statuses and lt001_mean is not None and lt001_last is not None:
            all_pumps_off = all(p < 0.1 for p in p_statuses)
            level_rising = lt001_last > lt001_mean + 0.05
            if all_pumps_off and level_rising:
                violations.append(
                    "Z1: All pumps OFF but tank level rising — possible FDI on level sensor"
                )
                subsystem_flags['zone1_intake'] = True

        # Rule 3: Level switch alarms should match level transmitter
        ls001 = self._get(window, '1_LS_001_AL')
        ls002 = self._get(window, '1_LS_002_AL')
        lt001 = self._get(window, '1_LT_001_PV')
        if ls001 is not None and lt001 is not None:
            # LS_001 = low level alarm, should trigger when LT < 0.2
            if ls001 > 0.5 and lt001 > 0.4:
                violations.append(
                    "Z1: Low-level alarm active but level transmitter reads normal"
                )
                subsystem_flags['zone1_intake'] = True
        if ls002 is not None and lt001 is not None:
            # LS_002 = high level alarm, should trigger when LT > 0.8
            if ls002 > 0.5 and lt001 < 0.6:
                violations.append(
                    "Z1: High-level alarm active but level transmitter reads normal"
                )
                subsystem_flags['zone1_intake'] = True

        # -------------------------------------------------------
        # ZONE 2 RULES — UF Treatment
        # -------------------------------------------------------

        # Rule 4: FIC controller output vs actual flow consistency
        # CO (controller output) and PV (process value) should correlate
        for stream in ['101', '201', '301', '401', '501', '601']:
            co = self._get(window, f'2_FIC_{stream}_CO')
            pv = self._get(window, f'2_FIC_{stream}_PV')
            if co is not None and pv is not None:
                # If controller demanding high flow but actual flow is zero
                if co > 0.7 and pv < 0.05:
                    violations.append(
                        f"Z2: FIC_{stream} controller demanding flow "
                        f"but none detected — valve may be locked"
                    )
                    subsystem_flags['zone2_treatment'] = True
                # If controller demanding zero but flow continues
                if co < 0.05 and pv > 0.3:
                    violations.append(
                        f"Z2: FIC_{stream} controller closed "
                        f"but flow continues — possible command injection"
                    )
                    subsystem_flags['zone2_treatment'] = True

        # Rule 5: Pressure consistency — differential pressure vs pump status
        p2_001 = self._get(window, '2_P_001_STATUS')
        p2_002 = self._get(window, '2_P_002_STATUS')
        pit001 = self._get(window, '2_PIT_001_PV')
        if p2_001 is not None and p2_002 is not None and pit001 is not None:
            both_pumps_off = p2_001 < 0.1 and p2_002 < 0.1
            high_pressure = pit001 > 0.7
            if both_pumps_off and high_pressure:
                violations.append(
                    "Z2: Both treatment pumps off but high pressure detected"
                )
                subsystem_flags['zone2_treatment'] = True

        # Rule 6: Level switch vs level transmitter consistency in Zone 2
        lt2_001 = self._get(window, '2_LT_001_PV')
        ls2_001_al = self._get(window, '2_LS_001_AL')
        if lt2_001 is not None and ls2_001_al is not None:
            if ls2_001_al > 0.5 and lt2_001 > 0.5:
                violations.append(
                    "Z2: Low-level switch alarm but transmitter reads half-full"
                )
                subsystem_flags['zone2_treatment'] = True

        # -------------------------------------------------------
        # ZONE 3 RULES — Distribution
        # -------------------------------------------------------

        # Rule 7: Distribution pump vs flow consistency
        p3_001 = self._get(window, '3_P_001_STATUS')
        p3_002 = self._get(window, '3_P_002_STATUS')
        fit3_001 = self._get(window, '3_FIT_001_PV')
        if p3_001 is not None and p3_002 is not None and fit3_001 is not None:
            both_dist_pumps_off = p3_001 < 0.1 and p3_002 < 0.1
            flow_present = fit3_001 > 0.15
            if both_dist_pumps_off and flow_present:
                violations.append(
                    "Z3: Distribution pumps off but outflow detected — "
                    "possible sensor spoofing or unauthorized valve open"
                )
                subsystem_flags['zone3_distribution'] = True

        # Rule 8: Tank level vs demand flow plausibility
        lt3_001 = self._get(window, '3_LT_001_PV')
        total_flow = self._get(window, 'TOTAL_CONS_REQUIRED_FLOW')
        if lt3_001 is not None and total_flow is not None:
            # If tank near empty but demand signal shows no consumption
            if lt3_001 < 0.1 and total_flow < 0.05:
                violations.append(
                    "Z3: Tank critically low but zero demand recorded — "
                    "demand signal may be suppressed"
                )
                subsystem_flags['zone3_distribution'] = True

        # Rule 9: Water quality plausibility in distribution
        # AIT sensors should be within reasonable bounds
        # Values near 0 or 1 (scaled) simultaneously = sensor manipulation
        ait_sensors = [
            '3_AIT_001_PV', '3_AIT_002_PV',
            '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV'
        ]
        ait_vals = [self._get(window, s) for s in ait_sensors]
        ait_vals = [v for v in ait_vals if v is not None]
        if ait_vals:
            # All AIT sensors simultaneously at extremes = coordinated attack
            all_at_zero = all(v < 0.02 for v in ait_vals)
            all_at_max = all(v > 0.98 for v in ait_vals)
            if all_at_zero or all_at_max:
                violations.append(
                    "Z3: All quality sensors simultaneously at boundary — "
                    "coordinated sensor manipulation suspected"
                )
                subsystem_flags['zone3_distribution'] = True

        # -------------------------------------------------------
        # LEAK DETECTION RULES
        # -------------------------------------------------------

        # Rule 10: Abnormal differential pressure
        leak_dp = self._get(window, 'LEAK_DIFF_PRESSURE')
        if leak_dp is not None:
            if leak_dp > 0.9:
                violations.append(
                    "LEAK: Differential pressure critically high — "
                    "possible pipe rupture or valve manipulation"
                )
                subsystem_flags['leak_detection'] = True

        # -------------------------------------------------------
        # Compute violation score
        # -------------------------------------------------------
        total_rules = 10
        violation_score = len(violations) / total_rules
        # Cap at 1.0
        violation_score = min(violation_score, 1.0)

        return violations, violation_score, subsystem_flags

    def get_criticality(self):
        """Returns the criticality ranking dict for Layer 5."""
        return self.criticality

    def get_subsystem_sensors(self):
        """Returns the subsystem-to-sensor mapping for Layer 5."""
        return self.subsystems


if __name__ == "__main__":
    # Quick test with dummy window
    import numpy as np
    dummy_cols = []
    import pandas as pd
    df = pd.read_csv('data/WADI_14days_new.csv', nrows=2, low_memory=False)
    cols = [c for c in df.columns 
            if c.strip() not in ['Row', 'Date', 'Time']]
    
    checker = WADIConstraintChecker(cols)
    
    # Normal window — all zeros (everything off, no flow)
    normal_window = np.zeros((60, 127))
    violations, score, flags = checker.check(normal_window)
    print(f"Normal window violations: {len(violations)}, score: {score:.3f}")
    
    # Simulated attack — pump off but flow sensor reports high flow
    attack_window = np.zeros((60, 127))
    # Set pump to off (already 0) but flow to high
    flow_idx = checker.col_idx.get('1_FIT_001_PV')
    valve_idx = checker.col_idx.get('1_MV_001_STATUS')
    if flow_idx and valve_idx:
        attack_window[:, flow_idx] = 0.8   # high flow
        attack_window[:, valve_idx] = 0.0  # valve closed
    
    violations, score, flags = checker.check(attack_window)
    print(f"Attack window violations: {len(violations)}, score: {score:.3f}")
    for v in violations:
        print(f"  -> {v}")
"""
Bilateration Implementation for Indoor Positioning using WiFi Sensing
VS Code Ready Version - Works without scene files
Combines CSI (Channel State Information) and FTM (Fine Timing Measurement)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares

print("✓ All imports successful!")


# ============================================================================
# 1. BILATERATION CORE ALGORITHMS
# ============================================================================

class BilaterationSolver:
    """
    Bilateration positioning using distance and angle measurements
    Combines range (from FTM) and angle (from CSI) information
    """
    
    @staticmethod
    def bilateration_2d_geometric(anchor1_pos, anchor2_pos, 
                                   distance1, distance2,
                                   angle1=None, angle2=None):
        """
        Pure geometric bilateration in 2D
        
        Args:
            anchor1_pos: Position of anchor 1 [x, y]
            anchor2_pos: Position of anchor 2 [x, y]
            distance1: Distance from anchor 1
            distance2: Distance from anchor 2
            angle1: Optional AoA from anchor 1 (radians)
            angle2: Optional AoA from anchor 2 (radians)
            
        Returns:
            position: Estimated position [x, y]
        """
        x1, y1 = anchor1_pos
        x2, y2 = anchor2_pos
        d1, d2 = distance1, distance2
        
        # Distance between anchors
        d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Check if solution exists
        if d > d1 + d2 or d < abs(d1 - d2):
            # No intersection - use least squares
            return BilaterationSolver._fallback_least_squares_2d(
                [anchor1_pos, anchor2_pos], [d1, d2]
            )
        
        # Find intersection points
        a = (d1**2 - d2**2 + d**2) / (2 * d)
        h_sq = d1**2 - a**2
        if h_sq < 0:
            h_sq = 0
        h = np.sqrt(h_sq)
        
        # Point on line between anchors
        px = x1 + a * (x2 - x1) / d
        py = y1 + a * (y2 - y1) / d
        
        # Two possible positions
        pos1 = np.array([
            px + h * (y2 - y1) / d,
            py - h * (x2 - x1) / d
        ])
        
        pos2 = np.array([
            px - h * (y2 - y1) / d,
            py + h * (x2 - x1) / d
        ])
        
        # If angles provided, use them to disambiguate
        if angle1 is not None or angle2 is not None:
            return BilaterationSolver._select_position_with_angle(
                pos1, pos2, anchor1_pos, anchor2_pos, angle1, angle2
            )
        
        return pos1
    
    @staticmethod
    def _select_position_with_angle(pos1, pos2, anchor1_pos, anchor2_pos, 
                                    angle1, angle2):
        """Select correct position using angle information"""
        scores = [0, 0]
        
        for idx, pos in enumerate([pos1, pos2]):
            if angle1 is not None:
                direction = pos - anchor1_pos
                measured_angle = np.arctan2(direction[1], direction[0])
                angle_error = abs(measured_angle - angle1)
                angle_error = min(angle_error, 2*np.pi - angle_error)
                scores[idx] -= angle_error
            
            if angle2 is not None:
                direction = pos - anchor2_pos
                measured_angle = np.arctan2(direction[1], direction[0])
                angle_error = abs(measured_angle - angle2)
                angle_error = min(angle_error, 2*np.pi - angle_error)
                scores[idx] -= angle_error
        
        return pos1 if scores[0] > scores[1] else pos2
    
    @staticmethod
    def _fallback_least_squares_2d(anchor_positions, distances):
        """Least squares fallback when geometric solution fails"""
        def objective(pos):
            estimated_distances = np.linalg.norm(
                np.array(anchor_positions) - pos, axis=1
            )
            return np.sum((np.array(distances) - estimated_distances)**2)
        
        x0 = np.mean(anchor_positions, axis=0)
        result = minimize(objective, x0, method='BFGS')
        return result.x
    
    @staticmethod
    def bilateration_3d(anchor1_pos, anchor2_pos, anchor3_pos,
                        distance1, distance2, distance3,
                        angle1=None, angle2=None, angle3=None):
        """
        3D bilateration with three anchors
        
        Args:
            anchor1_pos, anchor2_pos, anchor3_pos: Anchor positions [x, y, z]
            distance1, distance2, distance3: Distances from anchors
            angle1, angle2, angle3: Optional AoA measurements (azimuth)
            
        Returns:
            position: Estimated 3D position [x, y, z]
        """
        anchors = np.array([anchor1_pos, anchor2_pos, anchor3_pos])
        distances = np.array([distance1, distance2, distance3])
        
        def objective(pos):
            estimated_distances = np.linalg.norm(anchors - pos, axis=1)
            residual = distances - estimated_distances
            
            if angle1 is not None or angle2 is not None or angle3 is not None:
                angles = [angle1, angle2, angle3]
                angle_penalty = 0
                
                for i, angle in enumerate(angles):
                    if angle is not None:
                        direction = pos[:2] - anchors[i, :2]
                        measured_angle = np.arctan2(direction[1], direction[0])
                        angle_error = abs(measured_angle - angle)
                        angle_error = min(angle_error, 2*np.pi - angle_error)
                        angle_penalty += angle_error * 2
                
                return np.concatenate([residual, [angle_penalty]])
            
            return residual
        
        x0 = np.mean(anchors, axis=0)
        result = least_squares(objective, x0, method='trf')
        return result.x
    
    @staticmethod
    def hybrid_range_angle_bilateration(anchor_positions, distances, angles,
                                        distance_std=0.5, angle_std=0.1):
        """
        Weighted hybrid bilateration using both range and angle
        
        Args:
            anchor_positions: Positions of anchors [num_anchors, 2/3]
            distances: Distance measurements [num_anchors]
            angles: Angle measurements [num_anchors] (radians)
            distance_std: Standard deviation of distance measurements
            angle_std: Standard deviation of angle measurements
            
        Returns:
            position: Estimated position
            covariance: Position covariance matrix
        """
        num_anchors = len(anchor_positions)
        dim = anchor_positions.shape[1]
        
        # Weights based on measurement accuracy
        distance_weight = 1 / (distance_std**2)
        angle_weight = 1 / (angle_std**2)
        
        def weighted_objective(pos):
            error = 0
            
            # Distance errors
            for i in range(num_anchors):
                estimated_dist = np.linalg.norm(pos - anchor_positions[i])
                error += distance_weight * (distances[i] - estimated_dist)**2
            
            # Angle errors
            for i in range(num_anchors):
                if not np.isnan(angles[i]):
                    direction = pos[:2] - anchor_positions[i, :2]
                    estimated_angle = np.arctan2(direction[1], direction[0])
                    angle_diff = angles[i] - estimated_angle
                    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
                    error += angle_weight * angle_diff**2
            
            return error
        
        x0 = np.mean(anchor_positions, axis=0)
        result = minimize(weighted_objective, x0, method='BFGS')
        position = result.x
        
        # Simplified covariance estimate
        covariance = np.eye(dim) * (distance_std**2)
        
        return position, covariance


# ============================================================================
# 2. CSI-BASED ANGLE OF ARRIVAL ESTIMATION
# ============================================================================

class CSIAngleEstimator:
    """Extract Angle of Arrival from CSI measurements"""
    
    def __init__(self, num_antennas, antenna_spacing, wavelength):
        """
        Initialize AoA estimator
        
        Args:
            num_antennas: Number of antenna elements
            antenna_spacing: Spacing between elements (meters)
            wavelength: Signal wavelength (meters)
        """
        self.num_antennas = num_antennas
        self.antenna_spacing = antenna_spacing
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength
    
    def steering_vector(self, angle):
        """Compute array steering vector for given angle"""
        positions = np.arange(self.num_antennas) * self.antenna_spacing
        phase_shifts = self.k * positions * np.sin(angle)
        return np.exp(1j * phase_shifts)
    
    def estimate_aoa_music(self, csi_matrix, num_sources=1, 
                           angle_search_range=(-np.pi/2, np.pi/2),
                           num_angles=180):
        """
        MUSIC algorithm for AoA estimation from CSI
        
        Args:
            csi_matrix: CSI from antenna array [num_antennas, num_subcarriers]
            num_sources: Number of signal sources
            angle_search_range: Range of angles to search
            num_angles: Number of angle grid points
            
        Returns:
            estimated_angle: Estimated AoA in radians
            spectrum: MUSIC spectrum for visualization
        """
        # Average CSI across subcarriers
        avg_csi = np.mean(csi_matrix, axis=1)
        
        # Covariance matrix
        R = np.outer(avg_csi, avg_csi.conj())
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Noise subspace
        noise_subspace = eigenvectors[:, num_sources:]
        
        # Angular search
        angles = np.linspace(angle_search_range[0], angle_search_range[1], num_angles)
        spectrum = np.zeros(num_angles)
        
        for i, angle in enumerate(angles):
            a = self.steering_vector(angle)
            spectrum[i] = 1.0 / (np.abs(
                a.conj() @ noise_subspace @ noise_subspace.conj().T @ a
            ) + 1e-10)
        
        # Find peak
        peak_idx = np.argmax(spectrum)
        estimated_angle = angles[peak_idx]
        
        return estimated_angle, (angles, spectrum)


# ============================================================================
# 3. FTM-BASED RANGE ESTIMATION
# ============================================================================

class FTMRangeEstimator:
    """Extract range from Fine Timing Measurements"""
    
    def __init__(self, speed_of_light=3e8, bandwidth=20e6):
        """Initialize FTM range estimator"""
        self.c = speed_of_light
        self.bandwidth = bandwidth
        self.time_resolution = 1 / bandwidth
    
    def estimate_range_from_phase_slope(self, csi, subcarrier_spacing):
        """
        Estimate range from CSI phase slope across subcarriers
        
        Args:
            csi: CSI measurements [num_subcarriers]
            subcarrier_spacing: Spacing between subcarriers in Hz
            
        Returns:
            estimated_range: Range in meters
        """
        # Extract phase
        phase = np.unwrap(np.angle(csi))
        
        # Subcarrier indices
        subcarrier_idx = np.arange(len(phase))
        
        # Linear regression to find phase slope
        coeffs = np.polyfit(subcarrier_idx * subcarrier_spacing, phase, 1)
        phase_slope = coeffs[0]
        
        # Range from phase slope
        toa = -phase_slope / (2 * np.pi)
        estimated_range = abs(self.c * toa / 2)
        
        return estimated_range


# ============================================================================
# 4. VISUALIZATION
# ============================================================================

class BilaterationVisualizer:
    """Visualization tools for bilateration results"""
    
    @staticmethod
    def plot_2d_localization(anchor_positions, estimated_position, 
                            true_position=None, ranges=None, angles=None,
                            title="Bilateration Result"):
        """Plot 2D visualization of localization result"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Extract 2D coordinates
        anchors_2d = anchor_positions[:, :2]
        est_2d = estimated_position[:2]
        
        # Plot anchors
        ax.scatter(anchors_2d[:, 0], anchors_2d[:, 1], 
                  s=200, c='blue', marker='^', label='Anchors', 
                  edgecolors='black', linewidths=2)
        
        # Label anchors
        for i, pos in enumerate(anchors_2d):
            ax.text(pos[0], pos[1] + 0.3, f'AP{i+1}', 
                   ha='center', fontsize=10, fontweight='bold')
        
        # Plot range circles
        if ranges is not None:
            for i, (pos, r) in enumerate(zip(anchors_2d, ranges)):
                circle = plt.Circle(pos, r, fill=False, 
                                   linestyle='--', color='blue', alpha=0.3)
                ax.add_patch(circle)
        
        # Plot angle rays
        if angles is not None:
            ray_length = 5
            for i, (pos, angle) in enumerate(zip(anchors_2d, angles)):
                end_x = pos[0] + ray_length * np.cos(angle)
                end_y = pos[1] + ray_length * np.sin(angle)
                ax.plot([pos[0], end_x], [pos[1], end_y], 
                       'b--', alpha=0.3, linewidth=1)
        
        # Plot estimated position
        ax.scatter(est_2d[0], est_2d[1], 
                  s=300, c='red', marker='*', label='Estimated', 
                  edgecolors='black', linewidths=2)
        
        # Plot true position
        if true_position is not None:
            true_2d = true_position[:2]
            ax.scatter(true_2d[0], true_2d[1], 
                      s=300, c='green', marker='o', label='True', 
                      edgecolors='black', linewidths=2)
            
            error = np.linalg.norm(est_2d - true_2d)
            ax.plot([est_2d[0], true_2d[0]], [est_2d[1], true_2d[1]], 
                   'r--', linewidth=2, label=f'Error: {error:.2f}m')
        
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        return fig


# ============================================================================
# 5. EXAMPLE SIMULATION
# ============================================================================

def example_bilateration_simulation():
    """Complete example: Bilateration using CSI and FTM"""
    
    print("\n" + "=" * 70)
    print("WiFi Bilateration with CSI and FTM")
    print("=" * 70)
    
    # Simulate anchor positions (square arrangement)
    anchor_positions = np.array([
        [0, 0, 2.5],
        [10, 0, 2.5],
        [10, 10, 2.5],
        [0, 10, 2.5]
    ])
    
    # True target position
    true_position = np.array([6.5, 4.0, 1.5])
    
    print(f"\nAnchors: {len(anchor_positions)} APs")
    print(f"True target position: {true_position}")
    
    # Initialize estimators
    wavelength = 3e8 / 2.4e9  # 2.4 GHz
    
    print("\n" + "-" * 70)
    print("Simulating Measurements")
    print("-" * 70)
    
    measurements = {}
    
    for i, anchor_pos in enumerate(anchor_positions):
        # True range
        true_range = np.linalg.norm(true_position - anchor_pos)
        
        # True angle
        direction = true_position[:2] - anchor_pos[:2]
        true_angle = np.arctan2(direction[1], direction[0])
        
        # Add noise
        range_noise = np.random.randn() * 0.3
        angle_noise = np.random.randn() * np.deg2rad(5)
        
        measured_range = true_range + range_noise
        measured_angle = true_angle + angle_noise
        
        measurements[f'anchor_{i}'] = {
            'range': measured_range,
            'angle': measured_angle,
            'anchor_position': anchor_pos,
            'true_range': true_range,
            'true_angle': true_angle
        }
        
        print(f"Anchor {i+1}: Range={measured_range:.2f}m (true={true_range:.2f}m), "
              f"Angle={np.rad2deg(measured_angle):.1f}° (true={np.rad2deg(true_angle):.1f}°)")
    
    # Perform localization
    print("\n" + "-" * 70)
    print("Performing Localization")
    print("-" * 70)
    
    solver = BilaterationSolver()
    
    # Extract data
    ranges = np.array([m['range'] for m in measurements.values()])
    angles = np.array([m['angle'] for m in measurements.values()])
    
    # Method 1: Hybrid (range + angle)
    print("\n[1] Hybrid Method (Range + Angle):")
    pos_hybrid, cov_hybrid = solver.hybrid_range_angle_bilateration(
        anchor_positions[:, :2],
        ranges,
        angles,
        distance_std=0.3,
        angle_std=np.deg2rad(5)
    )
    pos_hybrid_3d = np.append(pos_hybrid, true_position[2])
    error_hybrid = np.linalg.norm(pos_hybrid_3d - true_position)
    print(f"    Estimated: {pos_hybrid_3d}")
    print(f"    Error: {error_hybrid:.3f} m")
    
    # Method 2: Range only
    print("\n[2] Range-Only Method (FTM):")
    pos_range = solver.bilateration_3d(
        anchor_positions[0],
        anchor_positions[1],
        anchor_positions[2],
        ranges[0], ranges[1], ranges[2]
    )
    error_range = np.linalg.norm(pos_range - true_position)
    print(f"    Estimated: {pos_range}")
    print(f"    Error: {error_range:.3f} m")
    
    # Visualization
    print("\n" + "-" * 70)
    print("Generating Visualization")
    print("-" * 70)
    
    visualizer = BilaterationVisualizer()
    
    fig = visualizer.plot_2d_localization(
        anchor_positions,
        pos_hybrid_3d,
        true_position,
        ranges=ranges,
        angles=angles,
        title="Hybrid Bilateration (CSI + FTM)"
    )
    
    # Monte Carlo simulation
    print("\n" + "-" * 70)
    print("Monte Carlo Simulation (100 trials)")
    print("-" * 70)
    
    num_trials = 100
    errors_hybrid = []
    errors_range = []
    
    for trial in range(num_trials):
        noisy_ranges = ranges + np.random.randn(len(ranges)) * 0.3
        noisy_angles = angles + np.random.randn(len(angles)) * np.deg2rad(5)
        
        # Hybrid
        pos_h, _ = solver.hybrid_range_angle_bilateration(
            anchor_positions[:, :2],
            noisy_ranges,
            noisy_angles,
            distance_std=0.3,
            angle_std=np.deg2rad(5)
        )
        pos_h_3d = np.append(pos_h, true_position[2])
        errors_hybrid.append(np.linalg.norm(pos_h_3d - true_position))
        
        # Range-only
        pos_r = solver.bilateration_3d(
            anchor_positions[0],
            anchor_positions[1],
            anchor_positions[2],
            noisy_ranges[0], noisy_ranges[1], noisy_ranges[2]
        )
        errors_range.append(np.linalg.norm(pos_r - true_position))
    
    errors_hybrid = np.array(errors_hybrid)
    errors_range = np.array(errors_range)
    
    print(f"\nHybrid Method:")
    print(f"  Mean error: {np.mean(errors_hybrid):.3f} m")
    print(f"  Median: {np.median(errors_hybrid):.3f} m")
    print(f"  90th percentile: {np.percentile(errors_hybrid, 90):.3f} m")
    
    print(f"\nRange-Only Method:")
    print(f"  Mean error: {np.mean(errors_range):.3f} m")
    print(f"  Median: {np.median(errors_range):.3f} m")
    print(f"  90th percentile: {np.percentile(errors_range, 90):.3f} m")
    
    improvement = (np.mean(errors_range) - np.mean(errors_hybrid)) / np.mean(errors_range) * 100
    print(f"\nImprovement: {improvement:.1f}%")
    
    # Plot CDF
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    sorted_hybrid = np.sort(errors_hybrid)
    sorted_range = np.sort(errors_range)
    cdf = np.arange(1, len(sorted_hybrid) + 1) / len(sorted_hybrid)
    
    ax.plot(sorted_hybrid, cdf, 'b-', linewidth=2, label='Hybrid (CSI + FTM)')
    ax.plot(sorted_range, cdf, 'r--', linewidth=2, label='Range Only (FTM)')
    
    ax.set_xlabel('Localization Error (meters)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Localization Error Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)
    
    plt.show()
    
    return {
        'true_position': true_position,
        'estimated_hybrid': pos_hybrid_3d,
        'estimated_range': pos_range,
        'error_hybrid': error_hybrid,
        'error_range': error_range
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = example_bilateration_simulation()
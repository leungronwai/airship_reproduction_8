"""
Simulation performance evaluation module
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """
    Performance evaluation class for airship control simulation
    """
    
    def __init__(self):
        """Initialize performance evaluator"""
        self.metrics = {}
    
    def calculate_position_metrics(self, error_history):
        """
        Calculate position tracking performance metrics
        
        Args:
            error_history: Error history array
            
        Returns:
            dict: Position performance metrics
        """
        pos_errors = error_history[0:3, :]
        
        metrics = {
            'rmse': np.sqrt(np.mean(pos_errors**2, axis=1)),
            'max_error': np.max(np.abs(pos_errors), axis=1),
            'mean_error': np.mean(np.abs(pos_errors), axis=1),
            'std_error': np.std(pos_errors, axis=1)
        }
        
        return metrics
    
    def calculate_attitude_metrics(self, error_history):
        """
        Calculate attitude tracking performance metrics
        
        Args:
            error_history: Error history array
            
        Returns:
            dict: Attitude performance metrics
        """
        att_errors = error_history[3:6, :]
        
        metrics = {
            'rmse': np.sqrt(np.mean(att_errors**2, axis=1)),
            'max_error': np.max(np.abs(att_errors), axis=1),
            'mean_error': np.mean(np.abs(att_errors), axis=1),
            'std_error': np.std(att_errors, axis=1)
        }
        
        return metrics
    
    def calculate_control_metrics(self, control_history):
        """
        Calculate control input performance metrics
        
        Args:
            control_history: Control input history array
            
        Returns:
            dict: Control performance metrics
        """
        metrics = {
            'mean': np.mean(control_history, axis=1),
            'std': np.std(control_history, axis=1),
            'max': np.max(np.abs(control_history), axis=1),
            'total_variation': np.sum(np.abs(np.diff(control_history, axis=1)), axis=1)
        }
        
        return metrics
    
    def calculate_steady_state_metrics(self, error_history, sim_time, steady_state_ratio=0.8):
        """
        Calculate steady-state performance metrics
        
        Args:
            error_history: Error history array
            sim_time: Simulation time array
            steady_state_ratio: Ratio to determine steady-state start
            
        Returns:
            dict: Steady-state metrics
        """
        steady_start = int(steady_state_ratio * len(sim_time))
        
        pos_errors = error_history[0:3, steady_start:]
        att_errors = error_history[3:6, steady_start:]
        
        metrics = {
            'position': {
                'mean_error': np.mean(np.abs(pos_errors), axis=1),
                'max_error': np.max(np.abs(pos_errors), axis=1)
            },
            'attitude': {
                'mean_error': np.mean(np.abs(att_errors), axis=1),
                'max_error': np.max(np.abs(att_errors), axis=1)
            }
        }
        
        return metrics
    
    def evaluate_performance(self, results):
        """
        Comprehensive performance evaluation
        
        Args:
            results: Simulation results dictionary
            
        Returns:
            dict: Complete performance metrics
        """
        logger.info("=== Control Performance Evaluation ===")
        
        error_history = results['errors']
        control_history = results['controls']
        sim_time = results['time']
        
        # Calculate all metrics
        position_metrics = self.calculate_position_metrics(error_history)
        attitude_metrics = self.calculate_attitude_metrics(error_history)
        control_metrics = self.calculate_control_metrics(control_history)
        steady_state_metrics = self.calculate_steady_state_metrics(error_history, sim_time)
        
        # Log results
        self._log_performance_results(position_metrics, attitude_metrics, control_metrics, steady_state_metrics)
        
        # Compile all metrics
        all_metrics = {
            'position': position_metrics,
            'attitude': attitude_metrics,
            'control': control_metrics,
            'steady_state': steady_state_metrics
        }
        
        return all_metrics
    
    def _log_performance_results(self, position_metrics, attitude_metrics, control_metrics, steady_state_metrics):
        """Log performance evaluation results"""
        
        # Position metrics
        logger.info("Position RMSE: X=%.3fm, Y=%.3fm, Z=%.3fm", 
                   position_metrics['rmse'][0], position_metrics['rmse'][1], position_metrics['rmse'][2])
        logger.info("Position max error: X=%.3fm, Y=%.3fm, Z=%.3fm",
                   position_metrics['max_error'][0], position_metrics['max_error'][1], position_metrics['max_error'][2])
        
        # Attitude metrics
        logger.info("Attitude RMSE: φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                   np.rad2deg(attitude_metrics['rmse'][0]), 
                   np.rad2deg(attitude_metrics['rmse'][1]), 
                   np.rad2deg(attitude_metrics['rmse'][2]))
        logger.info("Attitude max error: φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                   np.rad2deg(attitude_metrics['max_error'][0]), 
                   np.rad2deg(attitude_metrics['max_error'][1]), 
                   np.rad2deg(attitude_metrics['max_error'][2]))
        
        # Control metrics
        logger.info("Control input mean: T=%.3fN, μ=%.3f°, ν=%.3f°",
                   control_metrics['mean'][0], 
                   np.rad2deg(control_metrics['mean'][1]), 
                   np.rad2deg(control_metrics['mean'][2]))
        logger.info("Control input standard deviation: T=%.3fN, μ=%.3f°, ν=%.3f°",
                   control_metrics['std'][0], 
                   np.rad2deg(control_metrics['std'][1]), 
                   np.rad2deg(control_metrics['std'][2]))
        
        # Steady-state metrics
        ss_pos = steady_state_metrics['position']['mean_error']
        ss_att = steady_state_metrics['attitude']['mean_error']
        logger.info("Steady-state position error: X=%.3fm, Y=%.3fm, Z=%.3fm",
                   ss_pos[0], ss_pos[1], ss_pos[2])
        logger.info("Steady-state attitude error: φ=%.3f°, θ=%.3f°, ψ=%.3f°",
                   np.rad2deg(ss_att[0]), np.rad2deg(ss_att[1]), np.rad2deg(ss_att[2]))


def evaluate_performance(results):
    """
    Convenient function to evaluate simulation performance
    
    Args:
        results: Simulation results dictionary
        
    Returns:
        dict: Performance metrics
    """
    evaluator = PerformanceEvaluator()
    return evaluator.evaluate_performance(results)
